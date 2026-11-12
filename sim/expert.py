"""The shared arm-motion base of the scripted truss picker (truss_expert.py, truss_ik.py, rrt.py).

    ik = arm_ik()                  ArmIK on the bare arm at the scene's base pose (built once per process)
    plan = Plan(ref, servo, phases, meta)
    while not plan.done:
        robot.send_action(plan.act(robot._present_norm()))

Plan plays a reference trajectory frame by frame the way a person on the leader would: the commanded joint
targets in use_degrees=false units. On servo frames (the slide onto the stalk, the close) it corrects the servo
sag the way a person does by eye: an integral term on the joint error (SAG_KI), so the fingers end up where IK put
them and not a few mm low. min_jerk times every move (smooth, rest to rest).

Started as the single-fruit picker (K7, 2026-10-27; removed 2026-11-08, task "truss" only since). What's left is
what truss_expert.py builds on: ArmIK / arm_ik, Plan, min_jerk, FPS and the timing constants (CLOSE_NORM,
JOINT_PEAK_NORM_S, CART_PEAK_M_S, SPEED, HOLD_START_S, HOLD_END_S). ⚠️ Change those and every truss episode changes.
generate_truss.py / dagger_truss.py reach arm_map through this module (expert.arm_map).
"""
import mujoco
import numpy as np

import arm_map
import scene

FPS = 30                           # config/hardware.env: what every dataset records at
# Close command, normalised 0..100 (START_NORM has 50): the jaw fully closed (norm 0 = the -10 deg stop), drawn
# per episode. K7 (2026-10-27): use the sim's own stall angle for now. ⚠️ A position servo commanded TO the stall
# angle doesn't squeeze; a person closes the leader past contact. The jaw stalls on what it holds by itself:
# that's what gripper.pos (the state) will read. Redo all episodes once the real follower's gripper.pos on a
# stalk is measured.
CLOSE_NORM = (0.0, 2.0)
# --- timing ---
# Peak joint speed in normalised units/s (1 unit ~ 1.5-1.8 deg). ~60 = ~100 deg/s peak, a calm teleop
# pace (guess). Minimum-jerk peaks at 1.875x the average speed.
JOINT_PEAK_NORM_S = 60.0
CART_PEAK_M_S = 0.08               # straight-line legs (the slide, the retreat), guess
SPEED = (0.8, 1.25)                # per-episode speed factor (domain randomisation of the motion)
HOLD_START_S = (0.4, 0.8)          # still at the start / end pose before / after the motion (guess)
HOLD_END_S = (0.5, 1.0)
# --- servo sag correction ---
SAG_KI = 0.15                      # per frame, on (reference - measured) in norm units (tuned in the sim)
SAG_MAX_NORM = 6.0                 # clamp on the integral, norm units (tuned in the sim)


def min_jerk(n):
    t = np.linspace(0, 1, max(int(n), 2))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class ArmIK:
    """IK on the bare arm at the scene's base pose: a point in the gripper frame to a world point, with the fingers'
    pitch and the jaw's tilt set. Gripper frame: fingers along -z, jaw closes along x (check_truss_cut.find_pose).
    truss_ik.TrussIK and rrt.py build on it."""

    W_ANG = 0.05                   # m per unit of the direction residuals

    def __init__(self, n_samples=8000):
        spec = mujoco.MjSpec()
        yaw = np.radians(scene.ARM_YAW_DEG)
        spec.attach(scene.arm_spec(), frame=spec.worldbody.add_frame(
            pos=scene.ARM_BASE, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), prefix="arm/")
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.q, _ = scene.arm_ids(self.m)
        self.G = self.m.body("arm/gripper").id
        lo, hi = arm_map.FollowerMap().rad_limits()
        self.lo, self.hi = lo[:5], hi[:5]
        rng = np.random.default_rng(0)
        self.Q = rng.uniform(self.lo, self.hi, (n_samples, 5))
        self.P, self.R = zip(*(self.fk(x) for x in self.Q))
        self.P, self.R = np.array(self.P), np.array(self.R)

    def fk(self, x):
        self.d.qpos[self.q[:5]] = x
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.G].copy(), self.d.xmat[self.G].reshape(3, 3).copy()

    def _res(self, x, p, point, fz, jz):
        o, R = self.fk(x)
        return np.r_[p - (o + R @ point), self.W_ANG * (fz + R[2, 2]), self.W_ANG * (jz - R[2, 0])]

    def solve(self, p, pitch_deg, tilt_deg, point, x0=None, iters=80):
        """Joint angles (5, rad) or None. x0: try from here first (keeps a path continuous), then from the
        nearest sampled poses. ⚠️ Warm-start alone failed on reachable points (local minima at the joint
        limits; 2026-10-27 scan: reachable at 7 cm back but not 5), hence the fallback."""
        fz, jz = -np.sin(np.radians(pitch_deg)), np.sin(np.radians(tilt_deg))
        g = self.P + np.einsum("nij,j->ni", self.R, point)
        cost = (np.linalg.norm(g - p, axis=1) + self.W_ANG * (np.abs(fz + self.R[:, 2, 2])
                                                               + np.abs(jz - self.R[:, 2, 0])))
        starts = ([np.asarray(x0, float)] if x0 is not None else []) + [self.Q[i] for i in np.argsort(cost)[:8]]
        for x in starts:
            x, e = self._lm(x.copy(), p, point, fz, jz, iters)
            if np.linalg.norm(e[:3]) < 0.002 and np.all(np.abs(e[3:]) < self.W_ANG * 0.05):
                return x
        return None

    def _lm(self, x, p, point, fz, jz, iters):
        """Levenberg-Marquardt inside the joint limits."""
        e = self._res(x, p, point, fz, jz)
        lam = 1e-3
        for _ in range(iters):
            if np.linalg.norm(e) < 2e-4:
                break
            J = np.empty((5, 5))
            for k in range(5):
                dx = np.zeros(5)
                dx[k] = 1e-5
                J[:, k] = (self._res(x + dx, p, point, fz, jz) - e) / 1e-5
            H, gr = J.T @ J, J.T @ e
            xn = np.clip(x - np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), gr), self.lo, self.hi)
            en = self._res(xn, p, point, fz, jz)
            if np.linalg.norm(en) < np.linalg.norm(e):
                x, e, lam = xn, en, max(lam / 3, 1e-7)
            else:
                lam *= 4
                if lam > 1e4:
                    break
        return x, e

    def frame(self, x, point):
        """Grasp point (world) and gripper rotation at joint angles x."""
        o, R = self.fk(x)
        return o + R @ point, R


_IK = None


def arm_ik():
    global _IK
    if _IK is None:
        _IK = ArmIK()
    return _IK


class Plan:
    """Reference trajectory (norm units, one row per frame) + the servo-sag correction at play time."""

    def __init__(self, ref, servo, phases, meta):
        self.ref, self.servo, self.phases, self.meta = ref, servo, phases, meta
        self.reset()

    def reset(self):
        self.k = 0
        self.I = np.zeros(5)

    @property
    def done(self):
        return self.k >= len(self.ref)

    def act(self, state_norm):
        """Action for this frame given the measured state (norm, 6). Integral sag correction on servo frames;
        it bleeds off again elsewhere, so the start/end poses are commanded exactly."""
        r = self.ref[self.k]
        if self.servo[self.k]:
            self.I = np.clip(self.I + SAG_KI * (r[:5] - np.asarray(state_norm)[:5]), -SAG_MAX_NORM, SAG_MAX_NORM)
        else:
            self.I *= 0.9
        self.k += 1
        a = r.copy()
        a[:5] += self.I
        return {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, a)}
