"""The scripted picker: plans one pick-and-place for the scene's target fruit, then plays it frame by frame.

    plan = Planner(robot).plan(rng, twist=False, near_miss=False)   # after the scene's first frame
    while not plan.done:
        robot.send_action(plan.act(robot))

Replaces the person on the leader arm for scripted episodes (K7, 2026-10-27). The motion, start to end:
hold at the start pose → swing to a pre-grasp point in front of the fruit (jaw opening on the way) →
straight-line approach along the fingers → close → pull it off (or twist, then pull) → lift → carry to
above the crate → open → back to the start pose → hold. All moves are minimum-jerk (smooth, rest to rest).

The SO-101 has 5 arm joints, so the approach direction is set by the pan (the fingers point away from the
base); what's chosen per episode is the fingers' PITCH (level to pointing down) and the jaw's TILT (wrist
roll, kept within JAW_TILT_DEG of level so a finger doesn't run into the pedicel or the truss). Every leg
is swept for collisions against the scene first (kinematically, with a clearance margin); a target with
no clean approach makes the plan fail, and the generator logs and skips that scene.

Actions are what a leader would send: the commanded joint targets in use_degrees=false units. During the
approach and the grasp the picker corrects the servo sag the way a person does by eye: an integral term
on the joint error (SAG_KI), so the fingers end up where IK put them and not a few mm low.
"""
import copy
import json

import mujoco
import numpy as np

import arm_map
import scene

CHAIN = json.loads((scene.FINRAY / "chain.json").read_text())
FPS = 30                           # config/hardware.env: what every dataset records at

# --- grasp geometry (gripper frame, m) ---
# Fruit centre below the gripper frame when held: check_grasp.HEIGHTS base -55 / mid -72 / tip -90 mm.
# Drawn between near-base and mid-finger: all 36 check_grasp cases held there (README table).
GRASP_DEPTH_M = (-0.078, -0.062)
GAP_FIXED_M = 0.003                # fruit skin to the fixed finger when the jaw starts closing (check_grasp off_x)
# Pre-grasp point this far back from the fruit along the approach (guess). The vines hang only ~9-12 cm in
# front of the shoulder, so 6-8 cm straight back along the fingers often pulls the hand into the arm's own
# reach limit (no IK at the pre-grasp: 397 of ~600 candidates, 2026-10-27). Shorter back-offs are tried next.
PREGRASP_M = (0.05, 0.07)
PREGRASP_FALLBACK_M = (0.04, 0.03)
# Approach direction: along the fingers, or tilted toward the gripper's y axis (out of the jaw plane),
# where the gap between the fingers is open too, so the fruit slides in from above / below between them.
# Tried in random order per candidate; also varies the approach angle (K7: "different orientations").
APPROACH_BLEND_DEG = (0.0, 30.0, -30.0, 60.0, -60.0)
APPROACH_STEPS = 12                # IK points along the straight approach / pull lines
# Fingers: 0 = level, + = pointing down. What's reachable depends on the fruit's height (IK scan of 15
# targets, 2026-10-27): fruit at z ~0.30 m only level-to-upward (-40..+20), at ~0.15-0.20 m only downward
# (+10..+70). So each target's reachable pitches are scanned first (PITCH_SCAN_DEG, jaw level) and the
# pitch is drawn among them. Fingers pointing up more than 25 deg is left out (reaching under the fruit).
PITCH_SCAN_DEG = (-25.0, 70.0, 5.0)
JAW_TILT_DEG = 25.0                # +- wrist roll from jaw-level (advisor, 2026-10-27: more rams a finger into the stem)
CANDIDATES = 40                    # approach (pitch, tilt, depth, open) draws tried per scene, random order
FEASIBLE_ENOUGH = 6                # stop drawing once this many full plans work, then pick among them
# Jaw opening while approaching, normalised 0..100 (START_NORM has 50). Smallest one that clears the fruit.
OPEN_NORM = (50.0, 60.0, 70.0)
# Close command: the jaw fully closed (norm 0 = the -10 deg stop). K7 (2026-10-27): use the sim's own
# stall angle for now. ⚠️ A position servo commanded TO the stall angle doesn't squeeze; a person closes
# the leader past contact. The jaw stalls on the fruit by itself, at about -9.5 deg for 20-25 mm fruit
# and up to +4 deg for 30 mm (README): that's what gripper.pos (the state) will read. Redo all episodes
# once the real follower's gripper.pos on a tomato is measured.
CLOSE_NORM = (0.0, 2.0)
# --- detach ---
# Pull back along the fingers (the truss pivots ~8 cm away and its spring gives ~5 cm before the
# tether reaches 2.5 N, tomato.RACHIS_*). 6-8 cm back ran into the arm's reach on 17 of 30 seeds
# (2026-10-27), so it's 5-7 cm, tried along the fingers, then along the approach, then tilted up / down.
PULL_M = (0.05, 0.07)
PULL_UP_M = (0.005, 0.02)          # plus a little up, the way a hand lifts while pulling
PULL_TILT_DEG = (0.0, 25.0, -25.0)
TWIST_DEG = (40.0, 80.0)           # twist variant: wrist roll this far (either way) before the pull
# ⚠️ The sim's tether only counts the pull (tomato.Detacher reads the weld's translational rows), so a twist
# doesn't weaken the stem here; the fruit still comes off on the pull. It's there for the motion's variety.
LIFT_M = 0.04                      # up after the pluck, before swinging to the crate
# --- crate ---
CRATE_ABOVE_M = (0.035, 0.05)      # fruit bottom this far above the crate rim at release
CRATE_PITCH_DEG = (40.0, 75.0)     # fingers pointing down into the crate
CRATE_JITTER_M = 0.02              # +- release point around the crate centre
# --- near miss (K7, 2026-10-27: some episodes miss and correct) ---
MISS_OFF_M = (0.015, 0.022)        # arrive this far above or below the fruit...
MISS_SHORT_M = 0.012               # ...and this far short of the grasp depth, pause, back off, correct
MISS_PAUSE_S = (0.3, 0.6)
# --- timing ---
# Peak joint speed in normalised units/s (1 unit ~ 1.5-1.8 deg). ~60 = ~100 deg/s peak, a calm teleop
# pace (guess). Minimum-jerk peaks at 1.875x the average speed.
JOINT_PEAK_NORM_S = 60.0
CART_PEAK_M_S = 0.08               # straight-line legs (approach, pull)
SPEED = (0.8, 1.25)                # per-episode speed factor (domain randomisation of the motion)
HOLD_START_S = (0.4, 0.8)
HOLD_END_S = (0.5, 1.0)
# --- collision sweep ---
CLEARANCE_M = 0.004                # arm geoms must stay this far from the greenhouse, crate, table (planning only)
# Unripe neighbours may be brushed: a hand reaching into a truss does. The open fingers may press into
# a neighbour by up to this much in the kinematic sweep (the sprung truss gives way in the physics); the dry
# run then rejects any scene where a neighbour actually came off. Among the working plans, the ones that
# press least are preferred (within NEIGHBOUR_PREFER_M of the best), the pick among those is random.
# 6 mm first: then the wrist servo, camera mount and roll bracket (real part outlines, not hulls of the
# whole arm) pressed deeper into some neighbour on most approaches into a cluster (2026-10-27), as a
# teleoperated arm would. 15 mm lets those through to the physics, which decides (sprung trusses give).
NEIGHBOUR_PUSH_M = 0.015
NEIGHBOUR_PREFER_M = 0.002
SWEEP_STEP_NORM = 2.0              # joint-space sweep resolution
# Big moves (start -> pre-grasp, lift -> crate, crate -> start) straight in joint space swung the hand
# through the vines and knocked fruit off (dry runs, 2026-10-27). They now go the way a person moves an
# arm: tuck in, turn the pan while tucked, reach out. Tucks (lift, elbow, wrist_flex, norm), searched over
# pan -90..60 in 4 scenes for no contact at all: UP holds the hand ~0.38 m up, above the trusses and
# behind the vine face; LOW is the start pose's own tuck. Straight is kept when it's clear.
TUCKS = {"up": (-90.0, 40.0, -40.0), "low": tuple(scene.START_NORM[1:4])}
ROUTE_PRESS_M = 0.003              # a big swing may only graze a neighbour (it moves fast)
# The closing jaw may press a neighbour this much in the sweep; the dry run judges the rest. 4 mm rejected
# 19 of 30 seeds outright; 8 mm lets them through to the physics, which keeps some (2026-10-27).
CLOSE_PRESS_M = 0.008
STANDOFF_M = 0.05                  # extra way-point this much further back (+ up) from the pre-grasp
# --- servo sag correction ---
SAG_KI = 0.15                      # per frame, on (reference - measured) in norm units (tuned in the sim)
SAG_MAX_NORM = 6.0


def min_jerk(n):
    t = np.linspace(0, 1, max(int(n), 2))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def flank_x(z_m):
    """x of the fixed finger's contact flank at height z (gripper frame, m); same as check_grasp.flank_x."""
    (x0, z0), (x1, z1) = CHAIN["fixed"]["front_line_gripper_mm"]
    z = z_m * 1000
    return (x0 + (x1 - x0) * (z - z0) / (z1 - z0)) / 1000


def grasp_point(radius, depth):
    """Where the fruit centre sits in the gripper frame when the jaw starts closing on it."""
    return np.array([flank_x(depth) + radius + GAP_FIXED_M, CHAIN["fixed"]["y_mid_mm"] / 1000, depth])


class ArmIK:
    """IK on the bare arm at the scene's base pose: grasp point to a world point, with the fingers' pitch
    and the jaw's tilt set. Gripper frame: fingers along -z, jaw closes along x (check_grasp.find_pose)."""

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


class Sweep:
    """Kinematic collision checks on a copy of the scene (fruit where they hang now)."""

    def __init__(self, m, d, qidx, allow_geoms=()):
        self.m = copy.deepcopy(m)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = d.qpos
        self.qidx = qidx
        body_arm = np.array([self.m.body(b).name.startswith("arm/") for b in range(self.m.nbody)])
        self.is_arm = body_arm[self.m.geom_bodyid]
        for g in np.flatnonzero(self.is_arm & (self.m.geom_contype > 0)):
            self.m.geom_margin[g] = CLEARANCE_M
        self.allow = set(allow_geoms)
        self.is_fruit = np.array([bool(self.m.body_jntnum[self.m.geom_bodyid[g]] > 0) and not self.is_arm[g]
                                  and self.m.geom(g).name.endswith("_g") for g in range(self.m.ngeom)])
        self.push = 0.0                      # deepest neighbour press seen by path_hits since the last reset
        self.limit = NEIGHBOUR_PUSH_M        # Planner.plan(strict=) lowers it for a retry

    def hits(self, q6, allow=()):
        """Names of what the arm hits at q6 (rad): any scenery within CLEARANCE_M, or a neighbour fruit
        pressed deeper than NEIGHBOUR_PUSH_M. Shallower neighbour presses are only counted in self.push."""
        self.d.qpos[self.qidx] = q6
        mujoco.mj_fwdPosition(self.m, self.d)
        out = set()
        for c in self.d.contact[:self.d.ncon]:
            a, b = self.is_arm[c.geom1], self.is_arm[c.geom2]
            if a == b:
                continue
            other = c.geom2 if a else c.geom1
            if other in self.allow or other in allow:
                continue
            if self.is_fruit[other]:
                if c.dist < 0:
                    self.push = max(self.push, -c.dist)
                if c.dist > -self.limit:
                    continue
            out.add(self.m.geom(other).name)
        return out

    def path_hits(self, qs, allow=()):
        out = set()
        for q in qs:
            out |= self.hits(q, allow)
        return out

    def path_press(self, qs, allow=()):
        """(hits, deepest neighbour press) along qs, without touching the plan-wide self.push."""
        keep, self.push = self.push, 0.0
        h = self.path_hits(qs, allow)
        press, self.push = self.push, max(keep, self.push)
        return h, press


def joint_path(q0, q1, fm, step=SWEEP_STEP_NORM):
    """Joint-space line between two rad configs, sampled every `step` norm units on the largest joint."""
    n0, n1 = fm.rad_to_norm(q0), fm.rad_to_norm(q1)
    n = int(np.ceil(np.abs(n1 - n0).max() / step)) + 1
    return [fm.norm_to_rad(n0 + (n1 - n0) * s) for s in np.linspace(0, 1, n)]


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


class PlanError(Exception):
    pass


class Planner:
    def __init__(self, m, d, info, target_k):
        self.m, self.d, self.info = m, d, info
        self.k = target_k
        self.fruit = info["fruit"][target_k]
        self.qidx, _ = scene.arm_ids(m)
        self.fm = arm_map.FollowerMap()
        self.ik = arm_ik()
        self.tgeom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self.fruit["geom"])
        self.sweep = Sweep(m, d, self.qidx)
        self.c = d.geom_xpos[self.tgeom].copy()        # target centre now (after the settle)
        self.r = float(m.geom_size[self.tgeom][0])

    # --- helpers ---
    def q6(self, x5, grip_norm):
        q = np.r_[x5, 0.0]
        q[5] = self.fm.norm_to_rad(np.r_[np.zeros(5), grip_norm])[5]
        return q

    def line(self, p0, p1, pitch, tilt, point, x0, n=APPROACH_STEPS):
        """IK along a straight line of grasp points; None if any point fails or the joints jump."""
        xs, x = [], x0
        for s in np.linspace(0, 1, n):
            x = self.ik.solve(p0 + (p1 - p0) * s, pitch, tilt, point, x0=x)
            if x is None or (xs and np.abs(self.fm.rad_to_norm(np.r_[x, 0])[:5]
                                           - self.fm.rad_to_norm(np.r_[xs[-1], 0])[:5]).max() > 8):
                return None
            xs.append(x)
        return xs

    def pitches(self):
        """Pitches (deg) at which IK reaches the target with the jaw level."""
        lo, hi, step = PITCH_SCAN_DEG
        pt = grasp_point(self.r, float(np.mean(GRASP_DEPTH_M)))
        return [float(p) for p in np.arange(lo, hi + 1e-9, step) if self.ik.solve(self.c, p, 0.0, pt) is not None]

    def candidates(self, rng):
        ok = self.pitches()
        if not ok:
            return
        for _ in range(CANDIDATES):
            pitch = float(rng.choice(ok) + rng.uniform(-PITCH_SCAN_DEG[2] / 2, PITCH_SCAN_DEG[2] / 2))
            yield dict(pitch=pitch, tilt=float(rng.uniform(-JAW_TILT_DEG, JAW_TILT_DEG)),
                       depth=float(rng.uniform(*GRASP_DEPTH_M)), pre=float(rng.uniform(*PREGRASP_M)))

    def grasp_leg(self, cand, rng):
        """Grasp + pre-grasp IK and the approach line for one candidate; returns dict or a reason string."""
        point = grasp_point(self.r, cand["depth"])
        xg = self.ik.solve(self.c, cand["pitch"], cand["tilt"], point)
        if xg is None:
            return "no IK at the fruit"
        _, R = self.ik.frame(xg, point)
        f, y = -R[:, 2], R[:, 1]                             # finger direction, out of the jaw plane
        why = "no IK at the pre-grasp"
        blends = [APPROACH_BLEND_DEG[i] for i in rng.permutation(len(APPROACH_BLEND_DEG))]
        for D in (cand["pre"],) + PREGRASP_FALLBACK_M:
            for beta in blends:
                b = np.radians(beta)
                v = np.cos(b) * f + np.sin(b) * y            # approach direction (unit)
                p_pre = self.c - D * v
                xp = self.ik.solve(p_pre, cand["pitch"], cand["tilt"], point, x0=xg)
                if xp is None:
                    continue
                app = self.line(p_pre, self.c, cand["pitch"], cand["tilt"], point, xp)
                if app is None:
                    why = "approach line breaks"
                    continue
                for g in OPEN_NORM:                          # smallest opening that clears everything
                    qs = [self.q6(x, g) for x in app]
                    # the target may only be near the fingers on the last third of the approach, never inside
                    far, near = qs[:2 * len(qs) // 3], qs[2 * len(qs) // 3:]
                    h = self.sweep.path_hits(far) | self.sweep.path_hits(near, allow={self.tgeom})
                    if not h and not self._penetrates_target(qs[-1]):
                        return dict(cand, pre=D, blend=beta, v=v, open=g, point=point, xg=xg, xp=xp, app=app,
                                    f=f, R=R)
                why = f"approach hits {sorted(h)[:3] if h else 'the target'}"
        return why

    def _penetrates_target(self, q6):
        self.sweep.d.qpos[self.qidx] = q6
        mujoco.mj_fwdPosition(self.sweep.m, self.sweep.d)
        for c in self.sweep.d.contact[:self.sweep.d.ncon]:
            if self.tgeom in (c.geom1, c.geom2) and (self.sweep.is_arm[c.geom1] or self.sweep.is_arm[c.geom2]):
                if c.dist < 0.001:
                    return True
        return False

    def free(self, q_from, q_to, allow=()):
        return not self.sweep.path_hits(joint_path(q_from, q_to, self.fm), allow)

    def route(self, q_from, q_to, allow=(), last=None):
        """Waypoints (rad, endpoints excluded) for a big move that stays out of the vines, or None.
        Straight, or through a tuck: tuck at the start's pan, turn to the goal's pan, reach out.
        last: optional way-point just before q_to (a stand-off in front of the fruit), tried as well."""
        fm = self.fm
        a, b = fm.rad_to_norm(q_from), fm.rad_to_norm(q_to)

        def tuck(n, pan, name):
            t = n.copy()
            t[0], t[1:4] = pan, TUCKS[name]
            return fm.norm_to_rad(t)

        options = [[]]
        for name in TUCKS:
            options += [[tuck(a, a[0], name), tuck(b, b[0], name)], [tuck(b, a[0], name)], [tuck(a, b[0], name)]]
        if last is not None:
            options += [o + [last] for o in options]
        best = None
        for vias in options:
            pts = [q_from] + vias + [q_to]
            qs = [q for u, v in zip(pts, pts[1:]) for q in joint_path(u, v, fm)]
            h, press = self.sweep.path_press(qs, allow)
            if not h and press <= ROUTE_PRESS_M and (best is None or press < best[1] - 1e-4):
                best = (vias, press)
                if press == 0:
                    break
        return None if best is None else best[0]

    # --- the plan ---
    def plan(self, rng, twist=False, near_miss=False, strict=1.0):
        """strict (0..1] scales the neighbour press limits: a retry after a dry run knocked a neighbour off
        asks for a cleaner path, so it isn't the same plan drawn again."""
        fm = self.fm
        self.sweep.limit = NEIGHBOUR_PUSH_M * strict
        start = np.asarray(scene.START_NORM, float)
        q_start = fm.norm_to_rad(start)
        speed = float(rng.uniform(*SPEED))
        tried, good = [], []
        for cand in self.candidates(rng):
            if len(good) >= FEASIBLE_ENOUGH:
                break
            self.sweep.push = 0.0
            leg = self.grasp_leg(cand, rng)
            if isinstance(leg, str):
                tried.append(leg)
                continue
            q_pre = self.q6(leg["xp"], leg["open"])
            p_so = self.c - (leg["pre"] + STANDOFF_M) * leg["v"] + np.array([0, 0, 0.02])
            xs = self.ik.solve(p_so, leg["pitch"], leg["tilt"], leg["point"], x0=leg["xp"])
            via = self.route(q_start, q_pre, last=None if xs is None else self.q6(xs, leg["open"]))  # start -> pre-grasp
            if via is None:
                tried.append("start -> pre-grasp blocked")
                continue
            # the closing jaw sweeps the moving finger through the neighbours too
            closing = [self.q6(leg["xg"], leg["open"] + (CLOSE_NORM[0] - leg["open"]) * u) for u in np.linspace(0, 1, 8)]
            h, press = self.sweep.path_press(closing, allow={self.tgeom})
            if h or press > CLOSE_PRESS_M * strict:
                tried.append("closing jaw pushes a neighbour")
                continue
            rest = self._after_grasp(rng, leg, twist)
            if isinstance(rest, str):
                tried.append(rest)
                continue
            miss = self._near_miss(rng, leg) if near_miss else None
            if near_miss and isinstance(miss, str):
                tried.append(miss)
                continue
            good.append((self.sweep.push, leg, via, rest, miss))
        if not good:
            raise PlanError("; ".join(sorted(set(tried))) or "IK reaches the target at no pitch")
        best = min(g[0] for g in good)
        near = [g for g in good if g[0] <= best + NEIGHBOUR_PREFER_M]
        push, leg, via, rest, miss = near[int(rng.integers(len(near)))]
        plan = self._assemble(rng, speed, leg, via, rest, miss, twist, tried)
        plan.meta.update(neighbour_press_mm=round(push * 1000, 1), feasible_plans=len(good))
        return plan

    def _after_grasp(self, rng, leg, twist):
        close = float(rng.uniform(*CLOSE_NORM))
        pull = float(rng.uniform(*PULL_M))
        up = float(rng.uniform(*PULL_UP_M))
        pl = xl = None
        for base in (leg["f"], leg["v"]):
            for tilt in PULL_TILT_DEG:
                t = np.radians(tilt)
                u = np.cos(t) * base + np.sin(t) * np.array([0, 0, -1.0])     # + tilt = pull back and up
                p_end = self.c - pull * u / np.linalg.norm(u) + np.array([0, 0, up])
                pl = self.line(self.c, p_end, leg["pitch"], leg["tilt"], leg["point"], leg["xg"])
                if pl is None:
                    continue
                xl = self.ik.solve(p_end + [0, 0, LIFT_M], leg["pitch"], leg["tilt"], leg["point"], x0=pl[-1])
                if xl is not None:
                    break
            if pl is not None and xl is not None:
                break
        if pl is None:
            return "no IK along the pull"
        if xl is None:
            return "no IK for the lift"
        roll = float(rng.choice([-1, 1]) * rng.uniform(*TWIST_DEG)) if twist else 0.0
        if twist:
            # the wrist stays twisted through pull + lift (IK above is the untwisted path, roll added after)
            lo, hi = self.ik.lo[4], self.ik.hi[4]
            if not lo < leg["xg"][4] + np.radians(roll) < hi:
                roll = -roll
            tw = np.r_[0, 0, 0, 0, np.radians(roll)]
            pl, xl = [x + tw for x in pl], xl + tw
        # Pull + lift: the target comes along (allowed), nothing else may be touched.
        if self.sweep.path_hits([self.q6(x, close) for x in pl] + [self.q6(xl, close)], allow={self.tgeom}):
            return "pull/lift hits a neighbour"
        # Crate pose: fingers down, release point jittered around the crate centre.
        (cx, cy), (_, _, h) = scene.CRATE_CENTRE, scene.CRATE_SIZE
        xc = None
        for _ in range(8):
            p = np.array([cx + rng.uniform(-CRATE_JITTER_M, CRATE_JITTER_M),
                          cy + rng.uniform(-CRATE_JITTER_M, CRATE_JITTER_M) * 0.8,
                          h + self.r + rng.uniform(*CRATE_ABOVE_M)])
            pitch_c = float(rng.uniform(*CRATE_PITCH_DEG))
            xc = self.ik.solve(p, pitch_c, float(rng.uniform(-10, 10)), leg["point"])
            if xc is not None:
                break
        if xc is None:
            return "no IK above the crate"
        q_lift, q_crate = self.q6(xl, close), self.q6(xc, close)
        # ⚠️ the planning copy has the target still hanging; the held fruit isn't checked on the way
        to_crate = self.route(q_lift, q_crate, allow={self.tgeom})
        if to_crate is None:
            return "lift -> crate blocked"
        q_start = self.fm.norm_to_rad(np.asarray(scene.START_NORM, float))
        q_rel = self.q6(xc, leg["open"])
        back = self.route(q_rel, q_start, allow={self.tgeom})
        if back is None:
            return "crate -> start blocked"
        return dict(close=close, pull=pull, up=up, roll=roll, pl=pl, xl=xl, xc=xc, crate_point=p.tolist(),
                    crate_pitch=pitch_c, to_crate=to_crate, back=back)

    def _near_miss(self, rng, leg):
        y = leg["R"][:, 1]                                   # out of the jaw plane: above / below the fruit
        off = float(rng.choice([-1, 1]) * rng.uniform(*MISS_OFF_M))
        y = y - (y @ leg["v"]) * leg["v"]
        y /= np.linalg.norm(y)                               # sideways to the approach line
        p_miss = self.c + off * y - MISS_SHORT_M * leg["v"]
        p_pre = self.c - leg["pre"] * leg["v"]
        ml = self.line(p_pre, p_miss, leg["pitch"], leg["tilt"], leg["point"], leg["xp"])
        if ml is None:
            return "no IK to the miss point"
        p_back = self.c - 0.03 * leg["v"]
        bl = self.line(p_miss, p_back, leg["pitch"], leg["tilt"], leg["point"], ml[-1], n=8)
        if bl is None:
            return "no IK back from the miss"
        qs = [self.q6(x, leg["open"]) for x in ml + bl]
        if self.sweep.path_hits(qs):
            return "near miss touches the fruit"
        # From p_back, the normal approach finishes the job: IK along p_back -> c.
        fin = self.line(p_back, self.c, leg["pitch"], leg["tilt"], leg["point"], bl[-1], n=6)
        if fin is None:
            return "no IK for the corrected approach"
        return dict(off=off, ml=ml, bl=bl, fin=fin, pause=float(rng.uniform(*MISS_PAUSE_S)))

    # --- turn waypoints into frames ---
    def _assemble(self, rng, speed, leg, via, rest, miss, twist, tried):
        fm = self.fm
        rows, servo, phases = [], [], []
        cur = np.asarray(scene.START_NORM, float)

        def hold(seconds, name, sv=False):
            n = int(round(seconds * FPS))
            rows.extend([rows[-1] if rows else cur] * n)
            servo.extend([sv] * n)
            phases.append((name, n))

        def joint_move(target_norm, name, sv=False):
            a = np.asarray(rows[-1] if rows else cur, float)
            b = np.asarray(target_norm, float)
            T = 1.875 * np.abs(b - a)[:5].max() / (JOINT_PEAK_NORM_S * speed)
            T = float(np.clip(T * rng.uniform(0.9, 1.1), 0.6, 4.0))
            s = min_jerk(T * FPS)[1:]
            rows.extend([a + (b - a) * si for si in s])
            servo.extend([sv] * len(s))
            phases.append((name, len(s)))

        def cart_move(xs, grip, length_m, name, sv=True, grip_from=None):
            """Follow IK points xs with min-jerk timing; gripper fixed or ramping from grip_from."""
            T = float(np.clip(1.875 * length_m / (CART_PEAK_M_S * speed) * rng.uniform(0.9, 1.1), 0.5, 3.0))
            s = min_jerk(T * FPS)[1:]
            N = np.array([fm.rad_to_norm(np.r_[x, 0])[:5] for x in xs])
            u = np.linspace(0, 1, len(N))
            for si in s:
                arm = np.array([np.interp(si, u, N[:, j]) for j in range(5)])
                g = grip if grip_from is None else grip_from + (grip - grip_from) * si
                rows.append(np.r_[arm, g])
            servo.extend([sv] * len(s))
            phases.append((name, len(s)))

        def norm(x5, g):
            return np.r_[fm.rad_to_norm(np.r_[x5, 0])[:5], g]

        g_open, close = leg["open"], rest["close"]
        def path_move(vias, target_norm, name):
            """One smooth move through way-points: min-jerk timing over the whole polyline (no stop at a
            tuck), corners rounded by a short moving average. Endpoints exact."""
            pts = [np.asarray(rows[-1], float)] + [np.r_[fm.rad_to_norm(q)[:5], target_norm[5]] for q in vias]
            pts.append(np.asarray(target_norm, float))
            seg = np.array([np.abs(b - a)[:5].max() for a, b in zip(pts, pts[1:])])
            L = seg.sum()
            T = float(np.clip(1.875 * L / (JOINT_PEAK_NORM_S * speed) * rng.uniform(0.9, 1.1), 0.6, 6.0))
            s = min_jerk(T * FPS) * L
            cum = np.r_[0, np.cumsum(seg)]
            P = np.array(pts)
            path = np.array([[np.interp(si, cum, P[:, j]) for j in range(6)] for si in s])
            if len(vias) and len(path) > 12:
                w = 9                                        # frames; rounds each corner over ~0.3 s
                pad = np.r_[np.repeat(path[:1], w, 0), path, np.repeat(path[-1:], w, 0)]
                ker = np.ones(w) / w
                path = np.array([np.convolve(pad[:, j], ker, "same")[w:-w] for j in range(6)]).T
                path[0], path[-1] = pts[0], pts[-1]
            rows.extend(path[1:])
            servo.extend([False] * (len(path) - 1))
            phases.append((name, len(path) - 1))

        hold(rng.uniform(*HOLD_START_S), "hold_start")
        path_move(via, norm(leg["xp"], g_open), "to_pregrasp")
        if miss is not None:
            cart_move(miss["ml"], g_open, leg["pre"] - MISS_SHORT_M, "near_miss")
            hold(miss["pause"], "near_miss_pause", sv=True)
            cart_move(miss["bl"], g_open, 0.03, "near_miss_back_off")
            cart_move(miss["fin"], g_open, 0.03, "approach_corrected")
        else:
            cart_move(leg["app"], g_open, leg["pre"], "approach")
        hold(rng.uniform(0.15, 0.3), "settle_at_fruit", sv=True)
        n_close = int(round(rng.uniform(0.5, 0.8) * FPS))
        a = rows[-1].copy()
        for si in min_jerk(n_close)[1:]:
            rows.append(np.r_[a[:5], g_open + (close - g_open) * si])
        servo.extend([True] * (n_close - 1))
        phases.append(("close", n_close - 1))
        hold(rng.uniform(0.3, 0.5), "squeeze", sv=True)
        if twist:
            b = rows[-1].copy()
            b[4] = fm.rad_to_norm(np.r_[rest["pl"][0], 0])[4]
            joint_move(b, "twist", sv=True)
        cart_move(rest["pl"], close, rest["pull"], "pull", sv=False)
        joint_move(norm(rest["xl"], close), "lift")
        path_move(rest["to_crate"], norm(rest["xc"], close), "to_crate")
        hold(rng.uniform(0.2, 0.4), "over_crate")
        joint_move(norm(rest["xc"], g_open), "release")
        hold(rng.uniform(0.4, 0.7), "drop")
        path_move(rest["back"], np.asarray(scene.START_NORM, float), "return")
        hold(rng.uniform(*HOLD_END_S), "hold_end")
        meta = dict(target=self.fruit["tomato"], target_radius_mm=round(self.r * 1000, 1),
                    approach_pitch_deg=round(leg["pitch"], 1), jaw_tilt_deg=round(leg["tilt"], 1),
                    grasp_depth_mm=round(leg["depth"] * 1000, 1), pregrasp_mm=round(leg["pre"] * 1000, 1),
                    approach_blend_deg=leg["blend"],
                    open_norm=g_open, close_norm=round(close, 2), tucks=[len(via), len(rest["to_crate"]), len(rest["back"])],
                    detach="twist+pull" if twist else "pull", twist_deg=round(rest["roll"], 1),
                    pull_mm=round(rest["pull"] * 1000, 1), near_miss=miss is not None,
                    near_miss_off_mm=None if miss is None else round(miss["off"] * 1000, 1),
                    speed=round(speed, 3), crate_point=np.round(rest["crate_point"], 3).tolist(),
                    frames=len(rows), seconds=round(len(rows) / FPS, 2), rejected_candidates=len(tried))
        return Plan(np.array(rows), np.array(servo), phases, meta)
