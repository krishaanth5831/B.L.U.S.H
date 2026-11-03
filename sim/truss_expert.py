"""The scripted truss picker: cut one ready truss's stalk, carry the truss to the crate (K7, 2026-11-03).

    plan = TrussPlanner(robot.m, robot.d, robot.info, k).plan(rng, miss=False)   # after the start frame
    while not plan.done:
        robot.send_action(plan.act(state_norm))

IK and RRT do different jobs (K7 2026-11-01: "RRT for every route, IK for the poses"):
- IK (truss_ik.py, expert.ArmIK) finds the goal poses: the cut pose (blades on the stalk's cut zone, the stalk
  across the jaw), the pre-cut pose SLIDE_M back along the fingers, the end of the retreat, above the crate,
  the release pose in the crate.
- RRT-Connect (rrt.py) finds every free move between them: start -> pre-cut, retreat -> above the crate,
  crate -> start. Short contact moves (the slide onto the stalk, the retreat, the lowering into the crate) are
  straight lines, collision-checked too.

What the arm must not touch (rrt.Checker sets):
- hard: the greenhouse, crate, table, every stem. Kept rrt.CLEARANCE away on every move.
- soft: every OTHER truss (stalks and fruit), NEIGHBOUR_CLEAR away: not brushed, or in a crowd at most pressed
  2 mm, and the judge checks they didn't move (K7: "doesn't affect other tomatoes").
- firm: the target truss's own fruit. Avoided on every move; a knocked-off fruit fails the episode.
- the target's stalk: avoided on the way in, ignored on the slide (it goes between the open tips), and the
  whole target truss is ignored once it's cut and held.

The motion: hold at start -> RRT to the pre-cut pose (jaw opening on the way) -> [miss: slide in SHORT of the
stalk, close on nothing, open, back off] -> slide onto the stalk -> close (the blades cut ~0.45 s in) -> squeeze
SQUEEZE_S (longer than the single-fruit picker's 0.3-0.5 s, so a policy learns not to pull while the jaw still
closes) -> retreat toward the aisle and up -> RRT to above the crate -> lower -> open -> wait -> rise -> RRT
back to start -> hold. Minimum-jerk timing; servo sag corrected like expert.Plan.
"""
import mujoco
import numpy as np

import arm_map
import expert
import rrt
import scene
import truss_ik
import truss_scene as ts

FPS = expert.FPS
SLIDE_M = 0.045                  # pre-cut pose this far back along the approach (SHEAR cutter.SLIDE_M 4 cm, guess)
SLIDE_FALLBACK_M = (0.035, 0.025)   # shorter slides when that pre-cut pose is out of reach (cut poses sit near
                                    # the reach edge: 5 of 10 seeds had no IK 4.5 cm back, 2026-11-03)
# The stalk may enter the jaw along the fingers, or sliding along itself at the same time (it lies along gripper
# y, so a y component keeps it between the open tips). Blend toward +-y, tried in random order (design choice).
APPROACH_BLEND_DEG = (0.0, 30.0, -30.0, 50.0, -50.0)
SLIDE_STEPS = 10
JAW_OPEN_DEG = 25.0              # open while approaching: tips ~30 mm apart past the pads (check_truss_cut)
CLOSE_S = (0.5, 0.8)             # closing time, like expert's
SQUEEZE_S = (0.6, 0.9)           # held shut before moving off (cut lands ~0.45 s into the close)
RETREAT_M = (0.03, 0.05)         # back along the approach after the cut (5-7 cm ran out of reach, 2026-11-03)...
RETREAT_UP_M = (0.02, 0.04)      # ...and up, so the truss clears its neighbours before any swing
# Retreat directions tried in order: back along the approach, toward the aisle, straight up (design choice).
CRATE_JITTER_M = 0.03            # release point around the crate centre
# Release: the truss held with its lowest fruit this far above the crate RIM and its outline (seen from above)
# inside the walls, then dropped in. ⚠️ Lowering it to just above the floor needed a rigid ~17 cm truss to fit
# 20 x 18 cm without touching a wall at its held angle, which almost never worked (2026-11-03). It falls 5-8 cm.
RELEASE_ABOVE_M = (0.01, 0.03)
ABOVE_CRATE_M = 0.08             # way-point this far above the release pose (lowered straight from there)
RELEASE_PITCH_DEG = (10.0, 85.0)  # fingers pointing down over the crate (wide: the IK scan reached the crate's
                                  # far half at 10-85, 2026-11-03)
RELEASE_SAMPLES = 120            # release poses tried; the one leaving the most room to the walls is kept
ABOVE_RIM_MAX_M = 0.25           # no carried fruit higher than this over the rim at release
RELEASE_SLACK_M = 0.005          # every fruit's outline at least this far inside the walls, seen from above
DROP_S = (0.5, 0.8)
MISS_SHORT_M = (0.015, 0.025)    # miss: the tips stop this far short of the stalk, close on nothing
MISS_PAUSE_S = (0.3, 0.5)
SPEED = expert.SPEED
JOINT_PEAK_NORM_S = expert.JOINT_PEAK_NORM_S
CART_PEAK_M_S = expert.CART_PEAK_M_S
RRT_SEEDS = 3                    # RRT attempts per route before giving up on a candidate (design choice)
# Neighbouring trusses: kept this far from the arm on every move (rrt "soft" set, soft_clearance). A plan is
# tried strict first; when nothing clears them, again letting the arm press them by up to 2 mm (the sprung
# trusses give; the dry run's judge then rejects any that swing more than generate_truss.NEIGHBOUR_MOVE_M or
# lose a fruit). In crowded scenes 6 of 12 strict plans had a neighbour within 4 mm of the slide (2026-11-03).
NEIGHBOUR_CLEAR = {"strict": rrt.CLEARANCE, "brush": -0.002}
# The target truss's OWN fruit (rrt "firm"): avoided by SOFT_CLEARANCE strictly; in "brush" mode they may be touched
# like neighbours. ⚠️ With the steep stalks a cut needs, the truss's cluster hangs right beside the jaw: every
# slide in seed 200003 met its own fruit (2026-11-03). The judge rejects any fruit that comes off.
OWN_FRUIT_CLEAR = {"strict": rrt.SOFT_CLEARANCE, "brush": -0.002}
# Carrying the cut truss to the crate: it and the arm keep this far from every neighbour. ⚠️ The carried truss swings
# on the way, which the static check (CarryChecker) doesn't see: at the cut's 2 mm it tore 2 fruit off a neighbour
# on the way to the crate (seed 200001, 2026-11-03).
CARRY_CLEAR_M = 0.010
CANDIDATES = 8                   # cut points drawn along the cut zone per plan


class PlanError(Exception):
    pass


def _pose(d, b):
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def _axis_rot(a, t):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


# ⚠️ How the cut truss rides in the jaw. "rigid": as it was at the cut, fixed in the gripper. Measured (seed 200000,
# 2026-11-03): it stays put, even with the pads' torsional friction cut to 0.002 m, because the pads touch a
# 13 mm length of stalk and the sliding friction along that line resists the truss's ~0.04-0.08 N.m (the
# swing-down model put the fruit 15 cm below the grip at release; they were 2 cm above it, and the truss
# tipped off the crate). A real rubber clamp holds it the same way. "hang" swings it under the grip instead.
CARRY = "rigid"


def hanging(o, R, rel, masses, grip_point):
    """World poses [(pos, R)] of the carried truss's bodies with the gripper at (o, R): held as at the cut (CARRY
    "rigid"), or also swung about the jaw's closing axis (gripper x, through the grip point) until its centre of
    mass hangs straight below it (CARRY "hang")."""
    P = o + R @ grip_point
    W = [(o + R @ p, R @ Rr) for Rr, p in rel]
    if CARRY == "rigid":
        return W
    a = R[:, 0]
    c = sum(mb * w[0] for mb, w in zip(masses, W)) / sum(masses) - P
    c_perp = c - (c @ a) * a
    down = np.array([0, 0, -1.0]) - (-a[2]) * a
    if np.linalg.norm(c_perp) < 1e-6 or np.linalg.norm(down) < 1e-6:
        return W                                  # closing axis vertical: it can't swing round it
    c_perp, down = c_perp / np.linalg.norm(c_perp), down / np.linalg.norm(down)
    t = np.arctan2(np.cross(c_perp, down) @ a, c_perp @ down)
    Q = _axis_rot(a, t)
    return [(P + Q @ (p - P), Q @ Rw) for p, Rw in W]


CRATE_POSE_SAMPLES = 60000      # random arm poses sampled once per process; the ones gripping above the crate kept
_CRATE_POSES = None


def crate_poses(ikf, point):
    """Arm poses (Q, gripper o, R) with the grip point above the crate area, from CRATE_POSE_SAMPLES random ones.
    ⚠️ Shifting a sampled pose to the crate and re-solving IK failed for 40 of 40 (a 5-joint arm can't keep an
    orientation at another spot, 2026-11-03); poses already over the crate need no shift."""
    global _CRATE_POSES
    if _CRATE_POSES is None:
        (cx, cy), (sx, sy, _) = ts.CRATE_CENTRE, ts.CRATE_SIZE
        rng = np.random.default_rng(7)
        Q = rng.uniform(ikf.lo, ikf.hi, (CRATE_POSE_SAMPLES, 5))
        keep = []
        for q in Q:
            o, R = ikf.fk(q)
            g = o + R @ point
            if abs(g[0] - cx) < sx / 2 + 0.05 and abs(g[1] - cy) < sy / 2 + 0.05 and 0.06 < g[2] < 0.40 and R[2, 2] > 0.1:
                keep.append((q, o, R))
        _CRATE_POSES = ([k[0] for k in keep], np.array([k[1] for k in keep]), np.array([k[2] for k in keep]))
    return _CRATE_POSES


class CarryChecker(rrt.Checker):
    """rrt.Checker with the cut truss riding in the jaw: before each check the truss and its fruit are put where
    they'd be (hanging(), CARRY), and their geoms count as part of the arm. So a route after the cut is checked for the
    whole carried truss (seed 200006 tore fruit off a neighbour on the way to the crate when it wasn't, 2026-11-03).
    ⚠️ Static: the swing on the way (the truss settles into hanging after the cut) isn't modelled; the dry run's
    judge counts what it hits."""

    def __init__(self, m, d, carried_bodies, rel, grip_point, **kw):
        super().__init__(m, d, **kw)
        mm = self.m
        self.G = mm.body("arm/gripper").id
        self.adr_c = [mm.jnt_qposadr[mm.body_jntadr[b]] for b in carried_bodies]
        self.rel, self.grip = rel, grip_point
        self.masses = [float(mm.body_mass[b]) for b in carried_bodies]
        carried_set = set(carried_bodies)
        for g in range(mm.ngeom):
            if mm.geom_bodyid[g] in carried_set:
                self.is_arm[g] = True
                mm.geom_margin[g] = rrt.CLEARANCE

    def hits(self, q, **kw):
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        o, R = _pose(self.d, self.G)
        for adr, (pw, Rw) in zip(self.adr_c, hanging(o, R, self.rel, self.masses, self.grip)):
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Rw.reshape(-1))
            self.d.qpos[adr:adr + 3], self.d.qpos[adr + 3:adr + 7] = pw, quat
        return super().hits(q, **kw)


class TrussPlanner:
    def __init__(self, m, d, info, k):
        self.m, self.d, self.info, self.k = m, d, info, k
        self.t = info["trusses"][k]
        self.fm = arm_map.FollowerMap()
        self.ik, self.ik_free = truss_ik.truss_ik(), expert.arm_ik()
        self.qidx, _ = scene.arm_ids(m)
        name = lambda g: m.geom(g).name or ""   # noqa: E731
        is_arm = np.array([m.body(m.geom_bodyid[g]).name.startswith("arm/") for g in range(m.ngeom)])
        mine = set(self.t["geoms"]) | {f["geom"] for f in self.t["fruit"]}
        self.target_stalk = {m.geom(g).id for g in self.t["geoms"]}
        self.target_fruit = {m.geom(f["geom"]).id for f in self.t["fruit"]}
        self.target_all = self.target_stalk | self.target_fruit
        trusses = {gn for t in info["trusses"] for gn in t["geoms"] + [f["geom"] for f in t["fruit"]]}
        hard, soft = set(), set()
        for g in range(m.ngeom):
            n = name(g)
            if is_arm[g] or not n or n in mine:
                continue
            if n in trusses:
                soft.add(g)
            elif n.startswith("stem") or m.geom_contype[g] or m.geom_conaffinity[g]:
                hard.add(g)
        self.hard, self.soft = hard, soft
        self.chk = rrt.Checker(m, d, hard=hard, soft=soft, firm=self.target_all)
        self.kw = dict(soft_clearance=NEIGHBOUR_CLEAR["strict"], firm_clearance=OWN_FRUIT_CLEAR["strict"])
        self.g_adr = m.joint("arm/gripper").qposadr[0]
        self.lo, self.hi = (np.asarray(v[:5]) for v in self.fm.rad_limits())

    # --- helpers ---
    def jaw(self, deg):
        """Set the checker's jaw angle (it plans the 5 arm joints; the fingers' opening matters for hits)."""
        self.chk.d.qpos[self.g_adr] = np.radians(deg)

    def q_now(self):
        return self.d.qpos[self.qidx][:5].copy()

    def line_ik(self, p0, p1, s, r, x0, n):
        xs, x = [], x0
        for u in np.linspace(0, 1, n):
            x = self.ik.solve(p0 + (p1 - p0) * u, s, r, x0=x)
            if x is None:
                return None
            xs.append(x)
        return xs

    def route(self, q0, q1, rng, **kw):
        for _ in range(RRT_SEEDS):
            p = rrt.plan(self.chk, q0, q1, rng, lo=self.lo, hi=self.hi, **{**self.kw, **kw})
            if p is not None:
                return p
        return None

    def hang_m(self):
        """How far the target's lowest fruit hangs below the cut point once it dangles from the grip."""
        p, _ = ts.cut_target_now(self.t, self.d, self.m)
        return max(np.linalg.norm(self.d.xpos[self.m.body(f["tomato"]).id] - p) + f["radius"] for f in self.t["fruit"])

    # --- the plan ---
    def plan(self, rng, miss=False):
        """Strict first, then letting the arm brush neighbours by 2 mm (NEIGHBOUR_CLEAR)."""
        why = []
        for mode in NEIGHBOUR_CLEAR:
            self.kw = dict(soft_clearance=NEIGHBOUR_CLEAR[mode], firm_clearance=OWN_FRUIT_CLEAR[mode])
            try:
                plan = self._plan(rng, miss)
                plan.meta["neighbours"] = mode
                return plan
            except PlanError as e:
                why.append(f"{mode}: {e}")
        raise PlanError(" / ".join(why))

    def _plan(self, rng, miss=False):
        tried = []
        r = self.t["r"]
        q_start = self.q_now()
        for _ in range(CANDIDATES):
            leg = self._cut_leg(rng, r)
            if isinstance(leg, str):
                tried.append(leg)
                continue
            x_cut, p_cut, s, f, slide, slide_m, beta, u = leg
            self.jaw(JAW_OPEN_DEG)
            to_pre = self.route(q_start, slide[0], rng)
            if to_pre is None:
                tried.append("start -> pre-cut: no RRT route")
                continue
            miss_leg = None
            if miss:
                d_short = rng.uniform(*MISS_SHORT_M)
                d_short = min(d_short, slide_m - 0.005)
                ml = self.line_ik(p_cut - f * slide_m, p_cut - f * d_short, s, r, slide[0], SLIDE_STEPS // 2 + 2)
                if ml is None or not self.chk.path_ok(ml, **self.kw):
                    tried.append("miss leg blocked")
                    continue
                miss_leg = dict(xs=ml, short=d_short, slide_m=slide_m)
            rest = self._after_cut(rng, x_cut, p_cut, f, s, r, self._carry(x_cut))
            if isinstance(rest, str):
                tried.append(rest)
                continue
            return self._assemble(rng, to_pre, slide, miss_leg, rest,
                                  dict(cut_u=round(u, 2), approach_blend_deg=float(beta), slide_mm=round(slide_m * 1000),
                                       tried=tried), slide_m)
        raise PlanError("; ".join(sorted(set(tried))) or "no candidate")

    def _cut_leg(self, rng, r):
        """One cut point drawn in the cut zone: its cut pose, an approach whose slide is clear, and a closing jaw
        that touches only the stalk. Returns (x_cut, p_cut, s, f, slide, slide_m, beta, u) or why not."""
        u = rng.uniform(0.2, 0.8)
        p_cut, s = ts.cut_target_now(self.t, self.d, self.m, u)
        x_cut = self.ik.solve(p_cut, s, r)
        if x_cut is None:
            return "no cut pose"
        _, R = self.ik.frame(x_cut, r)
        slide = None
        for beta in rng.permutation(APPROACH_BLEND_DEG):
            b = np.radians(beta)
            f = np.cos(b) * -R[:, 2] + np.sin(b) * R[:, 1]   # approach: along the fingers, blended along the stalk
            for L in (SLIDE_M,) + SLIDE_FALLBACK_M:
                slide = self.line_ik(p_cut - f * L, p_cut, s, r, x_cut, SLIDE_STEPS)   # warm-started at the cut
                if slide is not None:
                    break
            if slide is not None:
                break
        if slide is None:
            return "no IK along the slide"
        slide_m = L
        self.jaw(JAW_OPEN_DEG)
        # the slide: tips go round the stalk (ignored), never into its fruit or anything else
        if not self.chk.path_ok(slide, ignore=self.target_stalk, **self.kw):
            return "slide hits " + ",".join(sorted(self.chk.hits(slide[-1], ignore=self.target_stalk, **self.kw)))[:60]
        # the closing jaw sweeps the moving finger: still clear of everything but the stalk
        ok = True
        for deg in np.linspace(JAW_OPEN_DEG, -10.0, 6):
            self.jaw(deg)
            if self.chk.hits(slide[-1], ignore=self.target_stalk, **self.kw):
                ok = False
                break
        if not ok:
            return "closing jaw hits a neighbour"
        return x_cut, p_cut, s, f, slide, slide_m, beta, u

    def approach_ok(self, rng, tries=CANDIDATES):
        """Some cut point of this truss can be reached and closed on (neighbours brushed at most): the scene
        builder only calls such a truss ready (truss_scene.build)."""
        self.kw = dict(soft_clearance=NEIGHBOUR_CLEAR["brush"], firm_clearance=OWN_FRUIT_CLEAR["brush"])
        return any(not isinstance(self._cut_leg(rng, self.t["r"]), str) for _ in range(tries))

    def _carry(self, x_cut):
        """The cut truss's bodies (stalk + fruit) and their poses in the gripper frame at the cut pose. ⚠️ The arm
        IK model and the scene share the arm base pose, so the IK's gripper pose is the scene's (truss_ik)."""
        o, R = self.ik.fk(x_cut)
        m, d = self.m, self.d
        bodies = [m.body(self.t["body"]).id] + [m.body(f["tomato"]).id for f in self.t["fruit"]]
        rel = [(R.T @ d.xmat[b].reshape(3, 3), R.T @ (d.xpos[b] - o)) for b in bodies]
        # the stalk + pedicels' mass sits in the truss body: subtree mass, minus the fruit (free bodies of their own)
        masses = [float(m.body_subtreemass[bodies[0]])] + [float(m.body_mass[b]) for b in bodies[1:]]
        return bodies, rel, masses

    def _carried_fruit(self, x, carry):
        """Fruit centres (world) at arm joints x with the truss hanging from the grip (hanging())."""
        o, R = self.ik_free.fk(x)
        W = hanging(o, R, carry[1], carry[2], truss_ik.cut_point_in_gripper(self.t["r"]))
        return np.array([w[0] for w in W[1:]])

    def _after_cut(self, rng, x_cut, p_cut, f, s, r, carry):
        cchk = CarryChecker(self.m, self.d, carry[0], carry[1], truss_ik.cut_point_in_gripper(r), hard=self.hard,
                            soft=self.soft)
        cchk.d.qpos[self.g_adr] = np.radians(-10.0)
        # retreat: back along the fingers and up, the truss in hand (ignored); other trusses stay hard
        self.jaw(-10.0)
        dist, up = rng.uniform(*RETREAT_M), rng.uniform(*RETREAT_UP_M)
        aisle = np.array([0, -np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]), 0.0])
        ret, why = None, "no IK for the retreat"
        options = (p_cut - f * dist + [0, 0, up], p_cut + aisle * dist + [0, 0, up], p_cut + [0, 0, dist],
                   p_cut + aisle * dist * 1.5, p_cut - f * dist * 0.6 + aisle * dist * 0.6 + [0, 0, up])
        for back in options:
            xs = self.line_ik(p_cut, back, s, r, x_cut, 8)
            if xs is None:
                continue
            if not cchk.path_ok(xs, **self.kw):
                why = "retreat: the truss in hand hits " + ",".join(sorted(cchk.hits(xs[-1], **self.kw)))[:50]
                continue
            ret = xs
            break
        if ret is None:
            return why
        # release pose: seen from above every carried fruit inside the crate walls, the lowest just over the rim
        # (RELEASE_ABOVE_M); then the jaw opens and the truss drops in.
        (cx, cy), (sx, sy, h) = ts.CRATE_CENTRE, ts.CRATE_SIZE
        point = truss_ik.cut_point_in_gripper(r)
        radii = np.array([fr["radius"] for fr in self.t["fruit"]])
        inner = np.array([sx / 2 - scene.CRATE_WALL, sy / 2 - scene.CRATE_WALL])
        best = None
        # ⚠️ Sampling (pitch, tilt, spot) at random had no IK in 97 of 120 tries over the crate (2026-11-03). So the
        # candidates are sampled arm poses already over the crate (crate_poses): for each, where the held truss's
        # fruit would be; those that fit, most room to the walls first, are collision-checked with the lowering.
        ikf = self.ik_free
        rel = np.array([q for (_, q) in carry[1][1:]])                       # fruit centres in the gripper frame
        # ...and the stalk's ends: ⚠️ with fruit-only fits the long bare peduncle stuck out over a wall and dragged the
        # first fruit outside as it landed (seed 200002, 2026-11-03)
        Rt, pt = carry[1][0]
        stalk = np.array([pt + Rt @ (q - self.t["knuckle"]) for q in
                          (self.t["knuckle"], self.t["peduncle_end"], self.t["rachis_end"])])
        rel = np.vstack([rel, stalk])
        radii = np.r_[radii, [self.t["r"]] * len(stalk)]
        Qs, Os, Rs = crate_poses(ikf, point)
        G = Os[:, None, :] + np.einsum("nij,kj->nki", Rs, rel)              # (poses, fruit, xyz)
        slack = (inner[None, None, :] - (np.abs(G[:, :, :2] - [cx, cy]) + radii[None, :, None])).min((1, 2))
        low_z = (G[:, :, 2] - radii).min(1) - h                             # lowest fruit over the rim
        ok = np.flatnonzero((slack >= RELEASE_SLACK_M) & (low_z >= RELEASE_ABOVE_M[0]) & (low_z <= RELEASE_ABOVE_M[1] + 0.04))
        for i in ok[np.argsort(-slack[ok])][:RELEASE_SAMPLES // 4]:
            xr = np.asarray(Qs[i])
            p = Os[i] + Rs[i] @ point
            pitch = np.degrees(np.arcsin(np.clip(Rs[i][2, 2], -1, 1)))
            tilt = np.degrees(np.arcsin(np.clip(Rs[i][2, 0], -1, 1)))
            if not cchk.free(xr, soft_clearance=CARRY_CLEAR_M):
                continue
            low = [ikf.solve(p + [0, 0, ABOVE_CRATE_M * (1 - v)], pitch, tilt, point, x0=xr) for v in np.linspace(0, 1, 6)]
            if any(x is None for x in low) or not cchk.path_ok(low[:-1] + [xr], soft_clearance=CARRY_CLEAR_M):
                # ⚠️ the 5-joint arm often can't keep the release orientation along a straight 8 cm drop (22 of 22
                # fitting poses for seed 200000, 2026-11-03): then RRT goes straight to the release pose
                low = [xr]
            else:
                low[-1] = xr
            best = (float(slack[i]), xr, low[0], p, pitch, tilt, low)
            break
        if best is None:
            return "no release pose that fits the truss in the crate"
        _, x_rel, x_above, p, pitch, tilt, lower = best
        hang = float(-(self._carried_fruit(x_rel, carry)[:, 2] - p[2]).min())
        to_crate = None
        ckw = dict(soft_clearance=CARRY_CLEAR_M)
        for _ in range(RRT_SEEDS):
            to_crate = rrt.plan(cchk, ret[-1], x_above, rng, lo=self.lo, hi=self.hi, **ckw)
            if to_crate is not None:
                break
        if to_crate is None:
            return "retreat -> crate: no RRT route"
        self.jaw(JAW_OPEN_DEG)
        q_start = self.fm.norm_to_rad(np.asarray(scene.START_NORM, float))[:5]
        home = self.route(lower[-1], q_start, rng, ignore=self.target_all, firm=False)
        if home is None:
            return "crate -> start: no RRT route"
        return dict(ret=ret, to_crate=to_crate, lower=lower, home=home, release_point=p.tolist(),
                    release_pitch=round(pitch, 1), hang_mm=round(hang * 1000, 1), crate_slack_mm=round(best[0] * 1000, 1))

    # --- frames ---
    def _assemble(self, rng, to_pre, slide, miss_leg, rest, meta, slide_m):
        fm = self.fm
        speed = float(rng.uniform(*SPEED))
        rows, servo, phases = [], [], []
        cur = np.asarray(fm.rad_to_norm(np.r_[self.q_now(), self.d.qpos[self.g_adr]]), float)
        g_open = float(fm.rad_to_norm(np.r_[np.zeros(5), np.radians(JAW_OPEN_DEG)])[5])
        g_shut = float(rng.uniform(*expert.CLOSE_NORM))
        g_start = float(scene.START_NORM[5])
        n5 = lambda x: fm.rad_to_norm(np.r_[x, 0])[:5]   # noqa: E731

        def last():
            return np.asarray(rows[-1] if rows else cur, float)

        def hold(sec, name, sv=False):
            n = int(round(sec * FPS))
            rows.extend([last()] * n); servo.extend([sv] * n); phases.append((name, n))

        def path(xs, grip, name, sv=False, grip_to=None, cart=None):
            """Min-jerk along joint way-points xs (rad) with the jaw at `grip` (or ramping to grip_to). cart: the
            move's length in m for straight Cartesian legs (timed by CART_PEAK_M_S), else joint-space timing."""
            a = last()
            pts = [a[:5]] + [n5(x) for x in xs]
            seg = np.array([np.abs(q1 - q0).max() for q0, q1 in zip(pts, pts[1:])])
            L = max(seg.sum(), 1e-6)
            if cart is None:
                T = 1.875 * L / (JOINT_PEAK_NORM_S * speed)
                T = float(np.clip(T * rng.uniform(0.9, 1.1), 0.6, 6.0))
            else:
                T = float(np.clip(1.875 * cart / (CART_PEAK_M_S * speed) * rng.uniform(0.9, 1.1), 0.5, 3.0))
            sgrid = expert.min_jerk(T * FPS)[1:] * L
            cum = np.r_[0, np.cumsum(seg)]
            P = np.array(pts)
            g0 = a[5]
            g1 = grip if grip_to is None else grip_to
            for k, si in enumerate(sgrid):
                arm = np.array([np.interp(si, cum, P[:, j]) for j in range(5)])
                g = grip if grip_to is None else g0 + (g1 - g0) * min(1.0, si / L)
                rows.append(np.r_[arm, g])
            servo.extend([sv] * len(sgrid)); phases.append((name, len(sgrid)))

        def jaw_to(g, sec, name):
            a = last()
            n = max(2, int(round(sec * FPS)))
            for si in expert.min_jerk(n)[1:]:
                rows.append(np.r_[a[:5], a[5] + (g - a[5]) * si])
            servo.extend([True] * (n - 1)); phases.append((name, n - 1))

        hold(rng.uniform(*expert.HOLD_START_S), "hold_start")
        path(to_pre[1:], g_start, "to_precut", grip_to=g_open)
        if miss_leg is not None:
            path(miss_leg["xs"][1:], g_open, "miss_slide", sv=True, cart=slide_m - miss_leg["short"])
            jaw_to(g_shut, rng.uniform(*CLOSE_S), "miss_close")
            hold(rng.uniform(*MISS_PAUSE_S), "miss_pause", sv=True)
            jaw_to(g_open, 0.5, "miss_open")
            path(miss_leg["xs"][::-1][1:], g_open, "miss_back_off", sv=True, cart=slide_m - miss_leg["short"])
        path(slide[1:], g_open, "slide_in", sv=True, cart=slide_m)
        hold(rng.uniform(0.15, 0.3), "settle_at_stalk", sv=True)
        jaw_to(g_shut, rng.uniform(*CLOSE_S), "close_cut")
        hold(rng.uniform(*SQUEEZE_S), "squeeze", sv=True)
        path(rest["ret"][1:], g_shut, "retreat", cart=float(np.linalg.norm(np.r_[RETREAT_M[1], RETREAT_UP_M[1]])))
        path(rest["to_crate"][1:], g_shut, "to_crate")
        if len(rest["lower"]) > 1:
            path(rest["lower"][1:], g_shut, "lower", cart=ABOVE_CRATE_M)
        jaw_to(g_open, 0.5, "release")
        hold(rng.uniform(*DROP_S), "drop")
        path(rest["home"][1:], g_open, "return", grip_to=g_start)
        hold(rng.uniform(*expert.HOLD_END_S), "hold_end")
        meta.update(target=self.t["name"], n_fruit=self.t["n"], stalk_mm=round(self.t["r"] * 2000, 1),
                    miss=miss_leg is not None, miss_short_mm=None if miss_leg is None else round(miss_leg["short"] * 1000, 1),
                    speed=round(speed, 3), release_point=rest["release_point"], release_pitch=rest["release_pitch"],
                    hang_mm=rest["hang_mm"], rrt_waypoints=[len(to_pre), len(rest["to_crate"]), len(rest["home"])],
                    frames=len(rows), seconds=round(len(rows) / FPS, 2))
        return expert.Plan(np.array(rows), np.array(servo), phases, meta)
