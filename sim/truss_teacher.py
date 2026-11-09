"""Teacher v3: the scripted truss picker as a closed-loop teacher (post-mortem §6.1, K7 approved 2026-11-09).

    teacher = TrussTeacher(robot.m, robot.d, robot.info, k).plan()      # after the start frame; PlanError if none
    while not teacher.done:
        robot.send_action(teacher.act(state_norm))                      # the measured state, every frame

What changed from truss_expert.TrussPlanner (the v2 expert, which recorded blush_sim_truss_v2):
- Same scene, same motion. No random speed, holds, durations, cut point, approach order or retreat length, and no
  deliberate misses. Every choice comes from a fixed priority list checked against the scene (first clear one
  wins: CUT_U, APPROACH_BLEND_DEG, SLIDE_M, STANDOFF_M, retreat options), so it depends only on the scene.
  The post-mortem's R2: the camera can't see a coin-flip, so the policy averages over them.
- Closed loop. act() works from the MEASURED arm state every frame: it projects the state onto the current leg's
  joint path and commands a point a fixed lead further along it (pure pursuit), so after a push it steers back to
  the path instead of replaying a timed trajectory. That is what makes noise injection (§6.2) and DAgger labels
  (§6.4) possible. resync() re-enters the right leg from a state the teacher didn't drive to (DAgger, later).
- RRT checks, it doesn't drive. ⚠️ Supersedes "RRT for every arm move" (K7 2026-11-01) for the teacher, by K7's
  approval of §6.1 on 2026-11-09: every free move is a fixed template (a straight joint-space line), checked with
  rrt.Checker; a blocked template gets ONE fixed detour, else the candidate is dropped. The drop rate is measured
  (check_teacher.py solve), because a consistent teacher that can't solve some scenes shrinks what the data covers
  (Motion Policy Networks).
- The cut lands where the cue says. ⚠️ v2 cut at a random point 20-80 % along the cut zone while the drawn cross
  and observation.environment_state always marked 50 % (truss_target.cut_pixel, 2026-11-09 finding). The teacher
  tries CUT_U in order, 0.5 first, and reports the one it used (meta["cut_u"]); the recorder must draw and record
  the cut point at that u.

The motion (§6.1 sketch):
  start -> standoff (STANDOFF_M behind the pre-cut pose, straight joint line) -> pre-cut pose (SLIDE_M back along
  the approach axis, straight Cartesian line) -> slide straight in, slowly -> close, squeeze (fixed times) ->
  retreat straight back and up -> over to the release pose above the crate -> open, wait -> back to start ->
  short hold. Each free move (start -> standoff, retreat -> crate, crate -> start) is a straight joint line, or when
  that is blocked the one detour: up over a fixed via-pose high in front of the row, then down (VIA_NORM), or
  two such vias chained.
  ⚠️ §6.1 named a fixed via-pose over the crate; one constant pose 20 cm over the crate's centre hit a neighbour
  truss's fruit with the cut truss in hand in 2 of 3 test scenes (2026-11-09), so the carry uses the same rule.

Reused from truss_expert.py (B.L.U.S.H commit 6d1e02d): the obstacle sets and checker (TrussPlanner.__init__), the
IK, the carried-truss checker (CarryChecker), the release-pose fit (crate_poses + laid_flat, deterministic: its
samples come from a fixed seed). Timing constants come from expert.py; the sag correction is expert.Plan's,
applied to the cross-track error so the lead itself doesn't wind it up.
"""
import numpy as np

import arm_map
import expert
import scene
import truss_expert as tx
import truss_ik
import truss_scene as ts

FPS = expert.FPS
PlanError = tx.PlanError

# --- fixed priority lists (first that works wins; design choices, 2026-11-09) ---
CUT_U = (0.5, 0.4, 0.6, 0.3, 0.7)          # along the cut zone; 0.5 is where the cue points (truss_target.cut_pixel)
APPROACH_BLEND_DEG = tx.APPROACH_BLEND_DEG  # (0, 30, -30, 50, -50), now in this order, not shuffled
SLIDE_M = (tx.SLIDE_M,) + tx.SLIDE_FALLBACK_M   # 4.5 cm, then 3.5 / 2.5 cm at the reach edge (v2's values)
STANDOFF_M = (0.04, 0.02, 0.0)             # standoff this far behind the pre-cut pose (guess, unverified)
APPROACH_STEPS = 8                         # IK way-points from standoff to pre-cut (as the slide's 10, guess)
RETREAT_M, RETREAT_UP_M = 0.04, 0.03       # v2 drew 3-5 cm back and 2-4 cm up; now the middle of each
# Every free move: the template is one straight joint-space line; the one detour goes over a fixed via-pose high in
# front of the row (§6.1 "up, across, down"), the first of VIA_NORM that's clear. ⚠️ The trusses hang down and out
# into the aisle (fruit 3-22 cm high, out to 6 cm from the arm's base line) while the cut points sit near the stem at
# ~28 cm and the start pose low at 11 cm: a straight line from start to standoff crossed fruit in 3 of 3 test scenes,
# and rising straight up with the orientation held ran out of reach (2026-11-09). The v2 RRT's routes went over such
# a high pose (grip 34-46 cm up, 25-36 cm out) when the straight line was blocked (10 test scenes, 2026-11-09).
# Joint angles in norm units (arm_map, use_degrees=false), copied from three of those v2 routes; the grip point there
# in the comment. Tried nearest the move's goal first (joint space), so the choice depends only on the scene.
# ⚠️ Defined by IK as (point, pitch, tilt 0) first: unreachable, no detour was ever tried (2026-11-09).
VIA_NORM = ((-59.0, -44.0, 8.0, -34.0, 6.0),     # grip (0.27, 0.29, 0.46): high, left of centre (seed 9100009)
            (-54.0, -46.0, 44.0, -35.0, 19.0),   # grip (0.18, 0.36, 0.34): left (seed 9100005)
            (21.0, -64.0, -6.0, 76.0, 61.0),     # grip (0.42, 0.37, 0.32): right (seed 9100005)
            # Added from a search (2026-11-09): v2 RRT routes on seeds 9200000-9200059 (apart from the test seeds),
            # way-points pooled, the one that unblocks most scenes the line and the vias above can't reach kept.
            # 15 of 60 scenes were blocked start -> standoff; this pose unblocks 3, no other pose 2 more.
            (-90.0, -19.0, 41.0, -56.0, 22.0))   # gripper (0.18, 0.21, 0.23): the arm raised in place over its base
# ⚠️ 12 of those 15 scenes had no single clear via in 75 RRT way-points: their routes needed two. So the last
# fallback chains two vias (up out of the start, then over: §6.1's "up, across, down"), pairs in a fixed order.
# --- timing: fixed, the middle of v2's ranges ---
CLOSE_S = 0.65                             # v2 0.5-0.8
SQUEEZE_S = 0.75                           # v2 0.6-0.9 (the cut lands ~0.45 s into the close)
SETTLE_S = 0.2                             # v2 0.15-0.3 at the stalk before closing
RELEASE_S = 0.5                            # jaw opening over the crate (v2's fixed 0.5)
DROP_S = 0.65                              # v2 0.5-0.8
HOLD_END_S = 0.3                           # short: identical frames at home froze the n10 ACT run (post-mortem §2)
CLOSE_NORM = 1.0                           # the middle of expert.CLOSE_NORM (0-2)
# --- closed loop ---
# Lead of the commanded point along the path, norm units per frame. expert.JOINT_PEAK_NORM_S 60 /s = 2 per frame;
# carrying at tx.CARRY_SPEED. Cartesian legs are timed by CART_M_S along the line (v2's min-jerk average for
# CART_PEAK_M_S 0.08 m/s: 0.08 / 1.875).
FREE_LEAD = expert.JOINT_PEAK_NORM_S / FPS
CARRY_LEAD = FREE_LEAD * tx.CARRY_SPEED
CART_M_S = expert.CART_PEAK_M_S / 1.875
APPROACH_M_S = 2 * CART_M_S                # standoff -> pre-cut: nothing in reach yet, twice the slide's pace (guess)
END_TOL = {"servo": 0.5, "free": 2.0}      # norm units from a leg's end to call it reached (guess)
STALL_NORM, STALL_FRAMES = 0.05, 8         # ...or the arm moved less than this per frame for this many (sag, contact)
SEARCH_BACK = 0.5                          # projection window: this many leads behind the last projection...
SEARCH_AHEAD = 6.0                         # ...and this many ahead (keeps it from jumping where a path folds back)
MAX_S = 45.0                               # done after this long whatever happens (a stuck leg can't loop forever)


class Leg:
    """One piece of the motion: a joint path (norm units, (n, 5)) or a hold, the jaw command, the pace."""

    def __init__(self, name, W=None, hold=None, jaw=None, jaw_to=None, lead=FREE_LEAD, servo=False, frames=None,
                 checker_ok=True):
        self.name, self.lead, self.servo, self.frames = name, lead, servo, frames
        self.W = None if W is None else np.asarray(W, float)
        self.hold = None if hold is None else np.asarray(hold, float)
        self.jaw, self.jaw_to = jaw, jaw_to         # jaw: fixed command; jaw_to: (from, to) ramped over the leg
        if self.W is not None:
            seg = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
            self.cum = np.r_[0.0, np.cumsum(seg)]
            self.L = float(self.cum[-1])

    def point(self, s):
        s = float(np.clip(s, 0.0, self.L))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.W) - 2))
        seg = self.cum[i + 1] - self.cum[i]
        u = 0.0 if seg < 1e-12 else (s - self.cum[i]) / seg
        return self.W[i] + (self.W[i + 1] - self.W[i]) * u

    def project(self, q, s_lo, s_hi):
        """Arc length of the point on the path nearest q, searched in [s_lo, s_hi]."""
        best, best_s = np.inf, s_lo
        for i in range(len(self.W) - 1):
            a, b = self.W[i], self.W[i + 1]
            c0, c1 = self.cum[i], self.cum[i + 1]
            if c1 < s_lo or c0 > s_hi:
                continue
            ab = b - a
            n2 = float(ab @ ab)
            u = 0.0 if n2 < 1e-12 else float(np.clip((q - a) @ ab / n2, 0, 1))
            s = float(np.clip(c0 + (c1 - c0) * u, s_lo, s_hi))
            dist = np.linalg.norm(q - self.point(s))
            if dist < best:
                best, best_s = dist, s
        return best_s


class TrussTeacher(tx.TrussPlanner):
    """TrussPlanner's scene sets, checker and IK; deterministic legs instead of its sampled plan."""

    def __init__(self, m, d, info, k):
        super().__init__(m, d, info, k)
        self.legs, self.phases, self.meta = [], [], {}
        self.q_home = np.asarray(scene.START_NORM, float)
        self._vias = []

    # --- planning (once, at the start frame) ---
    def plan(self):
        why = []
        for mode in tx.NEIGHBOUR_CLEAR:
            self.kw = dict(soft_clearance=tx.NEIGHBOUR_CLEAR[mode], firm_clearance=tx.OWN_FRUIT_CLEAR[mode])
            try:
                self._plan(mode)
                return self
            except PlanError as e:
                why.append(f"{mode}: {e}")
        raise PlanError(" / ".join(why))

    def _plan(self, mode):
        tried = []
        r = self.t["r"]
        q0 = self.q_now()
        self.jaw(tx.JAW_OPEN_DEG)
        if not self.chk.free(q0, **self.kw):
            raise PlanError("start pose not clear")
        for u in CUT_U:
            for beta in APPROACH_BLEND_DEG:
                leg = self._cut_leg(u, beta, r)
                if isinstance(leg, str):
                    tried.append(leg)
                    continue
                to_std, detour = self._to_standoff(q0, leg)
                if to_std is None:
                    tried.append("start -> standoff blocked, detour too")
                    continue
                rest = self._after_cut(leg, r)
                if isinstance(rest, str):
                    tried.append(rest)
                    continue
                self._build_legs(q0, to_std, leg, rest)
                self.meta = dict(teacher="v3", neighbours=mode, cut_u=u, approach_blend_deg=float(beta),
                                 slide_mm=round(leg["slide_m"] * 1000), standoff_mm=round(leg["standoff_m"] * 1000),
                                 detours=dict(to_standoff=detour, to_crate=rest["detour_crate"],
                                              home=rest["detour_home"]),
                                 retreat_option=rest["retreat_option"], release_point=rest["release_point"],
                                 release_pitch=rest["release_pitch"], crate_slack_mm=rest["crate_slack_mm"],
                                 target=self.t["name"], n_fruit=self.t["n"], stalk_mm=round(r * 2000, 1),
                                 tried=sorted(set(tried)), miss=False)
                return
        raise PlanError("; ".join(sorted(set(tried))) or "no candidate")

    def _cut_leg(self, u, beta, r):
        """Cut pose at u along the cut zone, the slide in along the approach blended beta deg toward the stalk, and the
        standoff behind it. Same checks as TrussPlanner._cut_leg, plus the approach to the pre-cut pose."""
        p_cut, s = ts.cut_target_now(self.t, self.d, self.m, u)
        x_cut = self.ik.solve(p_cut, s, r)
        if x_cut is None:
            return "no cut pose"
        _, R = self.ik.frame(x_cut, r)
        b = np.radians(beta)
        f = np.cos(b) * -R[:, 2] + np.sin(b) * R[:, 1]
        slide = None
        for L in SLIDE_M:
            slide = self.line_ik(p_cut - f * L, p_cut, s, r, x_cut, tx.SLIDE_STEPS)
            if slide is not None:
                break
        if slide is None:
            return "no IK along the slide"
        self.jaw(tx.JAW_OPEN_DEG)
        if not self.chk.path_ok(slide, ignore=self.target_stalk, **self.kw):
            return "slide hits"
        for deg in np.linspace(tx.JAW_OPEN_DEG, -10.0, 6):
            self.jaw(deg)
            if self.chk.hits(slide[-1], ignore=self.target_stalk, **self.kw):
                self.jaw(tx.JAW_OPEN_DEG)
                return "closing jaw hits a neighbour"
        self.jaw(tx.JAW_OPEN_DEG)
        approaches = []                       # every standoff that works, in STANDOFF_M order
        for so in STANDOFF_M:
            if so == 0.0:
                approaches.append(([slide[0]], 0.0))
                continue
            xs = self.line_ik(p_cut - f * (L + so), p_cut - f * L, s, r, slide[0], APPROACH_STEPS)
            if xs is not None and self.chk.path_ok(xs, **self.kw):
                approaches.append((xs, so))
        return dict(x_cut=x_cut, p_cut=p_cut, s=s, f=f, slide=slide, slide_m=L, approaches=approaches, u=u,
                    beta=beta)

    def vias(self):
        """The fixed via-poses, 5 joints in rad (VIA_NORM)."""
        if not self._vias:
            self._vias = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in VIA_NORM]
        return self._vias

    def free_move(self, xa, xb, chk, r, **kw):
        """The template (one straight joint line xa -> xb), else the detour over a fixed via-pose (VIA_NORM, nearest
        xb first), else two of them chained. Returns (way-points, 0 for the line, the via's number from 1, or
        10 * first + second for a chain) or (None, None)."""
        if chk.path_ok([xa, xb], **kw):
            return [xa, xb], 0
        vias = self.vias()
        dist = lambda u, v: float(np.abs(u - v).max())   # noqa: E731
        for i in sorted(range(len(vias)), key=lambda j: dist(vias[j], xb)):
            if chk.path_ok([xa, vias[i], xb], **kw):
                return [xa, vias[i], xb], i + 1
        pairs = sorted(((i, j) for i in range(len(vias)) for j in range(len(vias)) if i != j),
                       key=lambda ij: dist(xa, vias[ij[0]]) + dist(vias[ij[0]], vias[ij[1]]) + dist(vias[ij[1]], xb))
        for i, j in pairs:
            if chk.path_ok([xa, vias[i], vias[j], xb], **kw):
                return [xa, vias[i], vias[j], xb], 10 * (i + 1) + j + 1
        return None, None

    def _to_standoff(self, q0, leg):
        """The first standoff (STANDOFF_M order) the arm can get to from the start; sets leg approach/standoff_m."""
        self.jaw(tx.JAW_OPEN_DEG)
        for xs, so in leg["approaches"]:
            path, how = self.free_move(q0, xs[0], self.chk, self.t["r"], **self.kw)
            if path is not None:
                leg["approach"], leg["standoff_m"] = xs, so
                return path, how
        return None, None

    def _after_cut(self, leg, r):
        """Retreat, the release pose, the carry onto it, the way home. A str says why not."""
        x_cut, p_cut, f, s = leg["x_cut"], leg["p_cut"], leg["f"], leg["s"]
        carry = self._carry(x_cut)
        point = truss_ik.cut_point_in_gripper(r)
        cchk = tx.CarryChecker(self.m, self.d, carry[0], carry[1], point, hard=self.hard, soft=self.soft)
        cchk.d.qpos[self.g_adr] = np.radians(-10.0)
        self.jaw(-10.0)
        aisle = np.array([0, -np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]), 0.0])
        dist, up = RETREAT_M, RETREAT_UP_M
        options = (p_cut - f * dist + [0, 0, up], p_cut + aisle * dist + [0, 0, up], p_cut + [0, 0, dist],
                   p_cut + aisle * dist * 1.5, p_cut - f * dist * 0.6 + aisle * dist * 0.6 + [0, 0, up])
        ret, opt, why = None, None, "no IK for the retreat"
        for i, back in enumerate(options):
            xs = self.line_ik(p_cut, back, s, r, x_cut, 8)
            if xs is None:
                continue
            if not cchk.path_ok(xs, **self.kw):
                why = "retreat: the truss in hand hits"
                continue
            ret, opt = xs, i
            break
        if ret is None:
            return why
        rel = self._release(carry, cchk, r)
        if isinstance(rel, str):
            return rel
        x_rel, x_above, lower, p, pitch, slack = rel
        to_crate, detour_crate = None, None
        for clear in (tx.CARRY_CLEAR_M, tx.CARRY_CLEAR_MIN_M):   # the retreat can end closer than 2 cm to a neighbour
            to_crate, detour_crate = self.free_move(ret[-1], x_above, cchk, r, soft_clearance=clear)
            if to_crate is not None:
                break
        if to_crate is None:
            return "retreat -> crate blocked, via-pose too"
        self.jaw(tx.JAW_OPEN_DEG)
        q_start = self.fm.norm_to_rad(self.q_home)[:5]
        home, detour_home = self.free_move(lower[-1], q_start, self.chk, r, ignore=self.target_all, firm=False,
                                           **self.kw)
        if home is None:
            return "crate -> start blocked, via-pose too"
        return dict(ret=ret, retreat_option=opt, to_crate=to_crate, lower=lower, home=home,
                    detour_crate=detour_crate, detour_home=detour_home, release_point=np.round(p, 4).tolist(),
                    release_pitch=round(pitch, 1), crate_slack_mm=round(slack * 1000, 1))

    def _release(self, carry, cchk, r):
        """The release pose: TrussPlanner._after_cut's fit (seen from above every carried fruit and the stalk inside
        the walls, also laid flat; most room to the walls first), deterministic. Returns (x_rel, x_above, lower,
        grip point, pitch, slack) or why not."""
        (cx, cy), (sx, sy, h) = ts.CRATE_CENTRE, ts.CRATE_SIZE
        point = truss_ik.cut_point_in_gripper(r)
        radii = np.array([fr["radius"] for fr in self.t["fruit"]])
        inner = np.array([sx / 2 - scene.CRATE_WALL, sy / 2 - scene.CRATE_WALL])
        ikf = self.ik_free
        rel = np.array([q for (_, q) in carry[1][1:]])
        Rt, pt = carry[1][0]
        stalk = np.array([pt + Rt @ (q - self.t["knuckle"]) for q in
                          (self.t["knuckle"], self.t["peduncle_end"], self.t["rachis_end"])])
        rel = np.vstack([rel, stalk])
        radii = np.r_[radii, [r] * len(stalk)]
        Qs, Os, Rs = tx.crate_poses(ikf, point)
        G = Os[:, None, :] + np.einsum("nij,kj->nki", Rs, rel)
        held = (inner[None, None, :] - (np.abs(G[:, :, :2] - [cx, cy]) + radii[None, :, None])).min((1, 2))
        flat = (inner[None, None, :] - (np.abs(tx.laid_flat(G) - [cx, cy]) + radii[None, :, None])).min((1, 2))
        slack = np.minimum(held, flat + tx.LEAN_M)
        low_z = (G[:, :, 2] - radii).min(1) - h
        high_z = (G[:, :, 2] + radii).max(1) - h
        ok = np.flatnonzero((slack >= tx.RELEASE_SLACK_M) & (low_z >= scene.CRATE_WALL + 0.01 - h)
                            & (low_z <= tx.RELEASE_ABOVE_M[1] + 0.04) & (high_z <= tx.ABOVE_RIM_MAX_M))
        for i in ok[np.argsort(-slack[ok], kind="stable")][:tx.RELEASE_SAMPLES // 4]:
            xr = np.asarray(Qs[i])
            p = Os[i] + Rs[i] @ point
            pitch = np.degrees(np.arcsin(np.clip(Rs[i][2, 2], -1, 1)))
            tilt = np.degrees(np.arcsin(np.clip(Rs[i][2, 0], -1, 1)))
            if cchk.hits(xr, soft_clearance=tx.CARRY_CLEAR_M):
                continue
            low = [ikf.solve(p + [0, 0, tx.ABOVE_CRATE_M * (1 - v)], pitch, tilt, point, x0=xr)
                   for v in np.linspace(0, 1, 6)]
            if any(x is None for x in low) or not cchk.path_ok(low[:-1] + [xr], soft_clearance=tx.CARRY_CLEAR_M):
                low = [xr]                     # no straight drop for this pose: the carry goes to it directly
            else:
                low[-1] = xr
            return xr, low[0], low, p, pitch, float(slack[i])
        return f"no release pose: {len(ok)} of {len(Qs)} poses over the crate fit the truss"

    def _build_legs(self, q0, to_std, leg, rest):
        fm = self.fm
        n5 = lambda xs: np.array([fm.rad_to_norm(np.r_[x, 0])[:5] for x in xs])   # noqa: E731
        g_open = float(fm.rad_to_norm(np.r_[np.zeros(5), np.radians(tx.JAW_OPEN_DEG)])[5])
        g_start = float(self.q_home[5])
        g_now = float(fm.rad_to_norm(np.r_[np.zeros(5), self.d.qpos[self.g_adr]])[5])
        self.g_open, self.g_shut = g_open, CLOSE_NORM

        def cart_lead(xs, metres, speed):
            W = n5(xs)
            L = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
            return max(L / max(metres / speed * FPS, 1.0), 1e-3)

        f = lambda s: max(1, int(round(s * FPS)))   # noqa: E731
        cut = n5([leg["x_cut"]])[0]
        L = []
        L.append(Leg("to_standoff", n5(to_std), jaw_to=(g_now, g_open), lead=FREE_LEAD))
        if len(leg["approach"]) > 1:
            L.append(Leg("approach", n5(leg["approach"]), jaw=g_open, servo=True,
                         lead=cart_lead(leg["approach"], leg["standoff_m"], APPROACH_M_S)))
        L.append(Leg("slide_in", n5(leg["slide"]), jaw=g_open, servo=True,
                     lead=cart_lead(leg["slide"], leg["slide_m"], CART_M_S)))
        L.append(Leg("settle_at_stalk", hold=cut, jaw=g_open, servo=True, frames=f(SETTLE_S)))
        L.append(Leg("close_cut", hold=cut, jaw_to=(g_open, CLOSE_NORM), servo=True, frames=f(CLOSE_S)))
        L.append(Leg("squeeze", hold=cut, jaw=CLOSE_NORM, servo=True, frames=f(SQUEEZE_S)))
        ret_m = float(np.linalg.norm([RETREAT_M, RETREAT_UP_M]))
        L.append(Leg("retreat", n5(rest["ret"]), jaw=CLOSE_NORM, lead=cart_lead(rest["ret"], ret_m, CART_M_S)))
        L.append(Leg("to_crate", n5(rest["to_crate"]), jaw=CLOSE_NORM, lead=CARRY_LEAD))
        if len(rest["lower"]) > 1:
            L.append(Leg("lower", n5(rest["lower"]), jaw=CLOSE_NORM,
                         lead=cart_lead(rest["lower"], tx.ABOVE_CRATE_M, CART_M_S)))
        x_rel = n5([rest["lower"][-1]])[0]
        L.append(Leg("release", hold=x_rel, jaw_to=(CLOSE_NORM, g_open), frames=f(RELEASE_S)))
        L.append(Leg("drop", hold=x_rel, jaw=g_open, frames=f(DROP_S)))
        L.append(Leg("return", n5(rest["home"]), jaw_to=(g_open, g_start), lead=FREE_LEAD))
        L.append(Leg("hold_end", hold=self.q_home[:5], jaw=g_start, frames=f(HOLD_END_S)))
        self.legs = L
        self.reset()

    # --- running (every frame) ---
    def reset(self):
        self.i, self.n_in_leg, self.n_total = 0, 0, 0
        self.s, self.I, self.stall, self.q_prev = 0.0, np.zeros(5), 0, None
        self.phases = []

    @property
    def done(self):
        return self.i >= len(self.legs) or self.n_total >= MAX_S * FPS

    @property
    def leg(self):
        return self.legs[min(self.i, len(self.legs) - 1)].name

    def _next_leg(self):
        self.phases.append((self.legs[self.i].name, self.n_in_leg))
        self.i += 1
        self.n_in_leg, self.s, self.stall, self.q_prev = 0, 0.0, 0, None
        if self.done:
            self.meta.update(frames=self.n_total, seconds=round(self.n_total / FPS, 2),
                             timed_out=self.i < len(self.legs))

    def act(self, state_norm):
        """The action for this frame from the measured state (norm, 6): {joint.pos: norm}."""
        q = np.asarray(state_norm, float)[:5]
        g = self.legs[self.i]
        if g.W is not None:
            s = g.project(q, self.s - SEARCH_BACK * g.lead, self.s + SEARCH_AHEAD * g.lead)
            self.s = s
            target = g.point(s + g.lead)
            ref = g.point(s)
            prog = 1.0 if g.L < 1e-9 else min(1.0, (s + g.lead) / g.L)
        else:
            target = ref = g.hold
            prog = min(1.0, (self.n_in_leg + 1) / g.frames)
        if g.servo:
            # the sag correction on the cross-track error (path point nearest the arm minus the arm), not on the lead
            self.I = np.clip(self.I + expert.SAG_KI * (ref - q), -expert.SAG_MAX_NORM, expert.SAG_MAX_NORM)
        else:
            self.I *= 0.9
        jaw = g.jaw if g.jaw_to is None else g.jaw_to[0] + (g.jaw_to[1] - g.jaw_to[0]) * prog
        a = np.r_[target + self.I, jaw]
        self.n_in_leg += 1
        self.n_total += 1
        # leg finished?
        if g.W is not None:
            end = g.W[-1]
            moved = np.inf if self.q_prev is None else float(np.abs(q - self.q_prev).max())
            self.stall = self.stall + 1 if moved < STALL_NORM else 0
            self.q_prev = q
            at_end = self.s + g.lead >= g.L
            close = float(np.abs(q - end).max()) < END_TOL["servo" if g.servo else "free"]
            if at_end and (close or self.stall >= STALL_FRAMES):
                self._next_leg()
        elif self.n_in_leg >= g.frames:
            self._next_leg()
        if not self.done and self.n_total >= MAX_S * FPS:
            self.meta.update(frames=self.n_total, seconds=round(self.n_total / FPS, 2), timed_out=True)
        return {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, a)}

    def resync(self, state_norm, cut, released):
        """Re-enter from a state the teacher didn't drive to (a DAgger takeover): the eligible leg whose path passes
        nearest the arm. cut: the target's stalk is cut; released: it lies in the crate. ⚠️ Not used by step 0's
        checks yet; written now so DAgger (§6.4) has one entry point."""
        q = np.asarray(state_norm, float)[:5]
        names = [g.name for g in self.legs]
        if released:
            pool = [names.index("return")]
        elif cut:
            pool = [k for k, n in enumerate(names) if n in ("squeeze", "retreat", "to_crate", "lower")]
        else:
            pool = [k for k, n in enumerate(names) if n in ("to_standoff", "approach", "slide_in")]

        def dist(k):
            g = self.legs[k]
            if g.W is None:
                return float(np.linalg.norm(q - g.hold))
            return float(np.linalg.norm(q - g.point(g.project(q, 0.0, g.L))))
        k = min(pool, key=dist)
        self.i, self.n_in_leg, self.stall, self.q_prev = k, 0, 0, None
        g = self.legs[k]
        self.s = 0.0 if g.W is None else g.project(q, 0.0, g.L)
        self.I = np.zeros(5)
        return g.name
