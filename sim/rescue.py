"""When the policy is failing: the gate of eval_truss.py --hybrid --trigger failing (K7 2026-11-19: "the teacher
should only jump in if the robot is failing, if it is actually on track to getting the truss then the teacher should
not jump in, basically like dagger").

The policy drives; every frame Gate.step asks "is it failing?" and the first yes hands the arm to the teacher, who
finishes the scene (TrussTeacher.rejoin before the cut, resync on its carry legs after it). A scene the policy does on
its own is never touched. This is HG-DAgger's gate (a supervisor takes over only when it sees the learner going
wrong); with eval_truss.py --record the takeovers become DAgger data (only scenes with a takeover are recorded).

Failing means one of (checked in this order, every frame):
  harm      before the cut: a neighbouring truss's fruit swung more than HARM_NEIGHBOUR_M (the data filter's
            limit, so the teacher's own brushing never counts), the arm pressed the greenhouse / crate / table harder than gt.STRAIN_HARD_N, or a
            servo sat stalled at its force limit gt.STRAIN_STALL_FRAMES frames in a row (pressing on something)
  stuck     before the cut: for PROGRESS_S the jaw got no PROGRESS_M closer to where the cut point hangs now
  late      not cut LATE_X x + LATE_PAD_S after the teacher's own plan would have cut, or (after the cut) not in the
            crate that long after its plan's drop (the plan's times are estimated from its legs, plan_times)
Not rescued: a wrong truss cut or fruit off (already failed, nothing to save), a truss dropped outside the crate.

⚠️ Thresholds are calibrated on the teacher itself: with eval_truss.py --teacher-drives the teacher plays every scene
from the start and the gate only logs when it WOULD have fired; on track means it never fires on the teacher.
Check of 2026-11-19 (heldout_v5.jsonl, first 20 scenes, outputs/gate_check_teacher.log): before the cut nothing
fired; a "stuck" rule for the carry fired in 7 of 20 (squeeze + retreat move the truss sideways for over 4 s before
the carry heads for the crate), so the carry has the late rule only; the teacher moved neighbours up to 10 mm, so
harm starts at the data filter's 15 mm, not at an earlier 10 mm; the plan's times run long (the teacher cut at
0.80-0.88 x plan_cut_s and reached the crate at 0.79-0.89 x plan_drop_s), so late never comes near the teacher.
"""
import numpy as np

import expert
import generate_truss as gt
import truss_scene as ts

PROGRESS_M = 0.005        # "closer" means at least this much nearer than the best so far (guess, unverified)
PROGRESS_S = 4.0          # no progress for this long = stuck (guess; never fired on the teacher before the cut, 20/20)
HARM_NEIGHBOUR_M = gt.NEIGHBOUR_MOVE_M   # 15 mm, the data filter's limit (the teacher itself reaches 10 mm)
LATE_X, LATE_PAD_S = 1.5, 5.0   # late = 1.5 x the plan's time + 5 s (guess; the policies' successes ran on the
#                                 teacher's schedule: crate at 12-14 s, teacher_s ~15 s, 2026-11-15)


def plan_times(T):
    """Seconds from the start to the teacher's cut (end of close_cut) and to its drop into the crate (end of drop),
    estimated from its legs: a path leg takes its length / lead frames (act() moves the target `lead` along the path
    each frame), a hold leg its frames."""
    t, cut_t, drop_t = 0.0, None, None
    for g in T.legs:
        n = g.frames if g.W is None else (g.L / max(g.lead, 1e-9))
        t += float(n or 0) / expert.FPS
        if g.name == "close_cut":
            cut_t = t
        if g.name == "drop":
            drop_t = t
    return cut_t, drop_t


class Gate:
    def __init__(self, robot, k, T, watch):
        self.robot, self.k, self.watch = robot, k, watch
        self.plan_cut_s, self.plan_drop_s = plan_times(T)
        self.best, self.t_best = np.inf, 0.0
        self.t_cut = None

    def step(self, t, jaw_pt):
        """t: seconds since the scene started; jaw_pt: the jaw's cut point (world). Returns why it is failing, or
        None while it is on track. Call once per frame, after watch.step()."""
        r, w = self.robot, self.watch
        if self.t_cut is None and r.cutters[self.k].cut_t is not None:
            self.t_cut = t
        if self.t_cut is not None:                       # the carry: late only (see the top of the file)
            if ts.in_crate(r.info, r.d, r.m, self.k) or self.plan_drop_s is None or self.plan_cut_s is None:
                return None                              # delivered, or nothing to time it by
            limit = self.t_cut + LATE_X * (self.plan_drop_s - self.plan_cut_s) + LATE_PAD_S
            return f"late: not in the crate at {t:.1f} s (plan drop {self.plan_drop_s:.1f} s)" if t > limit else None
        if w.max > HARM_NEIGHBOUR_M:
            return f"harm: a neighbour's fruit swung {w.max * 1000:.0f} mm"
        if w.hard_n > gt.STRAIN_HARD_N:
            return f"harm: pressed {w.hard_pair} with {w.hard_n:.1f} N"
        if w.sat_run >= gt.STRAIN_STALL_FRAMES:
            return f"harm: a servo stalled at its limit {w.sat_run} frames"
        p_now, _ = ts.cut_target_now(r.info["trusses"][self.k], r.d, r.m, r.cut_u)
        dist = float(np.linalg.norm(jaw_pt - p_now))
        if dist < self.best - PROGRESS_M:
            self.best, self.t_best = dist, t
        elif t - self.t_best > PROGRESS_S:
            return f"stuck: the jaw got no closer to the cut point for {t - self.t_best:.1f} s ({dist * 1000:.0f} mm)"
        if self.plan_cut_s is not None and t > LATE_X * self.plan_cut_s + LATE_PAD_S:
            return f"late: not cut at {t:.1f} s (plan {self.plan_cut_s:.1f} s)"
        return None
