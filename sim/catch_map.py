#!/usr/bin/env python3
"""Catch map: how far off can the jaw be aimed and still cut AND hold the truss? (v6 step 1, K7 2026-11-24;
post-mortem §5.7 / §8 step 5: "drive the teacher to the cut with deliberate offsets in each direction and record what
still cuts AND holds").

    python catch_map.py --seeds-from heldout_v5b.jsonl --scenes 10 --out ../outputs/v6/catch_map_jaw_v5.jsonl

Teacher only, no policy, nothing recorded. Per scene: the teacher's FIRST plan is played with the cut point it aims at
shifted by an offset, one axis at a time, in a frame built from the stalk at the cut:
  side    across the jaw opening (horizontal, perpendicular to the approach and the stalk)
  depth   along the approach: + = the jaw aimed beyond the stalk (the stalk ends deeper in the jaw)
  height  along the stalk: + = up toward the knuckle
The approach is the horizontal direction from the arm base to the cut point (the teacher's approach is blended from it
by beta; close enough for a map). Only scenes whose first plan succeeds at offset 0 are used, so a failure is the jaw's
tolerance, not the planner's search. Nothing in the teacher, the cutter or the physics is changed: the shift is applied
to truss_scene.cut_target_now as the teacher reads it (the judge reads the true truss).

Per run (one JSON line): ok + the judge's reason, cut on the first close, held 0.5 s after the cut (the truss moved
with the jaw: its body drifted < HOLD_DRIFT_M in the gripper frame), both pad forces and the moving blade's squeeze at
the cut, fruit off. Run in the lerobot-mj4 env (MuJoCo 4.0.1), outside the Claude sandbox.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402
import scene  # noqa: E402
import truss  # noqa: E402
import truss_expert as tx  # noqa: E402
import truss_scene as ts  # noqa: E402
import truss_teacher as tt  # noqa: E402

HOLD_DRIFT_M = 0.015          # as check_truss_cut.HOLD_DRIFT_M: a held truss swings 5-14 mm, a dropped one 100+ mm
HOLD_AFTER_S = 0.5            # the post-mortem's gate: still held half a second after the cut
# Offsets (mm), one axis at a time. Side and depth to +-12 mm: the policy's first close was 9-13 mm off the cut point
# on new scenes (post-mortem 4.6). Height +-6: the cut zone is 10 mm long (truss.CUT_ZONE_M), the teacher cuts mid-zone.
OFFSETS_MM = {"side": (-12, -9, -6, -3, 3, 6, 9, 12), "depth": (-12, -9, -6, -3, 3, 6, 9, 12),
              "height": (-6, -3, 3, 6)}

_shift = np.zeros(3)          # the current offset in (side, depth, height), metres
_orig_cut_target_now = ts.cut_target_now


def _frame(p, s):
    """(side, depth, height) unit vectors at cut point p with stalk direction s (world)."""
    h = s / np.linalg.norm(s)
    a = p - np.asarray(scene.ARM_BASE, float)
    a[2] = 0.0
    a -= h * (a @ h)
    a /= np.linalg.norm(a)
    side = np.cross(h, a)
    return side / np.linalg.norm(side), a, h


def _shifted_cut_target_now(t, d, m, u=0.5):
    p, s = _orig_cut_target_now(t, d, m, u)
    if not _shift.any():
        return p, s
    side, depth, height = _frame(p, s)
    return p + _shift[0] * side + _shift[1] * depth + _shift[2] * height, s


tt.ts.cut_target_now = _shifted_cut_target_now      # the teacher's view only (tt.ts is truss_scene)

# Both pad forces at every step (TrussCutter keeps only the weaker one): wrap its step.
_orig_step = truss.TrussCutter.step


def _step_with_pads(self):
    f = {g: 0.0 for g in self.pads}
    for i in range(self.d.ncon):
        c = self.d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        other = g2 if g1 in self.stalk else g1 if g2 in self.stalk else None
        if other in f:
            truss.mujoco.mj_contactForce(self.m, self.d, i, self._f6)
            f[other] += self._f6[0]
    self.pads_now = (f[self.pads[0]], f[self.pads[1]])
    was = self.cut_t
    out = _orig_step(self)
    if was is None and self.cut_t is not None:
        self.at_cut = dict(pad_fixed_n=round(self.pads_now[0], 2), pad_moving_n=round(self.pads_now[1], 2),
                           squeeze_n=round(self.squeeze, 2))
    return out


truss.TrussCutter.step = _step_with_pads


def run(robot, seed, shift_m):
    """Plays the teacher's first plan for `seed` with the aim shifted; returns the run's record."""
    _shift[:] = 0.0
    robot.new_scene(seed, log=False)
    robot.get_observation()
    if robot._tgt_k is None:
        return dict(ok=False, why="no target")
    _shift[:] = shift_m
    try:
        plan = tt.TrussTeacher(robot.m, robot.d, robot.info, robot._tgt_k).plan()
    except tx.PlanError as e:
        return dict(ok=False, why=f"no plan: {str(e)[:120]}")
    k = robot._tgt_k
    cutter = robot.cutters[k]
    body = robot.m.body(robot.info["trusses"][k]["body"]).id
    grip = robot.m.body("arm/gripper").id
    held = dict(rel0=None, t=None, drift=None)
    closes = []                                  # frames where the jaw command closes (the first close's time)

    def rel():
        R = robot.d.xmat[grip].reshape(3, 3)
        return R.T @ (robot.d.xpos[body] - robot.d.xpos[grip])

    plan.reset()
    watch = gt.Watch(robot)
    t0 = robot.d.time
    while not plan.done:
        act = plan.act(robot._present_norm())
        robot.send_action(act)
        watch.step(act)
        closes.append(act["gripper.pos"])
        if cutter.cut_t is not None and held["rel0"] is None:
            held["rel0"], held["t"] = rel(), robot.d.time
        if held["rel0"] is not None and held["drift"] is None and robot.d.time - held["t"] >= HOLD_AFTER_S:
            held["drift"] = float(np.linalg.norm(rel() - held["rel0"]))
    ok, why = gt.judge(robot, plan, t0, watch)
    g = np.array(closes)
    first_close = None                           # the first frame the jaw command starts closing
    for i in range(1, len(g)):
        if g[i] < g[i - 1] - 1e-6:
            first_close = (i / expert.FPS)
            break
    rec = dict(ok=ok, why=why, cut=cutter.cut_t is not None,
               cut_s=None if cutter.cut_t is None else round(cutter.cut_t - t0, 2),
               first_close_s=None if first_close is None else round(first_close, 2),
               held=None if held["drift"] is None else bool(held["drift"] < HOLD_DRIFT_M),
               hold_drift_mm=None if held["drift"] is None else round(held["drift"] * 1000, 1),
               fruit_off=int(np.sum(~np.isnan(robot.det.detached_at))),
               peak_squeeze_n=round(cutter.peak["squeeze"], 2), cut_u=plan.meta.get("cut_u"))
    rec.update(getattr(cutter, "at_cut", {}) if cutter.cut_t is not None else {})
    if hasattr(cutter, "at_cut"):
        del cutter.at_cut
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-from", type=pathlib.Path, required=True, help="jsonl with a seed per line")
    ap.add_argument("--scenes", type=int, default=10, help="scenes whose first plan works at offset 0")
    ap.add_argument("--max-tries", type=int, default=40, help="seeds tried to find --scenes usable ones")
    ap.add_argument("--axes", default="side,depth,height")
    ap.add_argument("--rig", default="base")
    ap.add_argument("--jaw", default="v5", choices=scene.JAWS, help="v5 (every dataset so far) or v6 (K7 2026-11-24)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    cfg = SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS,
                            domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig=a.rig,
                            start_jitter_deg=0.0, yolo_weights=None, jaw=a.jaw)
    robot = SO101Mujoco(cfg)
    robot.connect()
    seeds = [int(json.loads(x)["seed"]) for x in a.seeds_from.read_text().splitlines() if x.strip()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.open("a")
    used, t_start = 0, time.time()
    for seed in seeds[:a.max_tries]:
        if used >= a.scenes:
            break
        base = run(robot, seed, np.zeros(3))
        out.write(json.dumps(dict(seed=seed, axis="none", offset_mm=0, **base)) + "\n"); out.flush()
        print(f"seed {seed} offset 0: {'ok' if base['ok'] else base['why'][:80]}", flush=True)
        if not base["ok"]:
            continue
        used += 1
        for axis in a.axes.split(","):
            i = ("side", "depth", "height").index(axis)
            for mm in OFFSETS_MM[axis]:
                sh = np.zeros(3)
                sh[i] = mm / 1000.0
                r = run(robot, seed, sh)
                out.write(json.dumps(dict(seed=seed, axis=axis, offset_mm=mm, **r)) + "\n"); out.flush()
                print(f"  {axis} {mm:+d} mm: {'ok' if r['ok'] else r['why'][:70]}  cut {r.get('cut')} "
                      f"held {r.get('held')} pads {r.get('pad_fixed_n')}/{r.get('pad_moving_n')} N", flush=True)
    print(f"{used} scenes in {(time.time() - t_start) / 60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
