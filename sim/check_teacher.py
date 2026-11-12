#!/usr/bin/env python3
"""Step 0 of the post-mortem ladder (§6.6): the three tests teacher v3 (truss_teacher.py) must pass before any data.

    python check_teacher.py repeat --seed 9100000                # 1. same seed twice -> identical actions? PASS/FAIL
    python check_teacher.py push --seed 9100000                  # 2. pushed 1-2 cm mid-approach -> still a cut?
    python check_teacher.py push --seed 9100000 --watch          #    the same in a MuJoCo window at real time
    python check_teacher.py solve --scenes 200                   # 3. solve rate on 200 scenes + why the rest fail
    python check_teacher.py solve --scenes 1 --seed-start 9100003 --watch   # watch one scene

Scenes as generate_truss.py builds them (rig base, settle 1 s), but domain randomisation OFF: visual randomisation
stays off until the sim reaches 50 % (post-mortem §6.2), and the teacher is judged on the plain scene. The target
is the true ready truss (no truss YOLO). A run is judged with generate_truss.judge (cut, whole in the crate, no fruit
off, no neighbour moved > 15 mm, back at the start pose).

The push (test 2): at a fixed frame of a leg the arm is displaced, not pushed with a force: the 5 arm joints are
offset so the gripper moves PUSH_MM in the given direction (damped least squares on the gripper's Jacobian), joint
velocities zeroed. The teacher must steer back from where the arm now is. The displacement actually reached is
printed; legs: to_standoff (free move), approach (standoff -> pre-cut), slide_in (onto the stalk).

Writes only the solve log ~/.cache/blush_sim/teacher/solve_<stamp>.jsonl (and nothing for repeat / push).
Run in the lerobot env under the memory cap:
  systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python check_teacher.py solve --scenes 200
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402
import truss_teacher as tt  # noqa: E402

LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "teacher"
SEED_START = 9_100_000          # apart from the v2 data (7600000s), the pilots (200000s) and the evals
PUSH_MM = (10.0, 20.0)          # §6.1: "a 1-2 cm push mid-approach still ends in a cut"
PUSH_DIRS = {"down": (0, 0, -1.0), "side": (1.0, 0, 0), "back": (0, -1.0, 0)}   # world: -z, along the row, to the aisle
PUSH_FRAME = {"to_standoff": 12, "approach": 6, "slide_in": 10}   # frames into the leg (design choice)


def make_robot(watch=False):
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    cfg = SO101MujocoConfig(id="sim", seed=SEED_START, viewer=watch, overlay="true_box", realtime=False,
                            fps=expert.FPS, domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig="base",
                            start_jitter_deg=0.0, yolo_weights=None)
    robot = SO101Mujoco(cfg)
    robot.connect()
    return robot


def displace(robot, direction, metres):
    """Offset the 5 arm joints so the gripper moves `metres` along `direction` (world). Returns the move reached."""
    m, d = robot.m, robot.d
    b = m.body("arm/gripper").id
    jids = [m.joint("arm/" + j).id for j in expert.arm_map.JOINTS[:5]]
    qadr = [m.jnt_qposadr[j] for j in jids]
    dadr = [m.jnt_dofadr[j] for j in jids]
    p0 = d.xpos[b].copy()
    goal = p0 + np.asarray(direction, float) / np.linalg.norm(direction) * metres
    for _ in range(4):
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, b)
        J = jacp[:, dadr]
        e = goal - d.xpos[b]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        d.qpos[qadr] += dq
        mujoco.mj_forward(m, d)
    d.qvel[dadr] = 0.0
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(d.xpos[b] - p0))


def run(robot, seed, push=None, realtime=False):
    """One scene with the teacher, judged. push: (leg, direction name, metres) or None."""
    robot.new_scene(seed, log=False)
    robot.get_observation()                      # the start frame: picks the target (true box)
    out = dict(seed=seed)
    k = robot._tgt_k
    if k is None:
        return dict(out, solved=False, stage="target", reason=f"no target ({robot._box_info['reason']})")
    t_plan = time.time()
    try:
        T = tt.TrussTeacher(robot.m, robot.d, robot.info, k).plan()
    except tt.PlanError as e:
        return dict(out, solved=False, stage="plan", reason=str(e), plan_s=round(time.time() - t_plan, 1))
    out.update(plan_s=round(time.time() - t_plan, 1), meta={k_: v for k_, v in T.meta.items() if k_ != "tried"})
    watch = gt.Watch(robot)
    t0 = robot.d.time
    acts, pushed = [], None
    while not T.done:
        t_f = time.time()
        if push is not None and pushed is None and T.leg == push[0] and T.n_in_leg == PUSH_FRAME[push[0]]:
            pushed = displace(robot, PUSH_DIRS[push[1]], push[2])
        a = T.act(robot._present_norm())
        robot.send_action(a)
        acts.append([a[f"{j}.pos"] for j in expert.arm_map.JOINTS])
        watch.step()
        if realtime:
            time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
    ok, why = gt.judge(robot, T, t0, watch)
    out.update(solved=ok, stage="dry run", reason=why, seconds=T.meta.get("seconds"),
               timed_out=T.meta.get("timed_out"), phases=T.phases, neighbour_mm=round(watch.max * 1000, 1))
    if push is not None:
        out.update(push=dict(leg=push[0], dir=push[1], asked_mm=round(push[2] * 1000, 1),
                             reached_mm=None if pushed is None else round(pushed * 1000, 1)))
    out["actions"] = np.array(acts)
    out["qpos_end"] = robot.d.qpos.copy()
    return out


def cmd_repeat(a):
    robot = make_robot(a.watch)
    try:
        r1 = run(robot, a.seed, realtime=a.watch)
        r2 = run(robot, a.seed, realtime=a.watch)
    finally:
        robot.disconnect()
    if "actions" not in r1 or "actions" not in r2:
        print(f"FAIL: no run to compare (seed {a.seed}: {r1.get('reason')})")
        return 1
    same_len = len(r1["actions"]) == len(r2["actions"])
    da = float(np.abs(r1["actions"] - r2["actions"]).max()) if same_len else float("inf")
    dq = float(np.abs(r1["qpos_end"] - r2["qpos_end"]).max())
    ok = same_len and da == 0.0 and dq == 0.0
    print(f"seed {a.seed}: run 1 {len(r1['actions'])} frames ({r1['reason']}), run 2 {len(r2['actions'])} frames "
          f"({r2['reason']}); max action difference {da:g} norm, max final qpos difference {dq:g}")
    print(f"{'PASS' if ok else 'FAIL'}: the same seed gives {'identical' if ok else 'DIFFERENT'} actions")
    return 0 if ok else 1


def cmd_push(a):
    robot = make_robot(a.watch)
    rows = []
    try:
        base = run(robot, a.seed)
        if not base["solved"]:
            print(f"FAIL: seed {a.seed} isn't solved without a push ({base['stage']}: {base['reason']}); pick another")
            return 1
        legs = [p for p, _ in base["phases"]]
        for leg in a.legs:
            if leg not in legs:
                print(f"  {leg}: not in this scene's motion (standoff 0 cm?), skipped")
                continue
            for dname in a.dirs:
                for mm in a.mm:
                    r = run(robot, a.seed, push=(leg, dname, mm / 1000), realtime=a.watch)
                    rows.append(r)
                    p = r.get("push", {})
                    print(f"  push {leg:12s} {dname:5s} asked {mm:4.0f} mm, reached {p.get('reached_mm')} mm -> "
                          f"{'cut + crate' if r['solved'] else 'FAIL: ' + r['reason']}", flush=True)
    finally:
        robot.disconnect()
    n_ok = sum(r["solved"] for r in rows)
    ok = bool(rows) and n_ok == len(rows)
    print(f"{'PASS' if ok else 'FAIL'}: {n_ok} of {len(rows)} pushed runs still cut the truss and put it in the crate")
    return 0 if ok else 1


def cmd_solve(a):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"solve_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    robot = make_robot(a.watch)
    why = collections.Counter()
    meta_c = collections.Counter()
    n_ok, n = 0, 0
    t_start = time.time()
    print(f"log {path}", flush=True)
    try:
        with path.open("w") as log:
            for seed in range(a.seed_start, a.seed_start + a.scenes):
                t_s = time.time()
                r = run(robot, seed, realtime=a.watch)
                r.pop("actions", None)
                r.pop("qpos_end", None)
                r["wall_s"] = round(time.time() - t_s, 1)
                n += 1
                n_ok += r["solved"]
                if r["solved"]:
                    md = r["meta"]
                    meta_c[f"cut_u {md['cut_u']}"] += 1
                    meta_c[f"blend {md['approach_blend_deg']:g}"] += 1
                    meta_c[f"standoff {md['standoff_mm']} mm"] += 1
                    meta_c.update(f"detour {k}" for k, v in md["detours"].items() if v)
                else:
                    why[f"{r['stage']}: {r['reason'][:90]}"] += 1
                log.write(json.dumps(r, default=float) + "\n")
                log.flush()
                print(f"  seed {seed}: {'solved' if r['solved'] else 'FAIL ' + r['stage'] + ': ' + r['reason'][:100]}"
                      f"  [{r['wall_s']} s]  {n_ok}/{n}", flush=True)
    finally:
        robot.disconnect()
    lo, hi = wilson(n_ok, n)
    mins = (time.time() - t_start) / 60
    summary = dict(summary=True, solved=n_ok, scenes=n, rate=round(n_ok / max(n, 1), 3), wilson95=[lo, hi],
                   minutes=round(mins, 1), fails=dict(why.most_common()), solved_choices=dict(meta_c.most_common()))
    with path.open("a") as log:
        log.write(json.dumps(summary) + "\n")
    print(f"\nsolved {n_ok} of {n} scenes ({n_ok / max(n, 1):.0%}, 95 % interval {lo:.0%}-{hi:.0%}) in {mins:.0f} min")
    print("why the rest failed:")
    for k, v in why.most_common():
        print(f"  {v:4d}  {k}")
    print("choices in the solved scenes:")
    for k, v in sorted(meta_c.items()):
        print(f"  {v:4d}  {k}")
    print(f"log {path}")
    return 0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(float(c - h), 3), round(float(c + h), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repeat", help="test 1: same seed twice, identical actions")
    p.add_argument("--seed", type=int, default=SEED_START)
    p = sub.add_parser("push", help="test 2: pushed mid-approach, still a cut")
    p.add_argument("--seed", type=int, default=SEED_START)
    p.add_argument("--legs", nargs="+", default=list(PUSH_FRAME))
    p.add_argument("--dirs", nargs="+", default=list(PUSH_DIRS), choices=list(PUSH_DIRS))
    p.add_argument("--mm", nargs="+", type=float, default=list(PUSH_MM))
    p = sub.add_parser("solve", help="test 3: solve rate on random scenes")
    p.add_argument("--scenes", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    for p in sub.choices.values():
        p.add_argument("--watch", action="store_true", help="MuJoCo window at real time")
    a = ap.parse_args()
    sys.exit({"repeat": cmd_repeat, "push": cmd_push, "solve": cmd_solve}[a.cmd](a))


if __name__ == "__main__":
    main()
