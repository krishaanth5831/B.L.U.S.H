#!/usr/bin/env python3
"""Why does the cut truss lose a fruit on the way into the crate? (v6, K7 2026-11-24). Plays the teacher's first plan on
one scene and traces the TARGET truss frame by frame: the force on every fruit's tether, what each of its fruit touches
(crate, arm, another fruit, the stalk), where the truss's lowest fruit is above the crate floor, and how far the truss
has turned in the jaw since the cut. At each fruit that comes off: the leg, its tether force against its threshold,
and its contacts in the frames before.

    python fruit_off_trace.py --seed 9730005 --base 0.37,0.20 --old-start --jaw v6
    python fruit_off_trace.py --seed 9730005 --base 0.37,0.20 --old-start --jaw v5     the same scene, the v5 jaw

Writes outputs/v6/trace_<seed>_<jaw>.jsonl (a line per frame). Teacher only, nothing recorded. lerobot-mj4 env,
outside the Claude sandbox.
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default="0.37,0.20")
    ap.add_argument("--old-start", action="store_true")
    ap.add_argument("--jaw", default="v6", choices=scene.JAWS)
    a = ap.parse_args()
    b = [float(v) for v in a.base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    if not a.old_start:
        import base_grid
        scene.START_NORM = tuple(float(v) for v in base_grid.find_start(a.jaw)[1])

    import mujoco

    import expert
    import generate_truss as gt
    import truss_expert as tx
    import truss_teacher as tt
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    cfg = SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS,
                            domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig="base",
                            start_jitter_deg=0.0, yolo_weights=None, jaw=a.jaw)
    robot = SO101Mujoco(cfg)
    robot.connect()
    robot.new_scene(a.seed, log=False)
    robot.get_observation()
    m, d, info, k = robot.m, robot.d, robot.info, robot._tgt_k
    t = info["trusses"][k]
    try:
        plan = tt.TrussTeacher(m, d, info, k).plan()
    except tx.PlanError as e:
        sys.exit(f"no plan: {e}")
    det = robot.det
    own = {f["tomato"]: f for f in t["fruit"]}
    mine = [j for j, f in enumerate(info["fruit"]) if f["tomato"] in own]
    fruit_geoms = {m.geom(info["fruit"][j]["geom"]).id: j for j in mine}
    stalk_geoms = {m.geom(g).id for g in t["geoms"]}
    names = {j: info["fruit"][j]["tomato"] for j in mine}
    body = m.body(t["body"]).id
    grip = m.body("arm/gripper").id
    crate_floor = scene.CRATE_WALL                 # the floor's top (crate_xml: slab at z 0..wall; as _release_fit)
    radii = {j: own[info["fruit"][j]["tomato"]]["radius"] for j in mine}
    thr = det.thresholds
    out = pathlib.Path(HERE.parent / "outputs" / "v6" / f"trace_{a.seed}_{a.jaw}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.open("w")
    cutter = robot.cutters[k]
    state = dict(R0=None, recent=collections.deque(maxlen=15), off=set(), frame=0)
    f6 = np.zeros(6)
    t0 = d.time

    def on_frame(_sent):
        state["frame"] += 1
        touch = collections.defaultdict(list)
        stalk_touch = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for g, o in ((g1, g2), (g2, g1)):
                if g in stalk_geoms and o not in stalk_geoms and o not in fruit_geoms:
                    mujoco.mj_contactForce(m, d, i, f6)
                    stalk_touch.append((m.geom(o).name or m.body(m.geom_bodyid[o]).name, round(float(f6[0]), 2)))
            for g, o in ((g1, g2), (g2, g1)):
                if g in fruit_geoms:
                    mujoco.mj_contactForce(m, d, i, f6)
                    other = m.geom(o).name or m.body(m.geom_bodyid[o]).name
                    touch[fruit_geoms[g]].append((other, round(float(f6[0]), 2)))
        Rg = d.xmat[grip].reshape(3, 3)
        Rt = d.xmat[body].reshape(3, 3)
        if cutter.cut_t is not None and state["R0"] is None:
            state["R0"] = Rg.T @ Rt
        turn, axis = None, None
        if state["R0"] is not None:
            dR = (Rg.T @ Rt) @ state["R0"].T                     # the turn since the cut, in the gripper frame
            turn = round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))), 1)
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            axis = None if np.linalg.norm(w) < 1e-9 else np.round(w / np.linalg.norm(w), 2).tolist()
        low = min(d.xpos[m.body(names[j]).id][2] - radii[j] for j in mine)
        phase = getattr(plan, "leg", None) or gt.phase_at(plan, d.time - t0)   # the teacher's live leg
        rec = dict(f=state["frame"], t=round(d.time - t0, 3), phase=phase, cut=cutter.cut_t is not None,
                   force=[round(float(det.force[j]), 2) for j in mine],
                   low_above_floor_mm=round((low - crate_floor) * 1000, 1),
                   turn_in_jaw_deg=turn, turn_axis_gripper=axis, touch={names[j]: v for j, v in touch.items()},
                   stalk_touch=stalk_touch,
                   jaw_deg=round(float(np.degrees(d.qpos[m.jnt_qposadr[m.joint("arm/gripper").id]])), 1))
        log.write(json.dumps(rec) + "\n")
        state["recent"].append(rec)
        for j in mine:
            if not np.isnan(det.detached_at[j]) and j not in state["off"]:
                state["off"].add(j)
                print(f"\n{names[j]} OFF in {phase} at {rec['t']} s: tether {det.force_at[j]:.2f} N "
                      f"(threshold {thr[j]:.2f}, peak {det.peak[j]:.2f}); truss turned {turn} deg in the jaw since the "
                      f"cut; lowest fruit {rec['low_above_floor_mm']} mm above the crate floor")
                for r in list(state["recent"])[-8:]:
                    print(f"   t {r['t']:6.3f} {r['phase']:12s} force {r['force'][mine.index(j)]:5.2f} N  "
                          f"touches {r['touch'].get(names[j], [])}")
        return None

    t_play, watch = gt.play(robot, plan, on_frame=on_frame)
    ok, why = gt.judge(robot, plan, t_play, watch)
    import truss_scene as ts
    (cx, cy), (sx, sy, h) = ts.CRATE_CENTRE, ts.CRATE_SIZE
    inner = np.array([sx / 2 - scene.CRATE_WALL, sy / 2 - scene.CRATE_WALL])
    print(f"crate inner half-size {np.round(inner * 1000, 0).tolist()} mm, rim at z {h * 1000:.0f} mm; release meta "
          f"{ {k: plan.meta.get(k) for k in ('release_point', 'crate_slack_mm', 'lower_stopped_early')} }")
    for j in mine:
        p = d.xpos[m.body(names[j]).id]
        rr = radii[j]
        out_x = abs(p[0] - cx) + rr - inner[0]
        out_y = abs(p[1] - cy) + rr - inner[1]
        flag = "  <- OUT" if max(out_x, out_y) > 0 else ""
        print(f"   {names[j]}: x {1000 * (p[0] - cx):6.0f} y {1000 * (p[1] - cy):6.0f} z {1000 * p[2]:5.0f} mm; "
              f"past the wall by x {1000 * out_x:5.0f} y {1000 * out_y:5.0f} mm{flag}")
    print(f"\nseed {a.seed} jaw {a.jaw}: {'ok' if ok else why}\ntarget {t['name']}, {t['n']} fruit, stalk "
          f"{t['r'] * 2000:.1f} mm; tether thresholds {np.round(thr[mine], 2).tolist()} N; peaks "
          f"{np.round(det.peak[mine], 2).tolist()} N; trace -> {out}")


if __name__ == "__main__":
    main()
