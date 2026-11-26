#!/usr/bin/env python3
"""Where the cut truss is when the jaw opens over the crate, and where it ends up (measurement only).

    python whatif/release_trace.py --seeds 9730054,9730080 --retries 2 --out ../outputs/v6/release_trace.jsonl

Plays the teacher's plans as base_grid does (base 0.34/0.14, --retries more plans after a failed one) and reads, per
plan:
  at the release   the grip point against the crate's centre; every fruit of the cut truss against the crate's inner
                   walls (margin_mm: the least room, negative = already outside) and floor (low_mm) and rim (high_mm);
                   how far the fruit are from where the planner's fit has them (fit_off_mm, the truss as cut)
  after it         frames until the stalk no longer touches the jaw (let_go_frames; None = it never let go)
  at the end       which fruit are outside the crate and by how much, what the truss rests on (crate walls, arm), fruit
                   that came off
Run in the lerobot-mj4 env, outside the Claude sandbox. Changes nothing in the sim.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import base_grid as bg  # noqa: E402
import scene  # noqa: E402

AFTER = ("release", "drop", "lift", "return", "hold_end")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--base", default="0.34,0.14")
    ap.add_argument("--jaw", default="v6", choices=scene.JAWS)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    b = [float(v) for v in a.base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    scene.START_NORM = tuple(float(v) for v in bg.find_start(a.jaw)[1])

    import expert
    import generate_truss as gt
    import truss_expert as tx
    import truss_ik
    import truss_scene as ts
    import truss_teacher as tt
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, rig="base", start_jitter_deg=0.0, yolo_weights=None, jaw=a.jaw))
    robot.connect()
    (cx, cy), (sx, sy, h) = ts.CRATE_CENTRE, ts.CRATE_SIZE
    wall = scene.CRATE_WALL
    out = a.out.open("a") if a.out else None
    for seed in [int(s) for s in a.seeds.split(",")]:
        robot.new_scene(seed, log=False)
        robot.get_observation()
        teacher = tt.TrussTeacher(robot.m, robot.d, robot.info, robot._tgt_k)
        try:
            plan = teacher.plan()
        except tx.PlanError as e:
            print(f"seed {seed}: no plan ({str(e)[:80]})", flush=True)
            continue
        k, failed = 0, set()
        while True:
            if k:
                robot.new_scene(seed, log=False)
                robot.get_observation()
            m, d, t = robot.m, robot.d, robot.info["trusses"][robot._tgt_k]
            name = t["name"]
            fr = [(f["tomato"], m.body(f["tomato"]).id, f["radius"]) for f in t["fruit"]]
            stalk = {m.geom(f"{name}_peduncle").id, m.geom(f"{name}_rachis").id}
            bgr = m.body("arm/gripper").id
            point = truss_ik.cut_point_in_gripper(t["r"])
            # the fit's truss: the fruit as they hang on the plant now, seen from the planned cut pose
            model = None
            if teacher._q_cut is not None:
                o_c, R_c = teacher.ik.fk(teacher._q_cut)
                model = np.array([R_c.T @ (d.xpos[i] - o_c) for _, i, _ in fr])
            st = dict(prev=None, n=0, at=None, let_go=None, since=0)

            def box(st=st, fr=fr, d=d):
                P = np.array([d.xpos[i] for _, i, _ in fr])
                R = np.array([r for _, _, r in fr])
                mx = (sx / 2 - wall) - np.abs(P[:, 0] - cx) - R
                my = (sy / 2 - wall) - np.abs(P[:, 1] - cy) - R
                return P, R, mx, my

            def on_frame(_sent, st=st, plan=plan, m=m, d=d, fr=fr, stalk=stalk, bgr=bgr, point=point, model=model,
                         box=box):
                leg = plan.leg
                st["n"] = st["n"] + 1 if leg == st["prev"] else 0
                first = leg == "release" and st["prev"] != "release"
                st["prev"] = leg
                if first:
                    Rg, og = d.xmat[bgr].reshape(3, 3), d.xpos[bgr]
                    P, R, mx, my = box()
                    grip = og + Rg @ point
                    fit = None
                    if model is not None:
                        fit = float(np.linalg.norm(np.array([Rg.T @ (p - og) for p in P]) - model, axis=1).max())
                    st["at"] = dict(grip_from_centre_mm=[round(float(v) * 1e3) for v in (grip[0] - cx, grip[1] - cy)],
                                    grip_over_rim_mm=round(float(grip[2] - h) * 1e3),
                                    margin_mm=round(float(min(mx.min(), my.min())) * 1e3),
                                    margin_x_mm=round(float(mx.min()) * 1e3), margin_y_mm=round(float(my.min()) * 1e3),
                                    low_mm=round(float((P[:, 2] - R).min() - wall) * 1e3),
                                    high_over_rim_mm=round(float((P[:, 2] + R).max() - h) * 1e3),
                                    fit_off_mm=None if fit is None else round(fit * 1e3))
                if leg in AFTER and st["at"] is not None and st["let_go"] is None:
                    st["since"] += 1
                    held = False
                    for i in range(d.ncon):
                        c = d.contact[i]
                        if ({c.geom1, c.geom2} & stalk) and any(
                                (m.geom(g).name or "").startswith("arm/") for g in (c.geom1, c.geom2)):
                            held = True
                            break
                    if not held:
                        st["let_go"] = st["since"]
                return None

            t0, watch = gt.play(robot, plan, on_frame=on_frame)
            ok, why = gt.judge(robot, plan, t0, watch)
            P, R, mx, my = box()
            outside = [dict(fruit=n, x_mm=round(float(mx[j]) * 1e3), y_mm=round(float(my[j]) * 1e3),
                            z_over_rim_mm=round(float(P[j, 2] - h) * 1e3))
                       for j, (n, _, _) in enumerate(fr)
                       if not (abs(P[j, 0] - cx) < sx / 2 and abs(P[j, 1] - cy) < sy / 2
                               and P[j, 2] < h + ts.IN_CRATE_TOP_M)]
            mine = stalk | {m.geom(f["geom"]).id if isinstance(f["geom"], str) else f["geom"] for f in t["fruit"]}
            rests = set()
            for i in range(d.ncon):
                c = d.contact[i]
                if {c.geom1, c.geom2} & mine:
                    o = c.geom2 if c.geom1 in mine else c.geom1
                    if o not in mine:
                        rests.add(m.geom(o).name or f"geom{o}")
            mt = plan.meta
            rec = dict(seed=seed, attempt=k, ok=bool(ok), why=str(why)[:100], release_point=mt.get("release_point"),
                       crate_slack_mm=mt.get("crate_slack_mm"), fallback=str(mt.get("release_fallback")),
                       legs=[g.name for g in plan.legs if g.name in ("lower", "settle_above", "lift")],
                       at_release=st["at"], let_go_frames=st["let_go"], outside=outside,
                       rests_on=sorted(x for x in rests if x.startswith(("crate", "arm/")) or x in ("floor", "table")))
            print(f"seed {seed} #{k}: ok {ok} {str(why)[:44]:44s} | at release {st['at']} | let go after {st['let_go']}"
                  f" frames | outside {outside} | rests on {rec['rests_on']} | fit slack {mt.get('crate_slack_mm')} mm"
                  f" legs {rec['legs']}", flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            if ok or k >= a.retries:
                break
            k += 1
            try:
                gt.next_plan_after(plan, why, failed)
            except tx.PlanError:
                break


if __name__ == "__main__":
    main()
