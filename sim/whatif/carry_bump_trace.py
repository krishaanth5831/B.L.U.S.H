#!/usr/bin/env python3
"""Who touches a neighbour during the carry, and was the cut truss where the planner's model put it (measurement only).

    BLUSH_V6='{"V6_ALIGN": true, "V6_CARRY": ["rigid", 0]}' python whatif/carry_bump_trace.py --seeds 9730005,9730031

Plays the teacher's first plan (base 0.34/0.14, v6 jaw, as base_grid) and on every frame of the retreat and the carry
reads:
  - contacts between a NEIGHBOUR truss (stalk or fruit) and the arm or the cut truss: who, with what force;
  - how far the cut truss's fruit are from where they were in the gripper frame when the retreat began, the stalk's
    angle to the pads' line, and how the truss has turned in the gripper frame against the planner's model (the
    truss as it hung on the plant, seen from the planned cut pose): degrees about the pads' normal (gripper x), the
    stalk's own axis (y) and the fingers (z).
Prints the first touch per pair of bodies, the worst touch, and the truss's drift at those frames. --noslip N is
DIAGNOSTIC only (pad_hold_torque.py explains it). Changes nothing in the sim.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import base_grid as bg  # noqa: E402
import scene  # noqa: E402
from pad_hold_torque import sample  # noqa: E402

LEGS = ("retreat", "to_crate", "settle_above", "lower")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--base", default="0.34,0.14")
    ap.add_argument("--noslip", type=int, default=0, help="DIAGNOSTIC: MuJoCo noslip iterations for this run")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--legs", default=",".join(LEGS),
                    help="legs to read (default: the carry). The cut: slide_in,settle_at_stalk,close_cut,squeeze")
    ap.add_argument("--jaw", default="v6", choices=scene.JAWS, help="v5 = the old jaw on the same scenes (control)")
    a = ap.parse_args()
    global LEGS
    LEGS = tuple(a.legs.split(","))
    b = [float(v) for v in a.base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    scene.START_NORM = tuple(float(v) for v in bg.find_start(a.jaw)[1])

    import expert
    import generate_truss as gt
    import truss_expert as tx
    import truss_ik
    import truss_teacher as tt
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, rig="base", start_jitter_deg=0.0, yolo_weights=None, jaw=a.jaw))
    robot.connect()
    out = a.out.open("a") if a.out else None
    for seed in [int(s) for s in a.seeds.split(",")]:
        robot.new_scene(seed, log=False)
        robot.m.opt.noslip_iterations = a.noslip
        robot.get_observation()
        m, d = robot.m, robot.d
        teacher = tt.TrussTeacher(m, d, robot.info, robot._tgt_k)
        try:
            plan = teacher.plan()
        except tx.PlanError as e:
            print(f"seed {seed}: no plan ({str(e)[:80]})", flush=True)
            continue
        # the planner's carried-truss model: the truss as it hangs on the plant now, seen from the planned cut pose
        bt = m.body(robot.info["trusses"][robot._tgt_k]["body"]).id
        R_model = teacher.ik.fk(teacher._q_cut)[1].T @ d.xmat[bt].reshape(3, 3) if teacher._q_cut is not None else None
        # the two carried-truss models' fruit positions in the gripper frame: "rigid" = as cut, "hang" = rolled about
        # the stalk's own axis until its centre of mass hangs under it (tx.CARRY note)
        models = {}
        if teacher._q_cut is not None:
            o_c, R_c = teacher.ik.fk(teacher._q_cut)
            carry = teacher._carry(teacher._q_cut)
            point = truss_ik.cut_point_in_gripper(teacher.t["r"])
            keep = tx.CARRY, tx.CARRY_AXIS
            for label, (ck, ca) in dict(rigid=("rigid", 0), hang=("hang", 1)).items():
                tx.CARRY, tx.CARRY_AXIS = ck, ca
                W = tx.hanging(o_c, R_c, carry[1], carry[2], point)
                models[label] = np.array([R_c.T @ (pw - o_c) for pw, _ in W[1:]])
            tx.CARRY, tx.CARRY_AXIS = keep
            fruit_ids = list(carry[0][1:])
        name = robot.info["trusses"][robot._tgt_k]["name"]
        own = {f["tomato"] for f in robot.info["trusses"][robot._tgt_k]["fruit"]}
        own_ids = [m.body(t).id for t in sorted(own)]
        bgr = m.body("arm/gripper").id

        def kind(body, name=name, own=own, m=m):
            n = m.body(body).name
            if n.startswith("arm/"):
                return "arm", n[4:]
            if n.startswith(name + "_") or n in own:
                return "cut truss", n
            if re.match(r"s\d+t\d+_", n):
                return "neighbour", n
            return "other", n

        st = dict(prev=None, n=0, ref=None, rows=[], f6=np.zeros(6))

        def on_frame(_sent, st=st, plan=plan, name=name, own=own, own_ids=own_ids, kind=kind, R_model=R_model, bt=bt,
                     models=models, fruit_ids=fruit_ids if models else None, m=m, d=d, bgr=bgr):
            leg = plan.leg
            st["n"] = st["n"] + 1 if leg == st["prev"] else 0
            st["prev"] = leg
            if leg not in LEGS:
                return None
            Rg, og = d.xmat[bgr].reshape(3, 3), d.xpos[bgr]
            P = np.array([Rg.T @ (d.xipos[i] - og) for i in own_ids])       # own fruit in the gripper frame
            if st["ref"] is None:
                st["ref"] = P
            touches = []
            for i in range(d.ncon):
                c = d.contact[i]
                k1, k2 = kind(m.geom_bodyid[c.geom1]), kind(m.geom_bodyid[c.geom2])
                if "neighbour" not in (k1[0], k2[0]) or {k1[0], k2[0]} == {"neighbour"}:
                    continue
                nb, ot = (k1, k2) if k1[0] == "neighbour" else (k2, k1)
                if ot[0] not in ("arm", "cut truss"):
                    continue
                mujoco.mj_contactForce(m, d, i, st["f6"])
                touches.append((ot[0], ot[1], nb[1], round(float(st["f6"][0]), 2)))
            s = sample(robot, name, own)
            turn = None
            if R_model is not None:        # how the truss has turned in the gripper frame against the model (deg)
                dR = (Rg.T @ d.xmat[bt].reshape(3, 3)) @ R_model.T
                ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                nrm = np.linalg.norm(ax)
                turn = [round(float(v), 1) for v in np.degrees(ang) * (ax / nrm if nrm > 1e-9 else ax)]
            off = {}
            if models:
                Pm = np.array([Rg.T @ (d.xpos[i] - og) for i in fruit_ids])
                off = {k: round(float(np.linalg.norm(Pm - v, axis=1).max()) * 1e3, 1) for k, v in models.items()}
            st["rows"].append(dict(leg=leg, frame=st["n"], touches=touches, turn_xyz_deg=turn, off_model_mm=off,
                                   drift_mm=round(float(np.linalg.norm(P - st["ref"], axis=1).max()) * 1e3, 1),
                                   stalk_deg=s["stalk_to_pads_line_deg"], contacts=s["n_contacts"]))
            return None

        t0, watch = gt.play(robot, plan, on_frame=on_frame)
        ok, why = gt.judge(robot, plan, t0, watch)
        mt = plan.meta
        n_leg = collections.Counter(r["leg"] for r in st["rows"])
        print(f"seed {seed}: ok {ok} {str(why)[:70]} | detour {mt.get('detours', {}).get('to_crate')} retreat"
              f" {mt.get('retreat_option')} align {mt.get('carry_align_deg')} noslip {a.noslip}", flush=True)
        first, worst = {}, {}
        for r in st["rows"]:
            for (k, ob, nb, f) in r["touches"]:
                key = (k, ob if k == "arm" else "", nb)
                first.setdefault(key, r)
                if f > worst.get(key, (0, None))[0]:
                    worst[key] = (f, r)
        for key, r in sorted(first.items(), key=lambda kv: (LEGS.index(kv[1]["leg"]), kv[1]["frame"])):
            f, rw = worst.get(key, (0.0, r))
            print(f"   {key[0]:9s} {key[1]:22s} touches {key[2]:14s} first {r['leg']} {r['frame']}/{n_leg[r['leg']]}"
                  f" (truss drift {r['drift_mm']} mm, stalk {r['stalk_deg']} deg) | worst {f} N at {rw['leg']}"
                  f" {rw['frame']} (drift {rw['drift_mm']} mm, stalk {rw['stalk_deg']} deg)", flush=True)
        for leg in LEGS:
            rr = [r for r in st["rows"] if r["leg"] == leg]
            if rr:
                q = [rr[int(u * (len(rr) - 1))] for u in (0, 0.25, 0.5, 0.75, 1)]
                print(f"   {leg:13s} {len(rr):3d} frames | drift mm " + " ".join(f"{r['drift_mm']:5.1f}" for r in q)
                      + " | stalk deg " + " ".join(f"{r['stalk_deg']:5.1f}" for r in q)
                      + " | pad contacts " + " ".join(str(r["contacts"]) for r in q), flush=True)
                for k in models:
                    print(f"                 worst fruit off the '{k}' model, mm: "
                          + " ".join(f"{r['off_model_mm'][k]:5.0f}" for r in q), flush=True)
                print("                 turn vs the planner's model, deg about pads' normal / stalk / fingers: "
                      + "  ".join("/".join(f"{v:.0f}" for v in r["turn_xyz_deg"]) for r in q if r["turn_xyz_deg"]),
                      flush=True)
        if out:
            out.write(json.dumps(dict(seed=seed, noslip=a.noslip, ok=ok, why=str(why)[:100],
                                      detour=str(mt.get("detours", {}).get("to_crate")), rows=st["rows"])) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
