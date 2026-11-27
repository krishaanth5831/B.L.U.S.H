#!/usr/bin/env python3
"""How much turning load the pads carry on the cut truss, and how much they could at most (measurement only).

    python whatif/pad_hold_torque.py --seeds 9730003,9730005,9730015,9730018 --out ../outputs/v6/pad_hold.jsonl

Plays the teacher's first plan (base 0.34/0.14, v6 jaw, as base_grid) and, at the last squeeze frame and every
--every frames of the carry (retreat, to_crate, settle_above, lower), reads the contacts between the target's stalk and
the arm:
  - which arm geom, where in the gripper frame (mm), the normal force (N), the contact's friction coefficients;
  - about the PADS' NORMAL (gripper x, the jaw's closing axis) through the contacts' force centre:
      gravity_nm   the truss's weight (stalk + its fruit still on) x its lever = what tries to turn it in the pads'
                   plane
      carried_nm   what the contacts carry right now (tangential forces x their levers + torsional torques)
      capacity_nm  the most they could carry: sum of mu x normal x lever + torsional coefficient x normal
  - the stalk's angle to the pads' line (gripper y), to see when it turns.
capacity < gravity = the grip cannot hold the truss against turning, whatever the planner does. The same number for a
real pad is a bench test: hang a weight on a 4 mm rod clamped in the pads and see at what lever arm it starts to turn.
Run in the lerobot-mj4 env, outside the Claude sandbox. Changes nothing in the sim.
"""
import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import base_grid as bg  # noqa: E402
import scene  # noqa: E402

CARRY_LEGS = ("retreat", "to_crate", "settle_above", "lower")


def sample(robot, name, own):
    m, d = robot.m, robot.d
    bt, bgr = m.body(f"{name}_truss").id, m.body("arm/gripper").id
    Rg, og = d.xmat[bgr].reshape(3, 3), d.xpos[bgr]
    stalk = {m.geom(f"{name}_peduncle").id, m.geom(f"{name}_rachis").id}
    rows, f6 = [], np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if not ({c.geom1, c.geom2} & stalk):
            continue
        other = c.geom2 if c.geom1 in stalk else c.geom1
        gname = m.geom(other).name
        if not gname.startswith("arm/"):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        F = c.frame.reshape(3, 3)                       # rows: normal, tangent 1, tangent 2 (world)
        rows.append(dict(geom=gname[4:], p=Rg.T @ (c.pos - og), n=float(f6[0]),
                         ft=Rg.T @ (F[1] * f6[1] + F[2] * f6[2]),        # tangential force, gripper frame
                         tors=float(f6[3]) * float((Rg.T @ F[0])[0]),     # torsional torque about gripper x
                         mu=float(c.friction[0]), mu_t=float(c.friction[2]), dim=int(c.dim)))
    out = dict(n_contacts=len(rows))
    # the truss's weight and centre of mass: stalk + the fruit still attached to it
    bodies = [bt] + [m.body(f["tomato"]).id for j, f in enumerate(robot.info["fruit"])
                     if f["tomato"] in own and np.isnan(robot.det.detached_at[j])]
    mass = float(sum(m.body_mass[b] for b in bodies))
    com = sum(m.body_mass[b] * d.xipos[b] for b in bodies) / mass
    g0 = m.geom(f"{name}_peduncle").id
    a = d.geom_xmat[g0].reshape(3, 3)[:, 2]                            # the peduncle capsule's axis (world)
    ay = Rg[:, 1]
    out["stalk_to_pads_line_deg"] = round(float(np.degrees(np.arccos(np.clip(abs(a @ ay), 0, 1)))), 1)
    out["mass_g"] = round(mass * 1e3, 1)
    if not rows or sum(r["n"] for r in rows) <= 1e-6:
        return out
    N = np.array([r["n"] for r in rows])
    P = np.array([r["p"] for r in rows])
    c0 = (N[:, None] * P).sum(0) / N.sum()                             # force centre, gripper frame
    lever = np.linalg.norm((P - c0)[:, 1:], axis=1)                    # in the pads' plane (y, z)
    w = Rg.T @ (mass * m.opt.gravity)                                  # weight, gripper frame
    r_com = Rg.T @ (com - og) - c0
    grav = np.cross(r_com, w)
    carried = sum(np.cross(r["p"] - c0, r["ft"])[0] + r["tors"] for r in rows)
    cap = float(sum(r["mu"] * r["n"] * lv + r["mu_t"] * r["n"] for r, lv in zip(rows, lever)))
    out.update(gravity_nm=[round(float(v), 4) for v in grav], carried_x_nm=round(float(carried), 4),
               capacity_x_nm=round(cap, 4), com_from_grip_mm=[round(float(v) * 1e3, 1) for v in r_com],
               contacts=[dict(geom=r["geom"], p_mm=[round(float(v) * 1e3, 1) for v in r["p"]], n=round(r["n"], 2),
                              lever_mm=round(float(lv) * 1e3, 1), mu=r["mu"], mu_t=r["mu_t"], dim=r["dim"])
                         for r, lv in zip(rows, lever)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--base", default="0.34,0.14")
    ap.add_argument("--every", type=int, default=15, help="carry frames between samples (30 fps)")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--noslip", type=int, default=0,
                    help="DIAGNOSTIC: MuJoCo noslip solver iterations for this run (0 = the sim as it is). Friction in "
                         "MuJoCo is a soft constraint: a loaded contact creeps even far below its friction limit; the "
                         "noslip pass removes that creep. Tells creep (solver) from a real friction limit")
    ap.add_argument("--impratio", type=float, default=None, help="DIAGNOSTIC: opt.impratio for this run")
    a = ap.parse_args()
    b = [float(v) for v in a.base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    st = bg.find_start("v6")
    scene.START_NORM = tuple(float(v) for v in st[1])

    import expert
    import generate_truss as gt
    import truss_expert as tx
    import truss_teacher as tt
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, rig="base", start_jitter_deg=0.0, yolo_weights=None, jaw="v6"))
    robot.connect()
    out = a.out.open("a") if a.out else None
    for seed in [int(s) for s in a.seeds.split(",")]:
        robot.new_scene(seed, log=False)
        robot.m.opt.noslip_iterations = a.noslip
        if a.impratio is not None:
            robot.m.opt.impratio = a.impratio
        robot.get_observation()
        try:
            plan = tt.TrussTeacher(robot.m, robot.d, robot.info, robot._tgt_k).plan()
        except tx.PlanError as e:
            print(f"seed {seed}: no plan ({str(e)[:80]})", flush=True)
            continue
        name = robot.info["trusses"][robot._tgt_k]["name"]
        own = {f["tomato"] for f in robot.info["trusses"][robot._tgt_k]["fruit"]}
        state = dict(prev=None, n=0, rows=[])

        def on_frame(_sent, state=state, plan=plan, name=name, own=own):
            leg = plan.leg
            last_squeeze = state["prev"] == "squeeze" and leg != "squeeze"
            state["n"] = state["n"] + 1 if leg == state["prev"] else 0
            if last_squeeze or (leg in CARRY_LEGS and state["n"] % a.every == 0):
                state["rows"].append(dict(leg="squeeze_end" if last_squeeze else leg, frame=state["n"],
                                          **sample(robot, name, own)))
            state["prev"] = leg
            return None

        t0, watch = gt.play(robot, plan, on_frame=on_frame)
        ok, why = gt.judge(robot, plan, t0, watch)
        rec = dict(seed=seed, noslip=a.noslip, impratio=float(robot.m.opt.impratio), ok=ok, why=str(why)[:100],
                   swing_deg=plan.meta.get("release_swing_deg"), align_deg=plan.meta.get("carry_align_deg"),
                   phase=plan.meta.get("phase"), release=plan.meta.get("release_point"),
                   samples=state["rows"])
        print(f"seed {seed}: ok {ok} {str(why)[:60]}  align {rec['align_deg']} phase {rec['phase']}"
              f" detour {plan.meta.get('detours', {}).get('to_crate')} retreat {plan.meta.get('retreat_option')}",
              flush=True)
        for r in state["rows"]:
            g = r.get("gravity_nm")
            line = (f"   {r['leg']:13s} {r['frame']:3d}  contacts {r['n_contacts']}"
                    f"  stalk {r['stalk_to_pads_line_deg']:5.1f} deg")
            if g:
                line += (f"  gravity x {abs(g[0]):.3f}  carried {abs(r['carried_x_nm']):.3f}"
                         f"  capacity {r['capacity_x_nm']:.3f} N.m")
            print(line, flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
