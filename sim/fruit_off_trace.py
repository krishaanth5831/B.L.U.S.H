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

    import truss_ik
    grip_pt = truss_ik.cut_point_in_gripper(t["r"])
    model_err = {}

    def predict_vs_live(R, o):
        """At the first frame above the crate: the fruit as the teacher's models put them vs where they are (mm)."""
        live = np.array([d.xpos[m.body(names[j]).id] for j in mine])
        rel = state["rel"]
        rigid = o + rel @ R.T
        P = o + R @ grip_pt
        a_ = R[:, 1]                                          # the stalk's axis (tx.CARRY_AXIS 1)
        w = np.array([radii[j] ** 3 for j in mine])
        c = (rigid * w[:, None]).sum(0) / w.sum() - P
        c_perp = c - (c @ a_) * a_
        down = np.array([0, 0, -1.0]) + a_[2] * a_
        th = np.arctan2(np.cross(c_perp, down) @ a_, c_perp @ down)
        state["th"] = th
        v = rigid - P
        hang = P + v * np.cos(th) + np.cross(a_, v) * np.sin(th) + np.outer(v @ a_, a_) * (1 - np.cos(th))
        for nm, pred in (("held as at the cut", rigid), ("hanging under the stalk", hang)):
            e = (live - pred) * 1000
            model_err[nm] = dict(mean_mm=round(float(np.linalg.norm(e, axis=1).mean()), 1),
                                 max_mm=round(float(np.linalg.norm(e, axis=1).max()), 1),
                                 mean_xyz_mm=np.round(e.mean(0), 1).tolist())

    def rot(axis, ang, v):
        return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + np.outer(v @ axis, axis) * (1 - np.cos(ang))

    def split(Rg, og, Rb, xb):
        """Above the crate: the STALK vs the models (its rachis end), and the fruit vs the stalk (in its frame)."""
        P = og + Rg @ grip_pt
        a_ = Rg[:, 1]
        re_live = xb + Rb @ state["re_loc"]
        re_rigid = og + Rg @ (state["pb_rel"] + state["Rb_rel"] @ state["re_loc"])
        re_hang = P + rot(a_, state.get("th", 0.0), (re_rigid - P)[None, :])[0]
        floc = np.array([Rb.T @ (d.xpos[m.body(names[j]).id] - xb) for j in mine])
        state["split"] = dict(
            stalk_end_vs_rigid_mm=round(float(np.linalg.norm(re_live - re_rigid) * 1000), 1),
            stalk_end_vs_hang_mm=round(float(np.linalg.norm(re_live - re_hang) * 1000), 1),
            fruit_moved_on_stalk_mm=round(float(np.linalg.norm(floc - state["floc0"], axis=1).mean() * 1000), 1))

    def on_frame(_sent):
        state["frame"] += 1
        Rb_now, xb_now = d.xmat[body].reshape(3, 3), d.xpos[body]
        if state["frame"] == 1:                          # points fixed on the stalk body, in its frame
            ped = np.asarray(t["peduncle_end"], float) - np.asarray(t["knuckle"], float)
            state["ped_loc"] = Rb_now.T @ (ped / np.linalg.norm(ped))
            state["re_loc"] = Rb_now.T @ (np.asarray(t["rachis_end"], float) - xb_now)
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
            og = d.xpos[grip]
            state["rel"] = np.array([Rg.T @ (d.xpos[m.body(names[j]).id] - og) for j in mine])
            state["body0"] = Rg.T @ (d.xpos[body] - og) - grip_pt
            state["Rb_rel"], state["pb_rel"] = Rg.T @ Rt, Rg.T @ (d.xpos[body] - og)
            state["floc0"] = np.array([Rt.T @ (d.xpos[m.body(names[j]).id] - d.xpos[body]) for j in mine])
            P0 = og + Rg @ grip_pt
            state["s_ped0"] = float(state["ped_loc"] @ (Rt.T @ (P0 - d.xpos[body])))
        if state.get("s_ped0") is not None:
            # where the pad point sits along the peduncle, in the STALK's frame: changes only if the stalk slides
            # through the pads (a stalk pivoting in the pads keeps it)
            Pn = d.xpos[grip] + Rg @ grip_pt
            s_now = float(state["ped_loc"] @ (Rt.T @ (Pn - d.xpos[body])))
            state.setdefault("ped_slide", []).append((getattr(plan, "leg", None),
                                                      round((s_now - state["s_ped0"]) * 1000, 1)))
        if state.get("body0") is not None and cutter.cut_t is not None:
            # the truss root in the jaw since the cut: along the stalk (gripper y) = sliding; its distance from that
            # axis changing = coming out of the pads (rolling about the stalk keeps it)
            pb = Rg.T @ (d.xpos[body] - d.xpos[grip]) - grip_pt
            p0 = state["body0"]
            slide = (pb[1] - p0[1]) * 1000
            rad = (np.hypot(pb[0], pb[2]) - np.hypot(p0[0], p0[2])) * 1000
            state.setdefault("slide", []).append((getattr(plan, "leg", None), round(float(slide), 1),
                                                  round(float(rad), 1)))
        ph_now = getattr(plan, "leg", None)
        if state.get("rel") is not None and ph_now == "settle_above":
            predict_vs_live(Rg, d.xpos[grip])        # overwritten each frame: the error at the END of the settle
            split(Rg, d.xpos[grip], Rt, d.xpos[body])
            if "first" not in state:
                state["first"] = {k_: dict(v_) for k_, v_ in model_err.items()}   # and as the carry stops
        if ph_now == "lift":
            state.setdefault("lift_z", []).append(round(float(d.xpos[grip][2]) * 1000, 1))
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
    print(f"plan: phase {plan.meta.get('phase')}, release fallback {plan.meta.get('release_fallback')}, modelled room "
          f"{plan.meta.get('crate_slack_mm')} mm, lowering stopped early {plan.meta.get('lower_stopped_early')}; live "
          f"re-plan {plan.meta.get('live_release')} at {plan.meta.get('live_release_point')} room "
          f"{plan.meta.get('live_crate_slack_mm')} mm (planned at {plan.meta.get('release_point')})")
    for g in plan.legs:
        if g.name == "lower" and g.W is not None and len(g.W) > 1:
            print(f"lowering path: {len(g.W)} points, largest joint step {np.abs(np.diff(g.W, axis=0)).max():.1f} norm")
    print(f"model vs live fruit above the crate, as the carry stops: {state.get('first', 'not reached')}")
    print(f"model vs live fruit above the crate, after the settle: {model_err or 'not reached'}")
    print(f"above the crate, after the settle: {state.get('split', 'not reached')}")
    ps = [x for x in state.get("ped_slide", []) if x[0] in ("squeeze", "retreat", "to_crate", "settle_above", "lower")]
    if ps:
        byp = collections.OrderedDict()
        for ph, v in ps:
            byp.setdefault(ph, []).append(v)
        print("pad point along the peduncle since the cut (mm, stalk frame = true sliding): " +
              "; ".join(f"{ph} end {v[-1]} (max |{max(abs(x) for x in v)}|)" for ph, v in byp.items()))
    sl = [x for x in state.get("slide", []) if x[0] in ("squeeze", "retreat", "to_crate", "settle_above", "lower")]
    if sl:
        by = collections.OrderedDict()
        for ph, a_, r_ in sl:
            by.setdefault(ph, []).append((a_, r_))
        print("stalk in the jaw since the cut (mm; along the stalk = sliding, off the axis = leaving the pads): " +
              "; ".join(f"{ph} end along {v[-1][0]} off {v[-1][1]} (max |along| {max(abs(x[0]) for x in v)})"
                        for ph, v in by.items()))
    lz = state.get("lift_z", [])
    if lz:
        print(f"gripper height in the lift: first {lz[:6]} ... last {lz[-1]} mm (a dip = pushing the dropped truss)")
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
