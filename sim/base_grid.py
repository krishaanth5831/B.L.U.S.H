#!/usr/bin/env python3
"""Base position + start pose search for the v6 world (K7 2026-11-24: "better starting position of the arm instead of
it being at extremes of the shoulder pan motor and make sure that the camera is facing the row when starting ...
maybe place the base of the arm slightly more away from the row"; post-mortem §5.8-5.9).

    python base_grid.py --start-only --base 0.34,0.16          the start pose this base gets (no scenes)
    python base_grid.py --base 0.34,0.16 --scenes 30 --out ../outputs/v6/base_grid.jsonl
    python base_grid.py --base 0.37,0.20 --old-start ...       today's base and start pose, for comparison

One candidate per process: scene.ARM_BASE / ARM_YAW_DEG / START_NORM are set before anything is built (the IK models
are built once per process from them). Per candidate:
  1. the start pose (unless --old-start): a kinematic search over the 4 big joints, wrist roll 0, jaw half open, for
     the pose whose claw camera points closest at the truss band's middle (scene.BASE_CAM_AIM), with every joint at
     least START_MARGIN_NORM from its stop, the camera START_STANDOFF_M away from the aim, and the gripper at least
     START_PLANT_CLEAR_M in front of the stems (it never starts in the plants).
  2. per tuning scene (fresh seeds, as they come; the plugin draws the layout with this base): the teacher's plan
     (found or why not) and a dry run of it (--retries more plans if it fails), judged by generate_truss.judge.
Teacher v5 on the v6 jaw by default (--jaw). Writes one JSON line per scene plus one header line per candidate.
Run in the lerobot-mj4 env, outside the Claude sandbox.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene  # noqa: E402

START_MARGIN_NORM = 25.0      # every joint at least this far from its stop (norm units, stops at +-100): the v5 start
                              # pose had the pan at -90 and the elbow at +90 (post-mortem §5.9)
START_STANDOFF_M = (0.15, 0.32)   # claw camera to the band's middle: close enough to see trusses, far from the plant
START_PLANT_CLEAR_M = 0.16    # the gripper starts at least this far in front of the stem line (fruit hangs 5-8 cm out).
                              # 0.10 (first grid, 2026-11-24): more carry damage than today's start pose; 0.20 put
                              # the band 33-44 deg off the claw camera's axis (its view is +-33 deg vertically)
START_Z_M = (0.20, 0.45)      # gripper height at the start: above the crate's rim, below the crop wire (CROP_Z 0.52)


def find_start(jaw):
    """The start pose (norm, 6) for the current scene.ARM_BASE / ARM_YAW_DEG, and its score."""
    import mujoco

    import arm_map
    fm = arm_map.FollowerMap()
    m, d = scene.compose("", "", "", arm=True, scenery=False, cut_hold=True, jaw=jaw)
    qidx, _ = scene.arm_ids(m)
    cam = m.camera("arm/claw").id
    grip = m.body("arm/gripper").id
    aim = np.asarray(scene.BASE_CAM_AIM, float)
    stem_y = scene.TARGET_ROW - np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]) * scene.KEYHOLE_OFF
    toward_row = np.sign(scene.TARGET_ROW - scene.ARM_BASE[1])
    lim = 100.0 - START_MARGIN_NORM
    best = None
    for pan in np.arange(-lim, lim + 0.1, 5.0):
        for lift in np.arange(-lim, lim + 0.1, 10.0):
            for elbow in np.arange(-lim, lim + 0.1, 10.0):
                for wf in np.arange(-lim, lim + 0.1, 10.0):
                    n = np.array([pan, lift, elbow, wf, 0.0, 50.0])
                    d.qpos[qidx] = fm.norm_to_rad(n)
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_camlight(m, d)
                    g = d.xpos[grip]
                    if not (START_Z_M[0] <= g[2] <= START_Z_M[1]):
                        continue
                    if toward_row * (stem_y - g[1]) < START_PLANT_CLEAR_M:
                        continue
                    c = d.cam_xpos[cam]
                    v = aim - c
                    dist = np.linalg.norm(v)
                    if not (START_STANDOFF_M[0] <= dist <= START_STANDOFF_M[1]):
                        continue
                    view = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
                    ang = float(np.degrees(np.arccos(np.clip(view @ v / dist, -1, 1))))
                    if best is None or ang < best[0]:
                        best = (ang, n.copy(), float(dist), g.copy())
    return best


def damage_where(robot, plan, t0, watch, fps):
    """Where a dry run's first damage started (a fruit off, or a neighbour's fruit past generate_truss's
    NEIGHBOUR_MOVE_M): its leg, the share of that leg done (0..1), the fruit and whether it is on the target truss.
    None when nothing came off or moved. For the retry rule (to_crate damage early in the leg = leaving the plant)."""
    k = robot._tgt_k
    own = {f["tomato"] for f in robot.info["trusses"][k]["fruit"]}
    ev = [(float(robot.det.detached_at[j] - t0), f["tomato"]) for j, f in enumerate(robot.info["fruit"])
          if not np.isnan(robot.det.detached_at[j])]
    if watch.over_t is not None:
        ev.append((float(watch.over_t - t0 - 0.5 / fps), watch.over))
    if not ev:
        return None
    t, name = min(ev)
    i = int(t * fps)
    for leg, n in plan.phases:
        if i < n:
            return dict(leg=leg, frac=round(i / max(n, 1), 2), fruit=name, own=name in own)
        i -= n
    return dict(leg="after", frac=None, fruit=name, own=name in own)


def weld_hook(robot, plan, log):
    """--whatif-weld (WHAT-IF ONLY, scene.HOLD_WELD): play()'s on_frame. At the first retreat frame, if the target was
    cut and touches a pad, its hold weld goes on at the pose it has in the gripper right then (relpose = the gripper in
    the truss frame); at the first release / drop frame it goes off. The truss then cannot turn, slide or swing in the
    grip between the cut and the crate. log gets when it went on/off and the pad contacts at switch-on."""
    import mujoco
    m, d = robot.m, robot.d
    name = robot.info["trusses"][robot._tgt_k]["name"]
    eq, cut = m.equality(f"{name}_truss_hold").id, m.equality(f"{name}_cut").id
    bt, bg = m.body(f"{name}_truss").id, m.body("arm/gripper").id
    stalk = {m.geom(f"{name}_peduncle").id, m.geom(f"{name}_rachis").id}
    pads = {m.geom(g).id for g in ("arm/pad_fixed", "arm/pad_moving")}

    def in_grip():
        Rg = d.xmat[bg].reshape(3, 3)
        return Rg.T @ (d.xpos[bt] - d.xpos[bg]), Rg.T @ d.xmat[bt].reshape(3, 3)

    def on_frame(_sent):
        leg = plan.leg
        if leg == "retreat" and "on" not in log and not d.eq_active[cut]:
            touch = sum(1 for c in d.contact[:d.ncon]
                        if {c.geom1, c.geom2} & stalk and {c.geom1, c.geom2} & pads)
            log["pad_contacts"] = touch
            if touch:
                m.eq_data[eq, 0:3] = 0.0
                m.eq_data[eq, 3:6] = d.xmat[bt].reshape(3, 3).T @ (d.xpos[bg] - d.xpos[bt])
                qn, q = np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(qn, d.xquat[bt])
                mujoco.mju_mulQuat(q, qn, d.xquat[bg])
                m.eq_data[eq, 6:10] = q
                d.eq_active[eq] = 1
                log["on"] = round(float(d.time), 2)
                log["_rel"] = in_grip()
            else:
                log["on"] = None
        elif leg in ("release", "drop") and d.eq_active[eq]:
            d.eq_active[eq] = 0
            log["off"] = round(float(d.time), 2)
            (p0, R0), (p1, R1) = log.pop("_rel"), in_grip()     # self-check: ~0 if the weld held where it started
            log["drift_mm"] = round(float(np.linalg.norm(p1 - p0)) * 1e3, 1)
            log["drift_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))), 1)
        return None
    return on_frame


def attempt_record(plan, why, where):
    """One dry run of a scene, for the retry rule: the plan's candidate (cut, approach, retreat, release pose) + why."""
    m = plan.meta
    rp = m.get("release_point")
    return dict(phase=m.get("phase"), mode=m.get("neighbours"), cut_u=m.get("cut_u"),
                beta=m.get("approach_blend_deg"), tilt=m.get("tilt_deg"), retreat=str(m.get("retreat_option")),
                release=None if rp is None else [round(float(x), 3) for x in rp],
                fallback=str(m.get("release_fallback")), swing_deg=m.get("release_swing_deg"),
                align_deg=m.get("carry_align_deg"), detour=str((m.get("detours") or {}).get("to_crate")),
                detour_in=str((m.get("detours") or {}).get("to_standoff")),
                why=str(why)[:120], damage=where)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="x,y[,yaw_deg] of the arm base (world, m)")
    ap.add_argument("--old-start", action="store_true", help="keep today's scene.START_NORM")
    ap.add_argument("--start-only", action="store_true")
    ap.add_argument("--jaw", default="v6", choices=scene.JAWS)
    ap.add_argument("--seed-start", type=int, default=9730000, help="tuning seeds (unused by any dataset or test)")
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--seeds", help="comma-separated seeds instead of --seed-start/--scenes")
    ap.add_argument("--retries", type=int, default=0, help="more plans after a failed dry run (as generate_truss)")
    ap.add_argument("--plan-only", action="store_true",
                    help="census: plan each scene, no dry run; log the plan's phase or what blocked it (HIT_LOG)")
    ap.add_argument("--build-only", action="store_true",
                    help="census: build each scene, no plan, no dry run; log every truss (name, knuckle in mm, fruit), "
                         "which the arm can cut from here and the target. ⚠️ Which truss is ripe depends on the jaw, "
                         "the aim point, the base and the start pose (truss_scene.build), and a seed's layout can "
                         "differ too (redrawn while no truss is reachable; trusses touching the arm are removed)")
    ap.add_argument("--world", choices=("v5", "v6"), default=None,
                    help="the dataset world (truss_scene.WORLDS): v5 = the 23 x 28 cm crate of dataset v5 and its "
                         "held-out sets, v6 = the code default. Each seed draws different plants in each world.")
    ap.add_argument("--crate", help="WHAT-IF ONLY: 'sx,sy,cx,cy' (m) = another crate size and centre; the plants stay "
                                    "as with the standard crate (truss_scene.KEEP_OUT_CRATE). ⚠️ 30 x 28 is the most "
                                    "that fits the greenhouse (truss_scene.CRATE_SIZE note)")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--dedup", action="store_true",
                    help="retries skip plans repeating a release pose + retreat that already failed (gt.RETRY_DEDUP)")
    ap.add_argument("--whatif-weld", action="store_true",
                    help="WHAT-IF ONLY: weld the cut target to the gripper from the retreat to the release "
                         "(scene.HOLD_WELD), the ceiling any pad design could reach")
    a = ap.parse_args()
    scene.HOLD_WELD = a.whatif_weld
    b = [float(v) for v in a.base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    if len(b) > 2:
        scene.ARM_YAW_DEG = b[2]
    head = dict(base=list(scene.ARM_BASE), yaw_deg=scene.ARM_YAW_DEG, jaw=a.jaw, old_start=a.old_start,
                whatif_weld=a.whatif_weld, blush_v6=os.environ.get("BLUSH_V6"),   # the A/B switches of this run
                crate_whatif=a.crate)
    if not a.old_start:
        st = find_start(a.jaw)
        if st is None:
            sys.exit(f"no start pose for base {a.base} within the limits")
        ang, n, dist, g = st
        scene.START_NORM = tuple(float(v) for v in n)
        head.update(start_norm=list(scene.START_NORM), cam_off_aim_deg=round(ang, 1), cam_to_aim_m=round(dist, 3),
                    gripper_start=np.round(g, 3).tolist())
    else:
        head.update(start_norm=list(scene.START_NORM))
    print(json.dumps(head), flush=True)
    if a.start_only:
        return

    import expert
    import generate_truss as gt
    import truss_expert as tx
    import truss_scene as ts
    import truss_teacher as tt
    if a.world:
        ts.set_world(a.world)
    if a.crate:
        sx, sy, cx, cy = (float(v) for v in a.crate.split(","))
        ts.KEEP_OUT_CRATE = (ts.CRATE_CENTRE, ts.CRATE_SIZE)
        ts.CRATE_CENTRE, ts.CRATE_SIZE = (cx, cy), (sx, sy, ts.CRATE_SIZE[2])
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    cfg = SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS,
                            domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig="base",
                            start_jitter_deg=0.0, yolo_weights=None, jaw=a.jaw)
    robot = SO101Mujoco(cfg)
    robot.connect()
    out = a.out.open("a") if a.out else None
    if out:
        out.write(json.dumps(dict(header=True, **head)) + "\n")
        out.flush()
    n_plan = n_ok = 0
    t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else range(a.seed_start, a.seed_start + a.scenes)
    a.scenes = len(seeds)
    for seed in seeds:
        t_s = time.time()
        rec = dict(seed=seed, base=list(scene.ARM_BASE), old_start=a.old_start, crate=list(ts.CRATE_SIZE),
                   world=ts.world_name())
        robot.new_scene(seed, log=False)
        robot.get_observation()
        if robot._tgt_k is not None:
            rec["target"] = robot.info["trusses"][robot._tgt_k]["name"]
        if a.build_only:
            T = robot.info["trusses"]
            rec.update(layout=[[t["name"], *(round(float(v) * 1000) for v in t["knuckle"]), t["n"]] for t in T],
                       reachable=[T[k]["name"] for k in robot.info["reachable"]],
                       ready=[T[k]["name"] for k in robot.info["ready"]])
            if robot._tgt_k is not None:
                rec["target_knuckle_mm"] = [round(float(v) * 1000) for v in T[robot._tgt_k]["knuckle"]]
        elif robot._tgt_k is None:
            rec.update(plan=False, ok=False, why=f"no target ({robot._box_info['reason']})")
        else:
            tt.HIT_LOG.clear()
            tt.POOL_LOG.clear()
            try:
                plan = tt.TrussTeacher(robot.m, robot.d, robot.info, robot._tgt_k).plan()
                rec["plan"] = True
            except tx.PlanError as e:
                plan = None
                rec.update(plan=False, ok=False, why=f"no plan: {str(e)[:400]}")
                rec["blocked_by"] = [[r_, k_, n_] for (r_, k_), n_ in tt.HIT_LOG.most_common()]
            if a.plan_only:
                if plan is not None:
                    rec.update(phase=plan.meta.get("phase"), crate_slack_mm=plan.meta.get("crate_slack_mm"),
                               release_fallback=plan.meta.get("release_fallback"),
                               release_swing_deg=plan.meta.get("release_swing_deg"),
                               release_pool=plan.meta.get("release_pool"),
                               detours={k: str(v) for k, v in (plan.meta.get("detours") or {}).items()})
                pl = np.array(tt.POOL_LOG or [(0, 0)])
                rec["fits"] = dict(n=len(tt.POOL_LOG), capped_empty=int((pl[:, 1] == 0).sum()),
                                   median_fit=float(np.median(pl[:, 0])), median_capped=float(np.median(pl[:, 1])))
                plan = None
            k, failed_keys = 0, set()
            while plan is not None:
                if k:
                    robot.new_scene(seed, log=False)
                    robot.get_observation()
                wlog = {}
                t_play, watch = gt.play(robot, plan, on_frame=weld_hook(robot, plan, wlog) if a.whatif_weld else None)
                ok, why = gt.judge(robot, plan, t_play, watch)
                rec.update(ok=ok, why=why, tries=k + 1, phase=plan.meta.get("phase"))
                rec.setdefault("attempts", []).append(
                    attempt_record(plan, why, damage_where(robot, plan, t_play, watch, expert.FPS)))
                if a.whatif_weld:
                    rec["attempts"][-1]["weld"] = wlog
                # second reading, never the verdict: how far a neighbour's fruit swung at most (the judge's limit
                # gt.NEIGHBOUR_MOVE_M is a guess, 2026-11-03) and the verdict with that one rule left out, so
                # scripts/grid_ab.py --swing can say what any other limit would have scored
                keep_lim, gt.NEIGHBOUR_MOVE_M = gt.NEIGHBOUR_MOVE_M, float("inf")
                try:
                    rec["attempts"][-1].update(nb_mm=round(watch.max * 1000, 1),
                                               why_noswing=str(gt.judge(robot, plan, t_play, watch)[1])[:120])
                finally:
                    gt.NEIGHBOUR_MOVE_M = keep_lim
                if ok or k >= a.retries:
                    break
                k += 1
                try:
                    gt.next_plan_after(plan, why, failed_keys, dedup=a.dedup)
                except tx.PlanError:
                    break
        n_plan += bool(rec.get("plan"))
        n_ok += bool(rec.get("ok"))
        rec["wall_s"] = round(time.time() - t_s, 1)
        print(f"seed {seed}: plan {rec.get('plan')} ok {rec.get('ok')}  {str(rec.get('why'))[:90]}", flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    print(f"base {a.base} {'old' if a.old_start else 'new'} start: plan {n_plan}/{a.scenes}, ok {n_ok}/{a.scenes} "
          f"in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
