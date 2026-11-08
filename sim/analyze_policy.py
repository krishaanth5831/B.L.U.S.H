#!/usr/bin/env python3
"""Why does the policy close on the wrong fruit? Headless experiments on a checkpoint (K7, 2026-11-01).

    python analyze_policy.py <checkpoint>/pretrained_model [--n 10]

Every episode is closed loop (the policy drives the sim, as in eval_policy.py), no windows, not real time.
At the moment the policy CLOSES the jaw (commanded gripper.pos falls 15 below its peak, after 1 s), the
point between the two fin-ray fingers is compared with every fruit: which fruit is nearest, how far the
target is, how far the nearest green / other ripe fruit is. Experiments:
  train   seeds the policy trained on (same target as in training): is it precise where it has seen data?
  novel   new seeds: does it generalise?
  swap    the novel scenes with 2+ visible ripe fruits, run again with the green box moved to ANOTHER
          ripe fruit: if the policy reads the box, its close point moves to the new box
  nobox   the novel scenes again with no green box drawn: if nothing changes, the box is ignored
Writes ../outputs/analysis/<stamp>_<n or ens>/episodes.jsonl and the start birdseye frame of each episode (PNG).
⚠️ Single-fruit task only, and it can't run as is: the features come from eval_policy.TRAIN_DATASET
(blush_sim_scripted_v1), deleted locally and on the Hub 2026-11-03 (K7).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_policy as ev  # noqa: E402
import expert  # noqa: E402
import scene  # noqa: E402
import target as tg  # noqa: E402
from add_box_feature import box_vector  # noqa: E402
from generate_scripted import task_string  # noqa: E402

CLOSE_DROP = 15.0         # norm units below the peak gripper command = "closing" (open 50-70, close ~1)
EPISODE_S = 30.0          # the close happens 6-12 s into the scripted episodes; 30 s covers a slow policy
JAW_MID = ("arm/finray_fixed_f3", "arm/finray_moving_f3")   # middle ribs of the two fingers
GRIPPER = "arm/gripper"   # gripper frame: fingers along -z, jaw closes along x (check_grasp.find_pose)


def jaw_mid(robot):
    return (robot.d.xpos[robot.m.body(JAW_MID[0]).id] + robot.d.xpos[robot.m.body(JAW_MID[1]).id]) / 2


def in_gripper(robot, body):
    """A body's centre in the gripper frame (m)."""
    G = robot.m.body(GRIPPER).id
    return (robot.d.xpos[body] - robot.d.xpos[G]) @ robot.d.xmat[G].reshape(3, 3)


def set_target(robot, k):
    """Move the green box (and the grading) onto fruit k, using its true birdseye box."""
    boxes = tg.fruit_boxes(robot._cameras(), robot.d, robot.m, robot.info)
    if boxes[k][0] is None:
        return False
    robot._tgt_k, robot._tgt_body = k, robot.m.body(robot.info["fruit"][k]["tomato"]).id
    robot._box = list(boxes[k][0])
    robot._box_info = {**robot._box_info, "target": k, "target_name": robot.info["fruit"][k]["tomato"],
                       "source": "forced", "box": robot._box}
    return True


def episode(robot, policy, pre, post, features, rop, task, device, seed, mode, swap_to=None, png=None, full=False):
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    robot.new_scene(seed, log=False)
    obs = robot.get_observation()
    if robot._tgt_k is None:
        return None
    if mode == "swap" and not set_target(robot, swap_to):
        return None
    if mode == "nobox":
        robot._box = None
    obs = robot.get_observation()                   # redrawn with the moved / removed box
    if png is not None:
        cv2.imwrite(str(png), cv2.cvtColor(obs["birdseye"], cv2.COLOR_RGB2BGR))
    fruit = robot.info["fruit"]
    k = robot._tgt_k
    bodies = [robot.m.body(f["tomato"]).id for f in fruit]
    policy.reset()
    t0, peak, close, min_d = robot.d.time, -1e9, None, np.full(len(fruit), np.inf)
    min_d_t = (np.inf, None)                        # nearest the jaw got to the target, and when
    while robot.d.time - t0 < EPISODE_S:
        frame = build_dataset_frame(features, rop(obs), prefix=OBS_STR)
        if policy.config.env_state_feature:      # the box as numbers (add_box_feature.py)
            frame[OBS_ENV_STATE] = box_vector(robot._box)
        with torch.inference_mode():
            act = post(policy.select_action(pre(prepare_observation_for_inference(frame, device, task, robot.name))))
        a = make_robot_action(act, features)
        robot.send_action(a)
        obs = robot.get_observation()
        jm = jaw_mid(robot)
        d = np.linalg.norm(robot.d.xpos[bodies] - jm, axis=1)
        min_d = np.minimum(min_d, d)
        g, t = a["gripper.pos"], robot.d.time - t0
        peak = max(peak, g)
        if d[k] < min_d_t[0]:
            min_d_t = (d[k], t)
        if close is None and t > 1.0 and g < peak - CLOSE_DROP:
            close = dict(t=round(t, 2), d=d.copy(), jaw=jm.copy(), tg=in_gripper(robot, bodies[k]))
            close["fixed"] = in_gripper(robot, robot.m.body(JAW_MID[0]).id)
            close["moving"] = in_gripper(robot, robot.m.body(JAW_MID[1]).id)
            if mode != "train" and not full:        # the close point is what's measured; finish early
                break
        # after the close: every 0.25 s for 4 s, the gripper command, how far the jaw has shut and where
        # the target sits in the gripper (held fruit stays put; one squeezed out or missed drifts off)
        if close is not None and t - close["t"] <= 4.0 and round((t - close["t"]) * 30) % 8 == 0:
            close.setdefault("trace", []).append(dict(
                t=round(t - close["t"], 2), grip=round(float(g), 1),
                gap_mm=round(float(in_gripper(robot, robot.m.body(JAW_MID[1]).id)[0]
                                   - in_gripper(robot, robot.m.body(JAW_MID[0]).id)[0]) * 1000, 1),
                tg_mm=np.round(in_gripper(robot, bodies[k]) * 1000, 1).tolist(),
                detached=bool(not np.isnan(robot.det.detached_at[k]))))
    rec = dict(seed=seed, mode=mode, target=fruit[k]["tomato"],
               ripe=[fruit[j]["tomato"] for j in robot.info["ripe"]],
               target_min_mm=round(float(min_d[k]) * 1000, 1), t_target_min=round(min_d_t[1] or -1, 2))
    if close is None:
        rec.update(closed=False)
        return rec
    d = close["d"]
    near = int(np.argmin(d))
    others = [j for j in range(len(fruit)) if j != k]
    green = [j for j in others if fruit[j]["ripeness"] != "ripe"]
    ripe_o = [j for j in others if fruit[j]["ripeness"] == "ripe"]
    # Target centre in the gripper frame at the close, minus where the scripted picker puts it
    # (expert.grasp_point: skin 3 mm off the fixed finger, mid finger width, depth drawn from GRASP_DEPTH_M).
    r = fruit[k]["radius"]
    ideal = expert.grasp_point(r, float(np.mean(expert.GRASP_DEPTH_M)))
    err = close["tg"] - ideal
    rec.update(radius_mm=round(r * 1000, 1), in_gripper_mm=np.round(close["tg"] * 1000, 1).tolist(),
               err_across_mm=round(float(err[0]) * 1000, 1), err_side_mm=round(float(err[1]) * 1000, 1),
               err_depth_mm=round(float(err[2]) * 1000, 1),
               # fixed / moving finger middle ribs in the gripper frame at the close
               fixed_f3_mm=np.round(close["fixed"] * 1000, 1).tolist(),
               moving_f3_mm=np.round(close["moving"] * 1000, 1).tolist(), trace=close.get("trace", []))
    rec.update(closed=True, t_close=close["t"], target_mm=round(float(d[k]) * 1000, 1),
               nearest=fruit[near]["tomato"], nearest_ripeness=fruit[near]["ripeness"],
               nearest_is_target=near == k, nearest_mm=round(float(d[near]) * 1000, 1),
               nearest_green_mm=round(float(min(d[green])) * 1000, 1) if green else None,
               nearest_other_ripe_mm=round(float(min(d[ripe_o])) * 1000, 1) if ripe_o else None,
               # target-to-neighbour spacing in this scene: how precise the reach has to be
               target_to_nearest_other_mm=round(float(min(np.linalg.norm(
                   robot.d.xpos[[bodies[j] for j in others]] - robot.d.xpos[bodies[k]], axis=1))) * 1000, 1),
               jaw=np.round(close["jaw"], 4).tolist(), target_xyz=np.round(robot.d.xpos[bodies[k]], 4).tolist(),
               fruit_xyz={fruit[j]["tomato"]: np.round(robot.d.xpos[bodies[j]], 4).tolist() for j in range(len(fruit))})
    if mode == "train" or full:
        det = robot.det
        rec["success"] = bool(scene.in_crate(robot.d.xpos[bodies[k]]))
        rec["picked"] = bool(not np.isnan(det.detached_at[k]))
        rec["others_off"] = [fruit[j]["tomato"] for j in others if not np.isnan(det.detached_at[j])]
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=10, help="scenes per experiment")
    ap.add_argument("--seed-start", type=int, default=1_500_000_000)
    ap.add_argument("--modes", default="train,novel,nobox,swap")
    ap.add_argument("--full", action="store_true",
                    help="run novel / swap / nobox episodes to the end too (picked, success), not just to the close")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="re-plan every N steps instead of the trained 100 (no retraining needed)")
    ap.add_argument("--ensemble", type=float, default=None,
                    help="ACT temporal ensembling coefficient (the ACT paper's 0.01): re-plan every step, "
                         "average chunks")
    a = ap.parse_args()
    path = a.policy.resolve()
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco

    features = LeRobotDatasetMetadata(ev.TRAIN_DATASET, root=HF_LEROBOT_HOME / ev.TRAIN_DATASET).features
    _, _, rop = make_default_processors()
    device = torch.device("cuda")
    policy, pre, post = ev.load_policy(path, device)
    if a.ensemble is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff, policy.config.n_action_steps = a.ensemble, 1
        policy.temporal_ensembler = ACTTemporalEnsembler(a.ensemble, policy.config.chunk_size)
    elif a.n_action_steps is not None:
        policy.config.n_action_steps = a.n_action_steps
    print(f"execution: n_action_steps {policy.config.n_action_steps}, ensemble {policy.config.temporal_ensemble_coeff}",
          flush=True)
    modes = a.modes.split(",")
    task = task_string()
    robot = SO101Mujoco(ev.sim_config(viewer=False, overlay="yolo"))
    robot.connect()
    tag = "ens" if a.ensemble is not None else f"n{policy.config.n_action_steps}"
    out = HERE.parent / "outputs" / "analysis" / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "episodes.jsonl").open("w")

    def run(seed, mode, swap_to=None):
        tag = f"{mode}_{seed}" + (f"_to{swap_to}" if swap_to is not None else "")
        r = episode(robot, policy, pre, post, features, rop, task, device, seed, mode, swap_to, out / f"{tag}.png",
                    full=a.full)
        if r is not None:
            log.write(json.dumps(r) + "\n"); log.flush()
            print(f"{mode:6s} seed {seed}: target {r['target']}, "
                  + (f"closed {r['target_mm']} mm from it, nearest {r['nearest']} ({r['nearest_ripeness']}, "
                     f"{r['nearest_mm']} mm) at {r['t_close']} s (nearest {r['target_min_mm']} mm at "
                     f"{r['t_target_min']} s); in-gripper error across {r['err_across_mm']} side {r['err_side_mm']} "
                     f"depth {r['err_depth_mm']} mm"
                     + (f", picked {r['picked']}, success {r['success']}" if "success" in r else "")
                     if r["closed"] else f"never closed (came within {r['target_min_mm']} mm)"), flush=True)
        return r

    try:
        seeds = json.loads((ev.CACHE / f"train_targets_{ev.TRAIN_DATASET.split('/')[-1]}.json").read_text())["seeds"]
        for s in seeds[:a.n] if "train" in modes else []:
            run(s, "train")
        novel, seed = [], a.seed_start
        while len(novel) < a.n:
            r = run(seed, "novel")
            if r is not None:
                novel.append(r)
            seed += 1
        for r in novel if "nobox" in modes else []:
            run(r["seed"], "nobox")
        for r in novel if "swap" in modes else []:
            robot.new_scene(r["seed"], log=False)
            robot.get_observation()
            boxes = tg.fruit_boxes(robot._cameras(), robot.d, robot.m, robot.info)
            alt = [j for j in robot.info["ripe"] if j != robot._tgt_k and boxes[j][0] is not None]
            if alt:
                run(r["seed"], "swap", alt[0])
    finally:
        robot.disconnect()
        log.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
