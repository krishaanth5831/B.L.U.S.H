#!/usr/bin/env python3
"""Checks --robot.type=so101_mujoco the way lerobot uses it, with no arms plugged in.

  python check_plugin.py           PASS/FAIL per check, exit 1 on a FAIL

It builds the robot through lerobot's own factory, drives it at 30 fps, records a 3 s episode with
lerobot-record's feature pipeline and the AV1 encoder, reads it back, and sends one frame to
rerun. Everything it writes goes to a temp dir that is deleted afterwards. Run in the lerobot env
after `pip install -e plugin`.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")   # headless; the commands in sim/scripts open the windows

import numpy as np  # noqa: E402

FAILS = 0


def report(ok, msg):
    global FAILS
    FAILS += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def main():
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()                 # what every lerobot command does first
    from lerobot.robots.config import RobotConfig
    report("so101_mujoco" in RobotConfig.get_known_choices(), "lerobot discovers --robot.type=so101_mujoco")

    tmp = tempfile.mkdtemp(prefix="so101_mujoco_check_")
    cfg = RobotConfig.get_choice_class("so101_mujoco")(viewer=False, overlay="true_box", state_log_dir=f"{tmp}/states")
    robot = make_robot_from_config(cfg)
    report(robot.name == "so_follower", "robot_type matches the real follower (sim + real datasets can merge)")
    robot.connect()

    # Drive: a slow sweep of every joint, in normalised units, like a leader would send.
    fps, secs = 30, 3.0
    t_act, obs_log, act_log = [], [], []
    for k in range(int(fps * secs)):
        t0 = time.perf_counter()
        s = np.sin(2 * np.pi * k / (fps * secs))
        act = {f"{j}.pos": v for j, v in zip(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
            (30 * s, -20 + 10 * s, 20 * s, 15 * s, 40 * s, 50 + 40 * s))}
        sent = robot.send_action(act)
        obs = robot.get_observation()
        obs_log.append(obs); act_log.append(sent)
        t_act.append(time.perf_counter() - t0)
        time.sleep(max(0, 1 / fps - (time.perf_counter() - t0)))
    o = obs_log[-1]
    report(all(o[c].shape == (480, 640, 3) and o[c].dtype == np.uint8 for c in ("birdseye", "claw")),
           "cameras birdseye + claw come back as 480x640x3 uint8")
    js = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    S = np.array([[ob[f"{j}.pos"] for j in js] for ob in obs_log])[15:]
    A = np.array([[a[f"{j}.pos"] for j in js] for a in act_log])
    errs = [np.abs(S - A[15 - lag:len(A) - lag]).mean() for lag in range(0, 10)]
    lag = int(np.argmin(errs))
    report(errs[lag] < 3, f"sim arm follows the actions: mean error {errs[lag]:.2f} (normalised units) "
                          f"at a lag of {lag} frames ({lag * 1000 / fps:.0f} ms)")
    report(np.percentile(t_act, 95) < 1 / fps, f"send_action + get_observation p95 {np.percentile(t_act, 95)*1000:.1f} ms "
                                               f"(budget {1000/fps:.1f} ms at {fps} fps)")
    green = (o["birdseye"][..., 1] == 255) & (o["birdseye"][..., 0] == 0) & (o["birdseye"][..., 2] == 0)
    report(green.sum() > 50, f"true_box overlay drawn on birdseye ({int(green.sum())} box pixels)")

    # Record exactly like lerobot-record: default processors -> features -> frames -> AV1 episode.
    tap, rap, rop = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=tap, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=rop, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True))
    ds = LeRobotDataset.create("local/so101_mujoco_check", fps, root=f"{tmp}/ds", robot_type=robot.name,
                               features=features, use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1"))
    for obs, act in zip(obs_log, act_log):
        frame = {**build_dataset_frame(ds.features, rop(obs), prefix=OBS_STR),
                 **build_dataset_frame(ds.features, tap((act, obs)), prefix=ACTION),
                 "task": "Pick the ripe cherry tomato and place it in the crate"}
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    back = LeRobotDataset("local/so101_mujoco_check", root=f"{tmp}/ds")
    item = back[len(back) - 1]
    keys = sorted(k for k in item if k.startswith(("observation", "action")))
    report(len(back) == len(obs_log) and "observation.images.birdseye" in keys and "observation.images.claw" in keys,
           f"AV1 episode written and read back: {len(back)} frames, keys {keys}")
    st = item["observation.state"].numpy()
    report(np.all(np.abs(st[:5]) <= 100.5) and 0 <= st[5] <= 100.5,
           "observation.state is in use_degrees=false units (arm within ±100, gripper 0..100)")
    print("     dataset features:", {k: v["shape"] for k, v in back.meta.features.items() if k.startswith(("obs", "act"))})

    robot.disconnect()
    logs = os.listdir(f"{tmp}/states")
    report(len(logs) == 1 and any(f.endswith(".npz") for f in os.listdir(f"{tmp}/states/{logs[0]}")),
           "full sim state logged per frame for re-renders")

    # Rerun: the same call lerobot-teleoperate --display_data=true makes, into an in-memory recording.
    import rerun as rr
    from lerobot.utils.rerun_visualization import log_rerun_data
    rr.init("so101_mujoco_check", spawn=False)
    rr.memory_recording()
    try:
        log_rerun_data(observation=obs_log[-1], action=act_log[-1])
        report(True, "rerun accepts the observation (both camera images) and the action")
    except Exception as e:  # noqa: BLE001
        report(False, f"rerun logging: {e}")

    import shutil
    shutil.rmtree(tmp)
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
