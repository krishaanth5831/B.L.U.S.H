#!/usr/bin/env python3
"""Just watch a truss-harvesting policy run in the sim: no grading, no logs (K7, 2026-11-04).

    python rollout_truss.py ../outputs/train/smolvla_truss_lora_v1/checkpoints/last/pretrained_model

The MuJoCo viewer (third person) plus a window per camera (what the policy sees). Scene after scene at real time,
each up to EPISODE_S, new random layout every time. Keys in a camera window: N = next scene now, Q = quit.
Same scene settings as the recorded episodes (eval_truss.py has the graded version).
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_truss as ev  # noqa: E402
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402

EPISODE_S = 45.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=None, help="first scene (default: random)")
    a = ap.parse_args()
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    meta = LeRobotDatasetMetadata(ev.TRAIN_DATASET, root=HF_LEROBOT_HOME / ev.TRAIN_DATASET)
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    _, _, rop = make_default_processors()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post, pcfg = ev.load_policy(a.policy.resolve(), device, meta)
    wants_env = OBS_ENV_STATE in getattr(pcfg, "input_features", {})
    seed = a.seed if a.seed is not None else int(np.random.default_rng().integers(1_000_000, 2**31 - 100_000))
    yolo = str(gt.YOLO_TRUSS) if gt.YOLO_TRUSS.exists() else None
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=seed, viewer=True, overlay="yolo" if yolo else "true_box", realtime=False, fps=expert.FPS,
        domain_rand=True, settle_s=1.0, snap_to_first_action=False, task="truss", yolo_weights=yolo, state_log_dir=None))
    robot.connect()
    print("running: N = next scene, Q = quit (click a camera window first)", flush=True)
    first = True
    try:
        while True:
            if not first:                              # connect() already built the first scene
                print(f"building the next scene (seed {seed}) ...", flush=True)
                robot.new_scene(seed, log=False)
            first = False
            obs = robot.get_observation()
            if robot._tgt_k is None:
                seed += 1
                continue
            print(f"scene seed {seed}: target {robot.info['trusses'][robot._tgt_k]['name']}", flush=True)
            env = robot.env_state()
            policy.reset()
            t0, key = robot.d.time, 0
            while robot.d.time - t0 < EPISODE_S:
                t_f = time.time()
                frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
                if wants_env:
                    frame[OBS_ENV_STATE] = env
                with torch.inference_mode():
                    act = post(policy.select_action(pre(prepare_observation_for_inference(frame, device, gt.TASK, robot.name))))
                robot.send_action(make_robot_action(act, meta.features))
                obs = robot.get_observation()
                key = ev.show(obs, f"seed {seed}  t {robot.d.time - t0:4.1f} s")
                if key in (ord("q"), ord("n")):
                    break
                time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
            if key == ord("q"):
                break
            seed += 1
    finally:
        robot.disconnect()
        import cv2
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
