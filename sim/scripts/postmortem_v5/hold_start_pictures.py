#!/usr/bin/env python3
"""Post-mortem 2026-11-20: which start pictures make the policy stay home in a nothing-ripe scene, and go in a
picking scene? Builds scenes in the simulator (first observation only, no rollout).

    conda activate lerobot && python hold_start_pictures.py [--step 100000] [--scenes 20]

Why: the smoke test of eval_truss.py --codec-still 1 (both cameras' pictures through the dataset's video encoder as a
one-frame clip before the policy plans) made the arm leave its start pose in a nothing-ripe scene. hold_or_go.py showed
the CLAW start picture does not decide stay or go. This asks the same of the BASE picture: the base camera sees the
arm, and a recorded frame carries a trace of the frame after it, so in a recorded hold episode the base start frame
says "the arm is not about to move".

Held-out scenes: the stay-home scenes of the tests (eval_truss.HOLD_SEED_START + i) and the first picking scenes of
heldout_v5.jsonl. Per scene the raw first observation, then each camera's picture raw or through the encoder as a clip
of 1 or 2 copies; the plan's largest departure from the start pose (arm joints, norm units; a hold test allows 3).
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "sim"))
import expert  # noqa: E402

REPO = "krish5831/blush_sim_teacher_v5"
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
MOVES = 10.0     # a plan "goes" above this departure (norm units; as hold_or_go.py)


def q(a, p=50):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--scenes", type=int, default=20)
    a = ap.parse_args()
    import cv2
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins
    sys.argv = sys.argv[:1]
    import eval_truss as et
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    def enc(img, n):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            d = pathlib.Path(td) / "imgs"
            d.mkdir()
            for k in range(n):
                cv2.imwrite(str(d / f"frame-{k:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            v = pathlib.Path(td) / "clip.mp4"
            encode_video_frames(d, v, expert.FPS, video_encoder=RGBEncoderConfig(vcodec="libsvtav1"), overwrite=True)
            return decode_video_frames(str(v), [0.0], 1e-3, backend="pyav", return_uint8=True)[0].permute(1, 2, 0).numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(REPO, root=HF_LEROBOT_HOME / REPO)
    ck = ROOT / "outputs" / "train" / "act_teacher_v5" / "checkpoints" / f"{a.step:06d}" / "pretrained_model"
    policy, pre, post, _ = et.load_policy(ck, device, meta, dict(temporal_ensemble_coeff=None, n_action_steps=100))
    with torch.inference_mode():
        mean_t = post(torch.zeros(1, 6, device=device))[0].to(device).float()
        std_t = post(torch.ones(1, 6, device=device))[0].to(device).float() - mean_t

    def plan(frame):
        x = pre(prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco"))
        x[OBS_IMAGES] = [x[key] for key in policy.config.image_features]
        policy.eval()
        with torch.inference_mode():
            out = policy.model(x)[0]
        return (out * std_t + mean_t)[0].cpu().numpy().astype(float)

    _, rig = et.trained_on(ck)
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    _, _, rop = make_default_processors()
    picks = [r["seed"] for r in (json.loads(ln) for ln in open(ROOT / "sim" / "heldout_v5.jsonl") if ln.strip())
             if r.get("kept") is True and not r.get("hold")][:a.scenes]
    holds = [et.HOLD_SEED_START + i for i in range(a.scenes)]
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=holds[0], viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, yolo_weights=None, state_log_dir=None, rig=rig))
    robot.connect()
    B, C = "observation.images.base", "observation.images.claw"
    names = ("both raw (tests so far)", "both through the encoder, 1 frame", "both through the encoder, 2 copies",
             "claw through the encoder (1 frame), base raw", "base through the encoder (1 frame), claw raw",
             "base through the encoder (2 copies), claw raw", "claw 1 frame, base 2 copies")
    res = {"hold": {n: [] for n in names}, "pick": {n: [] for n in names}}
    for kind, seeds in (("hold", holds), ("pick", picks)):
        for seed in seeds:
            robot.new_scene(seed, log=False, hold=kind == "hold")
            obs = robot.get_observation()
            f = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
            f[OBS_ENV_STATE] = np.asarray(robot.env_state(), np.float32)
            b1, b2, c1, c2 = enc(f[B], 1), enc(f[B], 2), enc(f[C], 1), enc(f[C], 2)
            var = {names[0]: f, names[1]: {**f, B: b1, C: c1}, names[2]: {**f, B: b2, C: c2}, names[3]: {**f, C: c1},
                   names[4]: {**f, B: b1}, names[5]: {**f, B: b2}, names[6]: {**f, B: b2, C: c1}}
            s0 = np.asarray(f["observation.state"], float)[:5]
            for n, fr in var.items():
                res[kind][n].append(float(np.abs(plan(fr)[:, :5] - s0).max()))
            print(f"{kind} seed {seed}: " + ", ".join(f"{res[kind][n][-1]:.0f}" for n in names), flush=True)
    robot.disconnect()
    print(f"\ncheckpoint {a.step // 1000}k, held-out scenes, first observation only. Largest departure of the plan from the start pose "
          f"(arm joints, norm units), median; plans that go (more than {MOVES:.0f} units)")
    print(f"   pictures the policy plans from                     {'NOTHING RIPE (should stay)':30s} PICKING (should go)")
    for n in names:
        h, p = np.array(res["hold"][n]), np.array(res["pick"][n])
        print(f"   {n:50s} {q(h):6.1f} units, {int((h > MOVES).sum()):2d}/{len(h)} go      {q(p):6.1f} units, {int((p > MOVES).sum()):2d}/{len(p)} go")


if __name__ == "__main__":
    main()
