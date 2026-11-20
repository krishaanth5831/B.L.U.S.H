#!/usr/bin/env python3
"""Post-mortem 2026-11-20: does the claw start picture also decide whether the policy STAYS HOME or goes?
No simulator, no training: recorded start frames through the trained policy.

    conda activate lerobot && python hold_or_go.py [--step 100000] [--episodes 20]

Why: in the dataset 1 episode in 10 is a "hold" (nothing is ripe, the arm stays at its start pose for 4-6 s). The
recorded start frame carries a trace of the frame after it (replay_raw_vs_recorded.py); in a hold episode the next
frame is the same picture, in a picking episode the arm has started to move. So the claw start picture could tell
the policy "stay" or "go" by itself. The smoke test of eval_truss.py --codec-still 1 hinted at it: in a nothing-ripe
scene the arm went off (137 units) instead of staying.

For hold episodes and picking episodes, the start frame goes through the policy with the claw picture replaced by
each candidate, and the plan's largest departure from the start pose is measured (arm joints, norm units; the test
counts a hold as "stayed" within 3 units, eval_truss.HOLD_MOVE_NORM).
"""
import argparse
import glob
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "notes" / "2026-11-19_v5_postmortem"   # generated files (git-ignored), not the repo tree
sys.path.insert(0, str(ROOT / "sim"))

REPO = "krish5831/blush_sim_teacher_v5"
SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
MOVES = 10.0     # a plan "goes" when an arm joint leaves the start pose by more than this (norm units; a choice: the
                 # stay-home test allows 3, a picking plan moves the shoulder by 40 or more)


def q(a, p=50):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--raw-claw", type=pathlib.Path, default=OUT / "first_frame_10000001_claw.png",
                    help="a picture saved by first_frame_check.py: its middle third is the simulator's raw claw start picture")
    a = ap.parse_args()
    import cv2
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES
    sys.argv = sys.argv[:1]
    import eval_truss as et

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/data/*/*.parquet"))])
    first = df[df.frame_index == 0].set_index("episode_index")
    em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/meta/episodes/*/*.parquet"))])
    em = em.set_index("episode_index")
    rows = []
    for f in LOGS:
        rows += [json.loads(ln) for ln in open(f) if ln.strip()]
    kept = [r for r in rows if r.get("kept") is True]
    by_seed = {r["seed"]: i for i, r in enumerate(kept)}
    holds = [i for i, r in enumerate(kept) if r.get("hold")][:a.episodes]
    picks = [by_seed[json.loads(ln)["seed"]] for ln in open(HERE / "train_clean_50.jsonl") if ln.strip()][:a.episodes]

    def frame_rgb(ep, cam, k):
        key = f"videos/observation.images.{cam}"
        path = (f"{SNAP}/videos/observation.images.{cam}/chunk-{int(em.loc[ep, key + '/chunk_index']):03d}/"
                f"file-{int(em.loc[ep, key + '/file_index']):03d}.mp4")
        ts = float(em.loc[ep, key + "/from_timestamp"]) + k / FPS
        return decode_video_frames(path, [ts], 1e-3, backend="pyav", return_uint8=True)[0].permute(1, 2, 0).numpy()

    def first_of_clip(frames):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            d = pathlib.Path(td) / "imgs"
            d.mkdir()
            for k, img in enumerate(frames):
                cv2.imwrite(str(d / f"frame-{k:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            v = pathlib.Path(td) / "clip.mp4"
            encode_video_frames(d, v, FPS, video_encoder=RGBEncoderConfig(vcodec="libsvtav1"), overwrite=True)
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

    def frame0(i):
        r = first.loc[i]
        return {"observation.state": np.asarray(r["observation.state"], np.float32),
                OBS_ENV_STATE: np.asarray(r["observation.environment_state"], np.float32),
                "observation.images.base": frame_rgb(i, "base", 0), "observation.images.claw": frame_rgb(i, "claw", 0)}

    H, P = [frame0(i) for i in holds], [frame0(i) for i in picks]
    raw = cv2.cvtColor(cv2.imread(str(a.raw_claw))[:, 640:1280], cv2.COLOR_BGR2RGB)
    d_h = [np.abs(f["observation.images.claw"].astype(np.int16) - raw.astype(np.int16)).mean() for f in H]
    d_p = [np.abs(f["observation.images.claw"].astype(np.int16) - raw.astype(np.int16)).mean() for f in P]
    still2 = first_of_clip([raw] * 2)
    d_s = [np.abs(f["observation.images.claw"].astype(np.int16) - still2.astype(np.int16)).mean() for f in H]
    print(f"checkpoint {a.step // 1000}k; {len(H)} hold episodes, {len(P)} picking episodes (start frames)")
    print(f"recorded claw start picture vs the simulator's raw one, mean |difference| of 255: holds {q(d_h):.2f}, picking {q(d_p):.2f}")
    print(f"recorded HOLD claw start picture vs the raw picture encoded as 2 copies: {q(d_s):.3f} (0 = a hold's recorded start frame is "
          "exactly a still clip of the raw picture)")
    pics = {"its own recorded picture (training input)": None,
            "the simulator's raw picture (tests so far)": raw,
            "raw picture through the encoder, 1 frame": first_of_clip([raw]),
            "raw picture through the encoder, 2 copies": still2,
            "raw picture through the encoder, 17 copies": first_of_clip([raw] * 17),
            "a hold episode's recorded picture": H[-1]["observation.images.claw"],
            "a picking episode's recorded picture": P[-1]["observation.images.claw"]}

    def depart(f, img):
        p = plan(f if img is None else {**f, "observation.images.claw": img})
        return float(np.abs(p[:, :5] - np.asarray(f["observation.state"], float)[:5]).max())

    print(f"\nLargest departure of the 100-command plan from the start pose (arm joints, norm units), median; and how many plans go "
          f"(more than {MOVES:.0f} units)")
    print(f"   claw start picture                              {'HOLD episodes (should stay)':34s} PICKING episodes (should go)")
    for name, img in pics.items():
        dh = [depart(f, img) for f in (H[:-1] if name.startswith("a hold") else H)]
        dp = [depart(f, img) for f in (P[:-1] if name.startswith("a picking") else P)]
        print(f"   {name:46s} {q(dh):6.1f} units, {int((np.array(dh) > MOVES).sum()):2d}/{len(dh)} go          "
              f"{q(dp):6.1f} units, {int((np.array(dp) > MOVES).sum()):2d}/{len(dp)} go")


if __name__ == "__main__":
    main()
