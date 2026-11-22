#!/usr/bin/env python3
"""Post-mortem 2026-11-20: is the policy's miss one constant offset, and which way does it point? No simulator.

    conda activate lerobot && python offline_offset_shape.py [--step 100000] [--episodes 50]

On clean training episodes the recorded frame at 3.33 s (the policy's second look) goes through the policy; the
predicted commands are compared with the teacher's, step by step, as jaw points (forward kinematics), so both have
the same reference point and the same clock.

A. Constant or wobbling: the error vectors of the steps up to the teacher's close are split into their mean (one
   offset for the whole plan) and what is left (the plan's shape differing from the teacher's).
B. Direction at the teacher's close frame: along the teacher's own travel to the cut spot (minus = the policy's plan
   stops short of it, plus = overshoots) and across it; and in world axes.
C. Is the offset a pull toward the dataset's AVERAGE command (what an under-fitted network falls back on)? Per joint:
   the signed error at the teacher's close, next to how far the average command over all 383,877 frames lies from
   that episode's cut pose. Same sign and a similar ratio on every joint = the same pull on all of them.
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "notes" / "2026-11-19_v5_postmortem"   # generated files (git-ignored), not the repo tree
sys.path.insert(0, str(ROOT / "sim"))
import arm_map  # noqa: E402
import expert  # noqa: E402
import truss_ik  # noqa: E402

REPO = "krish5831/blush_sim_teacher_v5"
SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
LOOK2 = 100                 # frame of the policy's second look with 100-command chunks (3.33 s)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE
    sys.argv = sys.argv[:1]
    import eval_truss as et

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/data/*/*.parquet"))])
    df = df.sort_values(["episode_index", "frame_index"])
    eps = {int(k): g for k, g in df.groupby("episode_index")}
    em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/meta/episodes/*/*.parquet"))])
    em = em.set_index("episode_index")
    rows = []
    for f in LOGS:
        rows += [json.loads(ln) for ln in open(f) if ln.strip()]
    kept = [r for r in rows if r.get("kept") is True]
    by_seed = {r["seed"]: i for i, r in enumerate(kept)}
    seeds = [json.loads(ln)["seed"] for ln in open(HERE / "train_clean_50.jsonl") if ln.strip()][:a.episodes]

    ik, fm = expert.ArmIK(n_samples=10), arm_map.FollowerMap()

    def jaw(norm6, grip_pt):
        o, R = ik.fk(fm.norm_to_rad(norm6)[:5])
        return (o + R @ grip_pt) * 1000

    def frame_rgb(ep, cam, k):
        key = f"videos/observation.images.{cam}"
        path = (f"{SNAP}/videos/observation.images.{cam}/chunk-{int(em.loc[ep, key + '/chunk_index']):03d}/"
                f"file-{int(em.loc[ep, key + '/file_index']):03d}.mp4")
        ts = float(em.loc[ep, key + "/from_timestamp"]) + k / FPS
        return decode_video_frames(path, [ts], 1e-3, backend="pyav", return_uint8=True)[0].permute(1, 2, 0).numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(REPO, root=HF_LEROBOT_HOME / REPO)
    ck = ROOT / "outputs" / "train" / "act_teacher_v5" / "checkpoints" / f"{a.step:06d}" / "pretrained_model"
    policy, pre, post, _ = et.load_policy(ck, device, meta, dict(temporal_ensemble_coeff=None, n_action_steps=100))

    def predict(frame):
        with torch.inference_mode():
            # ⚠️ prepare_observation_for_inference writes tensors back into the dict it is given: hand it a copy
            x = prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco")
            return post(policy.predict_action_chunk(pre(x))[0]).cpu().numpy().astype(float)

    abar = np.stack(df["action"].to_numpy()).astype(float).mean(0)     # the average command over every frame
    A, B, C = [], [], []
    for seed in seeds:
        i = by_seed[seed]
        r, g = kept[i], eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])
        k = LOOK2
        if kc - k < 8:
            continue
        f0 = {"observation.state": st[k], OBS_ENV_STATE: env[k], "observation.images.base": frame_rgb(i, "base", k),
              "observation.images.claw": frame_rgb(i, "claw", k)}
        p = predict(f0)
        n = kc - k + 1                                            # steps up to and including the teacher's close
        P = np.array([jaw(p[j], grip_pt) for j in range(n)])
        T = np.array([jaw(ac[k + j], grip_pt) for j in range(n)])
        e = P - T
        mean = e.mean(0)
        A.append((float(np.linalg.norm(mean)), float(np.sqrt(((e - mean) ** 2).sum(1).mean())),
                  float(np.linalg.norm(e, axis=1).mean())))
        travel = T[-1] - T[0]                                     # the teacher's own way to the cut spot
        u = travel / np.linalg.norm(travel)
        ec = e[-1]
        B.append((float(ec @ u), float(np.linalg.norm(ec - (ec @ u) * u)), float(np.linalg.norm(travel)), *ec,
                  float(np.linalg.norm(P[-1] - P[0]))))
        ej = p[n - 1, :5] - ac[kc, :5]                            # signed joint error at the teacher's close, units
        dj = abar[:5] - ac[kc, :5]                                # from this cut pose toward the average command
        pull = jaw(np.r_[ac[kc, :5] + 0.05 * dj, ac[kc, 5]], grip_pt) - T[-1]   # where a 5 % pull would move the jaw
        C.append((*ej, *dj, float(ec @ pull / (np.linalg.norm(ec) * np.linalg.norm(pull) + 1e-9)),
                  float(ec @ pull / (pull @ pull + 1e-9)) * 0.05))
    A, B, C = np.array(A), np.array(B), np.array(C)
    print(f"checkpoint {a.step // 1000}k, {len(A)} clean training episodes, plan made at {LOOK2 / FPS:.2f} s, steps up to "
          f"the teacher's close ({q(B[:, 2], 50):.0f} mm of travel left for the teacher)")
    print(f"A. mean error over those steps {q(A[:, 2], 50):.1f} mm; of which ONE constant offset {q(A[:, 0], 50):.1f} mm "
          f"(25-75 %: {q(A[:, 0], 25):.1f}-{q(A[:, 0], 75):.1f}); what is left after taking it out "
          f"{q(A[:, 1], 50):.1f} mm (25-75 %: {q(A[:, 1], 25):.1f}-{q(A[:, 1], 75):.1f})")
    print(f"B. at the teacher's close frame: along the teacher's travel {q(B[:, 0], 50):+.1f} mm (25-75 %: "
          f"{q(B[:, 0], 25):+.1f} to {q(B[:, 0], 75):+.1f}; minus = short of the cut spot), short in "
          f"{int((B[:, 0] < 0).sum())}/{len(B)}; across it {q(B[:, 1], 50):.1f} mm (25-75 %: {q(B[:, 1], 25):.1f}-"
          f"{q(B[:, 1], 75):.1f})")
    print(f"   the policy's plan travels {q(B[:, 6], 50):.0f} mm in those steps, the teacher's {q(B[:, 2], 50):.0f} mm")
    print(f"   in world axes (x along the row, y toward the row, z up), median: {q(B[:, 3], 50):+.1f}, "
          f"{q(B[:, 4], 50):+.1f}, {q(B[:, 5], 50):+.1f} mm; share of episodes low {np.mean(B[:, 5] < 0):.2f}, "
          f"short of the row {np.mean(B[:, 4] < 0):.2f}")
    print("C. per joint at the teacher's close (units): signed error of the policy's command | from the cut pose to the "
          "dataset's average command | error as a share of that | episodes with the same sign")
    shares = []
    for j, name in enumerate(JOINTS):
        e, d = C[:, j], C[:, 5 + j]
        b = float(e @ d / (d @ d))                               # least squares through zero: e = b * d
        r = float(np.corrcoef(e, d)[0, 1])
        shares.append(b)
        print(f"   {name:14s} {q(e, 50):+5.1f} ({q(e, 25):+5.1f} to {q(e, 75):+5.1f})   |   {q(d, 50):+6.1f}   |   "
              f"{b:+.3f} (r {r:+.2f})   |   {int((np.sign(e) == np.sign(d)).sum())}/{len(e)}")
    print(f"   at the jaw: the miss points the same way as a pull toward the average command with cosine "
          f"{q(C[:, 10], 50):+.2f} (25-75 %: {q(C[:, 10], 25):+.2f} to {q(C[:, 10], 75):+.2f}; +1 = exactly that way, 0 = unrelated); "
          f"size of the pull that fits best {q(C[:, 11], 50):.3f} of the way")
    json.dump(dict(mean_err=q(A[:, 2], 50), constant=q(A[:, 0], 50), rest=q(A[:, 1], 50), along=q(B[:, 0], 50),
                   across=q(B[:, 1], 50), short=int((B[:, 0] < 0).sum()), n=len(B),
                   world=[q(B[:, 3], 50), q(B[:, 4], 50), q(B[:, 5], 50)], pull_share_per_joint=shares,
                   pull_cosine=q(C[:, 10], 50), abar=[float(x) for x in abar]),
              open(OUT / f"offline_offset_shape_{a.step // 1000}k.json", "w"), indent=1)


if __name__ == "__main__":
    main()
