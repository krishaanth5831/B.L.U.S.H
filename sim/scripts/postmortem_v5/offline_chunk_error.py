#!/usr/bin/env python3
"""Post-mortem 2026-11-19: on its own TRAINING frames, how far is the policy's predicted chunk from the teacher's?

    conda activate lerobot && python offline_chunk_error.py [--steps 20000 60000 100000] [--episodes 50]

No simulator. For clean training episodes (train_clean_50.jsonl, the same scenes the rollout test train_clean uses)
the recorded observation at a few frames goes through the policy exactly as in eval_truss.py; the predicted 100-step
chunk is compared with the 100 actions the teacher really sent from that frame on. Both are turned into the jaw's cut
point with forward kinematics, so the error is in mm where the blade would be commanded.

⚠️ This is not a rollout: the policy always starts from the teacher's own state, so it says how well the mapping was
LEARNED (underfitting), not how errors build up when the policy drives. Rollouts are the test of that.
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
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."   # generate_truss.TASK
FPS, CHUNK = 30, 100
JOINTS = arm_map.JOINTS


def q(a, p):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[20000, 60000, 100000])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--run", default="act_teacher_v5")
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
        fr = decode_video_frames(path, [ts], 1e-3, backend="pyav", return_uint8=True)
        return fr[0].permute(1, 2, 0).numpy()                       # H, W, 3 uint8, as the sim's cameras give it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(REPO, root=HF_LEROBOT_HOME / REPO)
    out = {}
    for step in a.steps:
        ck = ROOT / "outputs" / "train" / a.run / "checkpoints" / f"{step:06d}" / "pretrained_model"
        policy, pre, post, pcfg = et.load_policy(ck, device, meta, dict(temporal_ensemble_coeff=None, n_action_steps=100))
        recs = []
        for seed in seeds:
            i = by_seed[seed]
            r, g = kept[i], eps[i]
            st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
            ac = np.stack(g["action"].to_numpy()).astype(np.float32)
            env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
            grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
            opened = np.maximum.accumulate(ac[:, 5] > 20)
            kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])   # the teacher commands the jaw shut
            p_cut_cmd = jaw(ac[kc], grip_pt)                        # where the teacher's command has the jaw then
            for name, k in (("0.00s", 0), ("1.67s", 50), ("3.33s", 100), ("close-1s", kc - 30), ("close-0.33s", kc - 10),
                            ("6.67s", 200), ("10.0s", 300)):
                if k < 0 or k >= len(st) - 5:
                    continue
                frame = {"observation.state": st[k], OBS_ENV_STATE: env[k],
                         "observation.images.base": frame_rgb(i, "base", k),
                         "observation.images.claw": frame_rgb(i, "claw", k)}
                with torch.inference_mode():
                    x = prepare_observation_for_inference(frame, device, TASK, "so101_mujoco")
                    chunk = policy.predict_action_chunk(pre(x))[0]            # (100, 6) normalised
                    pred = post(chunk).cpu().numpy().astype(float)           # norm joint units, as sent to the arm
                n = min(CHUNK, len(ac) - k)
                tea = ac[k:k + n].astype(float)
                dj = np.abs(pred[:n] - tea)
                mm = np.array([np.linalg.norm(jaw(pred[h], grip_pt) - jaw(tea[h], grip_pt)) for h in range(n)])
                rec = dict(seed=seed, at=name, k=k, n=n, joint_abs=dj.mean(0).round(2).tolist(),
                           mm_h1=float(mm[0]), mm_h10=float(mm[min(9, n - 1)]), mm_h30=float(mm[min(29, n - 1)]),
                           mm_h60=float(mm[min(59, n - 1)]), mm_h100=float(mm[n - 1]), mm_mean=float(mm.mean()),
                           move_pred=float(np.linalg.norm(jaw(pred[n - 1], grip_pt) - jaw(pred[0], grip_pt))),
                           move_tea=float(np.linalg.norm(jaw(tea[n - 1], grip_pt) - jaw(tea[0], grip_pt))))
                if k <= kc < k + n:                                 # the close falls inside this chunk
                    h = kc - k
                    pj = jaw(pred[h], grip_pt)
                    pc = np.flatnonzero(pred[:n, 5] < 10)
                    rec.update(close_in_chunk=True, h_close=h, mm_at_close=float(np.linalg.norm(pj - p_cut_cmd)),
                               off_at_close=(pj - p_cut_cmd).round(1).tolist(),
                               grip_pred_at_close=float(pred[h, 5]),
                               close_dt=(float(pc[0] - h) / FPS if len(pc) else None))
                    # the best the predicted path does, whatever the time: its nearest point to the teacher's cut spot
                    rec["mm_nearest_any_time"] = float(min(np.linalg.norm(jaw(pred[j], grip_pt) - p_cut_cmd)
                                                           for j in range(n)))
                recs.append(rec)
        out[step] = recs
        print(f"\n=== checkpoint {step // 1000}k, {len(seeds)} clean training episodes "
              f"(jaw error = predicted command vs the teacher's command, mm; median and 90 %)")
        print("   from frame    n   after 1 step  10 steps   30 steps   60 steps  100 steps | chunk travel pred/teacher")
        for name in ("0.00s", "1.67s", "3.33s", "close-1s", "close-0.33s", "6.67s", "10.0s"):
            R = [x for x in recs if x["at"] == name]
            if not R:
                continue
            cells = "  ".join(f"{q([x[c] for x in R], 50):5.1f} ({q([x[c] for x in R], 90):5.1f})"
                              for c in ("mm_h1", "mm_h10", "mm_h30", "mm_h60", "mm_h100"))
            print(f"   {name:11s} {len(R):3d}   {cells} | {q([x['move_pred'] for x in R], 50):5.0f} / "
                  f"{q([x['move_tea'] for x in R], 50):5.0f}")
        for name in ("3.33s", "close-1s", "close-0.33s"):
            R = [x for x in recs if x["at"] == name and x.get("close_in_chunk")]
            if not R:
                continue
            dt = [x["close_dt"] for x in R if x["close_dt"] is not None]
            print(f"   close inside the chunk from {name} ({len(R)} eps): jaw is {q([x['mm_at_close'] for x in R], 50):.1f} mm "
                  f"(90 %: {q([x['mm_at_close'] for x in R], 90):.1f}) from the teacher's cut spot at the teacher's close frame; "
                  f"nearest the predicted path gets: {q([x['mm_nearest_any_time'] for x in R], 50):.1f} "
                  f"({q([x['mm_nearest_any_time'] for x in R], 90):.1f}); predicts a close in {len(dt)}/{len(R)}, "
                  f"timing off by {q(np.abs(dt), 50):.2f} s (90 %: {q(np.abs(dt), 90):.2f})")
        ja = np.array([x["joint_abs"] for x in recs])
        print("   mean |joint error| over a chunk, norm units (" + ", ".join(JOINTS) + "): "
              + ", ".join(f"{v:.1f}" for v in ja.mean(0)))
        del policy
        torch.cuda.empty_cache()
    json.dump(out, open(OUT / "offline_chunk_error.json", "w"))
    print(f"\nsaved {OUT / 'offline_chunk_error.json'}")


if __name__ == "__main__":
    main()
