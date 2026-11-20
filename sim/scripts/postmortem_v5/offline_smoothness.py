#!/usr/bin/env python3
"""Post-mortem 2026-11-19: is the policy's COMMAND jittery, and does it jump where one chunk hands over to the next?

    conda activate lerobot && python offline_smoothness.py [--step 100000] [--episodes 50]

No simulator. On clean training episodes the policy predicts a 100-step chunk from the recorded frames at 0, 3.33, 6.67
and 10 s. Compared with the teacher's own commands over the same frames:
  - how much the commanded jaw point changes speed from one step to the next (mm/s2 at 30 steps per second)
  - the jump between the last step of one chunk and the first step of the next
⚠️ Each chunk starts from the TEACHER's state here. In a rollout the next chunk starts from wherever the policy got to,
so real hand-over jumps can be larger than this.
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
    em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
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

    acc_p, acc_t, jump_p, jump_t, jq_p, jq_t, lead = [], [], [], [], [], [], []
    for seed in seeds:
        i = by_seed[seed]
        r, g = kept[i], eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        prev = None
        for k in (0, 100, 200, 300):
            if k + 100 > len(ac):
                break
            frame = {"observation.state": st[k], OBS_ENV_STATE: env[k], "observation.images.base": frame_rgb(i, "base", k),
                     "observation.images.claw": frame_rgb(i, "claw", k)}
            with torch.inference_mode():
                x = prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco")
                pred = post(policy.predict_action_chunk(pre(x))[0]).cpu().numpy().astype(float)
            pj = np.array([jaw(pred[h], grip_pt) for h in range(100)])
            tj = np.array([jaw(ac[k + h].astype(float), grip_pt) for h in range(100)])
            for P, out in ((pj, acc_p), (tj, acc_t)):
                v = np.linalg.norm(np.diff(P, axis=0), axis=1) * FPS
                out.append(np.abs(np.diff(v)).max() * FPS)
            jq_p.append(np.abs(np.diff(pred[:, :5], axis=0, n=2)).max())     # largest second difference of a joint, units
            jq_t.append(np.abs(np.diff(ac[k:k + 100, :5].astype(float), axis=0, n=2)).max())
            # the first command of the chunk vs where the arm is now (the teacher's command leads the arm by this much)
            lead.append((float(np.linalg.norm(jaw(pred[0], grip_pt) - jaw(st[k].astype(float), grip_pt))),
                         float(np.linalg.norm(jaw(ac[k].astype(float), grip_pt) - jaw(st[k].astype(float), grip_pt)))))
            if prev is not None:
                jump_p.append(float(np.linalg.norm(pj[0] - prev[0])))
                jump_t.append(float(np.linalg.norm(tj[0] - prev[1])))
            prev = (pj[-1], tj[-1])
    out = dict(acc_p=acc_p, acc_t=acc_t, jump_p=jump_p, jump_t=jump_t, jq_p=jq_p, jq_t=jq_t, lead=lead)
    json.dump(out, open(OUT / f"offline_smoothness_{a.step // 1000}k.json", "w"))
    print(f"checkpoint {a.step // 1000}k, {len(seeds)} clean training episodes, chunks from 0 / 3.33 / 6.67 / 10 s")
    print(f"   largest step-to-step change of commanded jaw speed inside a chunk: policy {q(acc_p, 50):.0f} mm/s2 "
          f"(90 %: {q(acc_p, 90):.0f}); teacher over the same frames {q(acc_t, 50):.0f} (90 %: {q(acc_t, 90):.0f})")
    print(f"   largest second difference of any arm joint inside a chunk: policy {q(jq_p, 50):.2f} units (90 %: {q(jq_p, 90):.2f}); "
          f"teacher {q(jq_t, 50):.2f} (90 %: {q(jq_t, 90):.2f})")
    print(f"   jump of the commanded jaw point where one chunk hands over to the next: policy {q(jump_p, 50):.1f} mm "
          f"(90 %: {q(jump_p, 90):.1f}); teacher's ordinary step there {q(jump_t, 50):.1f} mm (90 %: {q(jump_t, 90):.1f})")
    L = np.array(lead)
    print(f"   first command of a chunk vs where the arm is now: policy {q(L[:, 0], 50):.1f} mm (90 %: {q(L[:, 0], 90):.1f}); "
          f"teacher {q(L[:, 1], 50):.1f} mm (90 %: {q(L[:, 1], 90):.1f})")


if __name__ == "__main__":
    main()
