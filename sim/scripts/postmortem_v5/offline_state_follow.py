#!/usr/bin/env python3
"""Post-mortem 2026-11-20: does the policy use its own joint readings? No simulator.

    conda activate lerobot && python offline_state_follow.py [--step 100000] [--episodes 50]

Why: on training frames the policy's FIRST command is already about 10 mm from the teacher's, although the arm's
joint positions are an input and the teacher's first command is 3 mm from where the arm is. So either the policy
reads the joints and adds its own error, or it hardly reads them and judges where the arm is from the pictures.

Test: for clean training episodes, at three looks (0 s, 3.33 s, 0.33 s before the teacher's close), the recorded
frame goes through the policy unchanged and then with the READING of one joint raised by BUMP units while the pictures
stay as they were (`--bump -5` lowers it instead). A policy that steers by the joint readings moves that joint's command by about the same amount
(follow = 1); one that steers by the pictures does not move it (follow = 0). Also: every arm joint's reading raised
together, compared in mm at the jaw.
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
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STEPS = (0, 9, 29)          # which commands of the chunk to compare: the 1st, 10th and 30th


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--bump", type=float, default=5.0,
                    help="norm units added to one joint's reading (my choice: above the 2-4 unit command error); "
                         "negative to lower it, which keeps a joint at its end stop inside the range it was trained on")
    a = ap.parse_args()
    BUMP = a.bump
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

    looks = ("0s", "3.33s", "close-0.33s")
    follow = {lk: {j: {s: [] for s in STEPS} for j in JOINTS} for lk in looks}     # own joint, units per unit
    cross = {lk: {j: [] for j in JOINTS} for lk in looks}       # largest move of any OTHER joint's command, units
    alls = {lk: [] for lk in looks}                              # all joints together: (moved mm, wanted mm, along)
    lead = {lk: [] for lk in looks}                              # |first command - reading| per joint, policy vs teacher
    for seed in seeds:
        i = by_seed[seed]
        r, g = kept[i], eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])
        for lk, k in zip(looks, (0, 100, kc - 10)):
            if lk == "3.33s" and k >= kc:
                continue
            f0 = {"observation.state": st[k], OBS_ENV_STATE: env[k], "observation.images.base": frame_rgb(i, "base", k),
                  "observation.images.claw": frame_rgb(i, "claw", k)}
            p0 = predict(f0)
            lead[lk].append((np.abs(p0[0, :5] - st[k, :5]), np.abs(ac[k, :5] - st[k, :5])))
            for j, name in enumerate(JOINTS):
                bump = np.zeros(6, np.float32)
                bump[j] = BUMP
                p1 = predict({**f0, "observation.state": st[k] + bump})
                for s in STEPS:
                    follow[lk][name][s].append((p1[s, j] - p0[s, j]) / BUMP)
                others = [x for x in range(5) if x != j]
                cross[lk][name].append(float(np.abs(p1[0, others] - p0[0, others]).max()))
            bump = (np.array([3, 3, 3, 3, 3, 0]) * np.sign(BUMP)).astype(np.float32)   # float32: the policy's dtype
            p1 = predict({**f0, "observation.state": st[k] + bump})
            want = jaw(st[k] + bump, grip_pt) - jaw(st[k], grip_pt)           # where the jaw point would be if the
            got = jaw(p1[0], grip_pt) - jaw(p0[0], grip_pt)                   # arm really stood 3 units further
            alls[lk].append((float(np.linalg.norm(got)), float(np.linalg.norm(want)),
                             float(got @ want / (want @ want))))

    out = {}
    print(f"checkpoint {a.step // 1000}k, {len(seeds)} clean training episodes. One joint's READING raised by {BUMP:.0f} "
          f"units, pictures unchanged.\nfollow = how much that joint's COMMAND rises, per unit (1 = steers by the "
          f"reading, 0 = ignores it). Median (25-75 %).")
    for lk in looks:
        print(f"\n   look at {lk} ({len(alls[lk])} episodes)          1st command        10th command       30th command   "
              f"| other joints' 1st command moves by")
        for name in JOINTS:
            cells = []
            for s in STEPS:
                v = follow[lk][name][s]
                cells.append(f"{q(v, 50):5.2f} ({q(v, 25):5.2f} to {q(v, 75):5.2f})")
            print(f"      {name:14s}        " + "   ".join(cells) + f"   | {q(cross[lk][name], 50):4.1f} units")
            out[f"{lk}/{name}"] = {str(s + 1): q(follow[lk][name][s], 50) for s in STEPS}
        A = np.array(alls[lk])
        print(f"      all five readings {3 * np.sign(BUMP):+.0f} units: the 1st command's jaw point moves {q(A[:, 0], 50):.1f} mm; an arm really "
              f"standing there would be {q(A[:, 1], 50):.1f} mm away; share followed {q(A[:, 2], 50):.2f} "
              f"({q(A[:, 2], 25):.2f} to {q(A[:, 2], 75):.2f})")
        Lp = np.array([x[0] for x in lead[lk]])
        Lt = np.array([x[1] for x in lead[lk]])
        print("      |1st command - reading| per joint, units: policy "
              + ", ".join(f"{q(Lp[:, j], 50):.1f}" for j in range(5)) + "   teacher "
              + ", ".join(f"{q(Lt[:, j], 50):.1f}" for j in range(5)))
        out[f"{lk}/all"] = dict(moved_mm=q(A[:, 0], 50), wanted_mm=q(A[:, 1], 50), share=q(A[:, 2], 50),
                                lead_policy=[q(Lp[:, j], 50) for j in range(5)],
                                lead_teacher=[q(Lt[:, j], 50) for j in range(5)])
    json.dump(out, open(OUT / f"offline_state_follow_{a.step // 1000}k_bump{BUMP:+.0f}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
