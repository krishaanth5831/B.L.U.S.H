#!/usr/bin/env python3
"""Post-mortem 2026-11-20: does the policy's FIRST plan finish the turn of the shoulder? No simulator.

    conda activate lerobot && python offline_first_plan_turn.py [--step 100000] [--episodes 50] [--samples 16]

Why: in rollouts the arm still has about 6 deg of turn left when the second plan is made (3.33 s), and the second plan
mostly holds the shoulder where it is (turn_reach_height.py). This asks the same of the plan itself, on frames the
policy was TRAINED on: the recorded start frame of a clean training episode goes through the policy, and the
shoulder_pan command of its 100-step plan is compared, step by step, with the teacher's command in that episode.

"Turn left" at a step = (the teacher's shoulder command when it closes the jaw) - (the command at that step), in
degrees; + = not yet turned far enough. Printed for the teacher's own commands, for the policy as it runs (A) and for
the mean of --samples dropout-on answers (B, the condition the network is trained under).
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

RUNS = ROOT / "outputs" / "diag" / "runs"                       # traced rollouts (eval_truss.py --trace)

REPO = "krish5831/blush_sim_teacher_v5"
SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
UNIT_DEG = 1.116            # deg of shoulder_pan per norm unit (joint range +-111.6 deg = +-100 units; model jnt_range)
STEPS = (0, 29, 59, 89, 99)  # commands of the plan to look at: its first, after 1, 2 and 3 s, and its last (3.33 s)


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--samples", type=int, default=16)
    a = ap.parse_args()
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES
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

    with torch.inference_mode():
        mean_t = post(torch.zeros(1, 6, device=device))[0].to(device).float()          # un-normalising: x * std + mean
        std_t = post(torch.ones(1, 6, device=device))[0].to(device).float() - mean_t

    def predict(frame, dropout):
        """(100, 6) commands in joint units: the plan as it runs, or the mean of --samples dropout-on answers.
        ⚠️ The model is called directly: policy.predict_action_chunk() switches every module back to eval mode, which
        silently turned the dropout-on column into a copy of the other one (first version of this script)."""
        k = a.samples if dropout else 1
        x = pre(prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco"))
        b = {key: (v.repeat(k, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v) for key, v in x.items()}
        b[OBS_IMAGES] = [b[key] for key in policy.config.image_features]
        policy.eval()
        if dropout:                                   # only the dropout switches (offline_train_vs_inference.py)
            for m in policy.model.modules():
                if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)):
                    m.train()
        with torch.inference_mode():
            out = policy.model(b)[0]
        policy.eval()
        return (out * std_t + mean_t).cpu().numpy().astype(float).mean(0)

    ik, fm = expert.ArmIK(n_samples=10), arm_map.FollowerMap()

    def jaw(norm6, grip_pt):
        o, R = ik.fk(fm.norm_to_rad(norm6)[:5])
        return (o + R @ grip_pt) * 1000

    def rollout_jaw(run):
        """seed -> {plan step: jaw point (mm) the traced rollout really had at that step's clock time}"""
        out = {}
        f = RUNS / run / "trace.jsonl"
        if not f.exists():
            return out
        want = {round((j + 1) / FPS, 2): j for j in STEPS}
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                j = want.get(r["t"])
                if j is not None:
                    out.setdefault(r["seed"], {})[j] = np.array(r["jaw_mm"])
        return out

    roll_p, roll_t = rollout_jaw("train_clean"), rollout_jaw("train_teacher")
    gap_p, gap_t = [], []

    T, A, B, need, close_k = [], [], [], [], []
    for seed in seeds:
        i = by_seed[seed]
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])       # the teacher's close frame
        f0 = {"observation.state": st[0], OBS_ENV_STATE: env[0], "observation.images.base": frame_rgb(i, "base", 0),
              "observation.images.claw": frame_rgb(i, "claw", 0)}
        pa, pb = predict(f0, False), predict(f0, True)
        goal = float(ac[kc, 0])                                      # shoulder command when the teacher closes
        T.append([(goal - float(ac[j, 0])) * UNIT_DEG for j in STEPS])
        A.append([(goal - float(pa[j, 0])) * UNIT_DEG for j in STEPS])
        B.append([(goal - float(pb[j, 0])) * UNIT_DEG for j in STEPS])
        need.append((goal - float(st[0, 0])) * UNIT_DEG)
        close_k.append(kc)
        grip_pt = truss_ik.cut_point_in_gripper(kept[i]["plan"]["stalk_mm"] / 2000)
        if seed in roll_p and all(j in roll_p[seed] for j in STEPS):
            gap_p.append([float(np.linalg.norm(jaw(pa[j], grip_pt) - roll_p[seed][j])) for j in STEPS])
        if seed in roll_t and all(j in roll_t[seed] for j in STEPS):
            gap_t.append([float(np.linalg.norm(jaw(ac[j], grip_pt) - roll_t[seed][j])) for j in STEPS])
    T, A, B, need = np.array(T), np.array(A), np.array(B), np.array(need)
    print(f"checkpoint {a.step // 1000}k, {len(T)} clean training episodes, plan made from the start frame. The shoulder has to turn "
          f"{q(need, 50):.0f} deg (range {need.min():.0f} to {need.max():.0f}); the teacher closes the jaw at {q(close_k, 50) / FPS:.2f} s.")
    print("Turn left in the shoulder COMMAND, deg, median (25-75 %):")
    print("   command of the plan      the teacher's          A policy as it runs      B mean of dropout-on answers")
    for n, j in enumerate(STEPS):
        print(f"   {j + 1:3d} ({(j + 1) / FPS:4.2f} s)        " + "   ".join(
            f"{q(X[:, n], 50):+6.2f} ({q(X[:, n], 25):+6.2f} to {q(X[:, n], 75):+6.2f})" for X in (T, A, B)))
    last = len(STEPS) - 1
    for name, X in (("A policy as it runs", A), ("B mean of dropout-on answers", B)):
        d = X[:, last] - T[:, last]
        print(f"   last command, {name} minus the teacher's: {q(d, 50):+.2f} deg (25-75 %: {q(d, 25):+.2f} to {q(d, 75):+.2f}); "
              f"more than 2 deg short of the teacher in {int((d > 2).sum())}/{len(d)}, more than 2 deg past it in {int((d < -2).sum())}/{len(d)}")
    for lo, hi in ((40, 60), (60, 80), (80, 110), (110, 150)):
        m = (need >= lo) & (need < hi)
        if m.sum():
            print(f"   turn needed {lo:3d}-{hi:3d} deg: n {int(m.sum()):2d}, turn left in the last command: teacher {q(T[m, last], 50):+6.2f}, "
                  f"A {q(A[m, last], 50):+6.2f}, B {q(B[m, last], 50):+6.2f}")
    if gap_p and gap_t:
        gp, gt_ = np.array(gap_p), np.array(gap_t)
        print("Is that plan what the rollout on the same scene really did? Distance between the jaw point the COMMAND asks for "
              "(forward kinematics) and the jaw point the traced rollout had at that moment, mm, median (90 %):")
        print("   command of the plan      teacher: its recorded commands vs its rollout (train_teacher)      "
              "policy: this offline plan vs its rollout (train_clean)")
        for n, j in enumerate(STEPS):
            print(f"   {j + 1:3d} ({(j + 1) / FPS:4.2f} s)        {q(gt_[:, n], 50):6.1f} ({q(gt_[:, n], 90):6.1f})   n {len(gt_):2d}"
                  f"                                   {q(gp[:, n], 50):6.1f} ({q(gp[:, n], 90):6.1f})   n {len(gp):2d}")
    out = OUT / f"offline_first_plan_turn_{a.step // 1000}k.json"
    out.write_text(json.dumps(dict(steps=[j + 1 for j in STEPS], teacher=T.tolist(), policy=A.tolist(), dropout_mean=B.tolist(),
                                   need=need.tolist())))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
