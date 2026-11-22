#!/usr/bin/env python3
"""Post-mortem 2026-11-20: does the policy answer differently in TRAINING mode and in INFERENCE mode? No simulator.

    conda activate lerobot && python offline_train_vs_inference.py [--run act_teacher_v5] [--step 100000]
                                                                    [--episodes 50] [--samples 16]

Why: offline_offset_shape.py found the policy's commands a few percent "too far out" from the dataset's average
command on every joint, the same way in every episode. One thing that can do that: while it trains, a random 10 % of
the network's inner signals are switched off at every step ("dropout") and a random code is fed in next to the
pictures (the "latent" of ACT's VAE); when it runs, nothing is switched off and the code is zero. If the network's
answers are tuned to the noisy training condition, the clean inference condition can come out scaled differently.

Test, on recorded training frames of dataset v5 (plans made at several moments of the episode):
  A  inference mode: nothing switched off, code zero            (what every rollout uses)
  B  dropout ON, code zero                (mean of --samples answers)
  C  dropout off, code RANDOM             (mean of --samples answers; only at the 3.33 s look)
  D  dropout ON, code RANDOM              (the training condition; only at the 3.33 s look)
Reported:
  - "stretch": how far the commands lie from the pivot (the output layer's own offset, about the dataset's average
    command) relative to the teacher's commands, and A relative to B. 1.00 = the same; 1.04 = 4 % further out.
  - the jaw-point error at the teacher's close frame (mm), for the two looks that reach it.
  - what the inference answer's error would be with its stretch taken out (factor found on one half of the
    episodes, applied to the other half).
  - the signal that enters the output layer: its size in A against B (is the whole signal bigger at inference?).
  - at the 0 s look: the first command against the start pose the arm stands in.
--dataset picks whose frames go in; when they are not the run's own training data only A against B is meaningful.
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

CACHE = pathlib.Path.home() / ".cache"
# run -> (dataset it was trained on, its generator logs in ~/.cache/blush_sim/truss/); the one-position dataset was
# renamed on the Hub after it was recorded, its logs keep the old name
TRAINED_ON = {"act_teacher_v5": ("blush_sim_teacher_v5", "blush_sim_teacher_v5_p*.jsonl"),
              "act_teacher_onelayout_v1": ("blush_sim_teacher_v3_onelayout", "blush_sim_teacher_onelayout_v1_*.jsonl")}
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LOOKS = (("0 s", 0), ("1.67 s", 50), ("3.33 s", 100), ("close-0.33 s", None), ("6.67 s", 200), ("10 s", 300))
# ^ frames at which a plan is made; 0, 100, 200, 300 are the policy's own looks with 100-command chunks


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_teacher_v5", help="folder under outputs/train/")
    ap.add_argument("--step", default="100000", help="checkpoint step, or 'last'")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--samples", type=int, default=16, help="answers averaged per frame in modes B, C, D (my choice)")
    ap.add_argument("--dataset", default="blush_sim_teacher_v5",
                    help="whose frames to feed in (must be in the lerobot hub cache); the teacher columns are only "
                         "printed when it is the run's own training data")
    a = ap.parse_args()
    REPO = f"krish5831/{a.dataset}"
    SNAP = sorted(glob.glob(str(CACHE / f"huggingface/lerobot/hub/datasets--krish5831--{a.dataset}/snapshots/*")))[-1]
    LOGS = sorted(glob.glob(str(CACHE / "blush_sim/truss" / {v[0]: v[1] for v in TRAINED_ON.values()}[a.dataset])))
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES
    sys.argv = sys.argv[:1]
    import eval_truss as et

    own = TRAINED_ON.get(a.run, ("",))[0] == a.dataset     # the teacher's commands are this run's training labels
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
    if a.dataset == "blush_sim_teacher_v5":
        seeds = [json.loads(ln)["seed"] for ln in open(HERE / "train_clean_50.jsonl") if ln.strip()][:a.episodes]
    else:                                     # the one-position dataset: every episode is a cut with noise
        seeds = [r["seed"] for r in kept if not r.get("hold")][:a.episodes]

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
    root = HF_LEROBOT_HOME / REPO
    meta = LeRobotDatasetMetadata(REPO, root=root if (root / "meta").exists() else SNAP)
    step = a.step if a.step == "last" else f"{int(a.step):06d}"
    ck = ROOT / "outputs" / "train" / a.run / "checkpoints" / step / "pretrained_model"
    policy, pre, post, _ = et.load_policy(ck, device, meta, dict(temporal_ensemble_coeff=None, n_action_steps=100))

    K = a.samples
    with torch.inference_mode():
        mean_t = post(torch.zeros(1, 6, device=device))[0].to(device).float()          # un-normalising: x * std + mean
        std_t = post(torch.ones(1, 6, device=device))[0].to(device).float() - mean_t
    bias = policy.model.action_head.bias.detach().float()
    pivot = (bias * std_t + mean_t).cpu().numpy().astype(float)       # where the output layer sits with a zero signal
    print(f"{a.run} checkpoint {a.step} on {len(seeds)} episodes of {a.dataset}: average command of its training data (units) "
          f"{np.round(mean_t.cpu().numpy(), 1)}; spread {np.round(std_t.cpu().numpy(), 1)}; pivot {np.round(pivot, 1)}")

    grabbed = {}
    policy.model.action_head.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("h", inp[0].detach()))

    def run(frame, k_samples, dropout, zrand):
        """(k_samples, 100, 6) commands in joint units and the (k_samples, 100, D) signal entering the output layer."""
        x = pre(prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco"))
        b = {key: (v.repeat(k_samples, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v) for key, v in x.items()}
        b[OBS_IMAGES] = [b[key] for key in policy.config.image_features]
        policy.eval()
        if dropout:                                   # only the dropout switches; the model itself stays in
            for m in policy.model.modules():          # inference mode, so the latent stays zero unless zrand
                if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)):
                    m.train()
        hook = None
        if zrand:
            hook = policy.model.encoder_latent_input_proj.register_forward_pre_hook(
                lambda m, inp: (torch.randn_like(inp[0]),))
        with torch.inference_mode():
            out = policy.model(b)[0]
        if hook is not None:
            hook.remove()
        policy.eval()
        return (out * std_t + mean_t).cpu().numpy().astype(float), grabbed["h"].float()

    def stretch(P, T, cols):
        """least squares through the pivot: P = s * T, pooled over the given joints."""
        P, T = np.concatenate(P)[:, cols].ravel(), np.concatenate(T)[:, cols].ravel()
        return float(P @ T / (T @ T))

    ARM, GRIP = [0, 1, 2, 3, 4], [5]
    looks = [lk for lk, _ in LOOKS]
    S = {lk: dict(PA=[], PB=[], T=[], errA=[], errB=[], oneB=[], hn=[], hc=[], half=[], last=[], first=[]) for lk in looks}
    CD = dict(PC=[], PD=[], errC=[], errD=[])
    for n_ep, seed in enumerate(seeds):
        i = by_seed[seed]
        r, g = kept[i], eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])
        for lk, k in LOOKS:
            k = kc - 10 if k is None else k
            if k < 0 or k >= len(ac) - 10:
                continue
            f0 = {"observation.state": st[k], OBS_ENV_STATE: env[k], "observation.images.base": frame_rgb(i, "base", k),
                  "observation.images.claw": frame_rgb(i, "claw", k)}
            n_all = min(100, len(ac) - k)                           # steps of the chunk that exist in the episode
            pa, ha = run(f0, 1, False, False)
            pb, hb = run(f0, K, True, False)
            d = S[lk]
            d["PA"].append(pa[0, :n_all] - pivot)
            d["PB"].append(pb.mean(0)[:n_all] - pivot)
            d["T"].append(ac[k:k + n_all] - pivot)
            hbm = hb.mean(0)                                         # (100, D)
            d["hn"].append(float((ha[0].norm(dim=-1) / hbm.norm(dim=-1)).median()))
            d["hc"].append(float(torch.nn.functional.cosine_similarity(ha[0], hbm, dim=-1).median()))
            d["first"].append((pa[0, 0] - st[k], pb.mean(0)[0] - st[k]))
            if own and 8 <= kc - k < 100:                           # the plan reaches the teacher's close frame
                tgt = jaw(ac[kc], grip_pt)
                j = kc - k
                d["errA"].append(float(np.linalg.norm(jaw(pa[0, j], grip_pt) - tgt)))
                d["errB"].append(float(np.linalg.norm(jaw(pb.mean(0)[j], grip_pt) - tgt)))
                d["oneB"].append(float(np.median([np.linalg.norm(jaw(o[j], grip_pt) - tgt) for o in pb])))
                d["half"].append(n_ep % 2)
                d["last"].append((grip_pt, float(ac[kc, 5]), tgt, pa[0, j] - pivot, ac[k:kc + 1] - pivot,
                                  pa[0, :j + 1] - pivot))
                if lk == "3.33 s":
                    pc, _ = run(f0, K, False, True)
                    pd_, _ = run(f0, K, True, True)
                    CD["errC"].append(float(np.linalg.norm(jaw(pc.mean(0)[j], grip_pt) - tgt)))
                    CD["errD"].append(float(np.linalg.norm(jaw(pd_.mean(0)[j], grip_pt) - tgt)))

    out_json = {}
    print("\nStretch of the commands about the pivot, all commands of the plan pooled (1.00 = none). Arm joints | gripper")
    print("   plan made at      A inference vs B dropout-on   " + ("A vs teacher    B vs teacher   " if own else "")
          + "| size of the signal entering the output layer, A / B (and how alike its direction is)")
    for lk in looks:
        d = S[lk]
        if not d["PA"]:
            continue
        ab, abg = stretch(d["PA"], d["PB"], ARM), stretch(d["PA"], d["PB"], GRIP)
        line = f"   {lk:14s} n {len(d['PA']):2d}   {ab:.3f} | {abg:.3f}              "
        rec = dict(n=len(d["PA"]), A_vs_B=ab, A_vs_B_gripper=abg, signal_ratio=q(d["hn"], 50), signal_cos=q(d["hc"], 50))
        if own:
            at, bt = stretch(d["PA"], d["T"], ARM), stretch(d["PB"], d["T"], ARM)
            line += f"   {at:.3f}           {bt:.3f}       "
            rec.update(A_vs_teacher=at, B_vs_teacher=bt,
                       A_vs_teacher_per_joint=[stretch(d["PA"], d["T"], [j]) for j in range(6)])
        print(line + f"| {q(d['hn'], 50):.3f} (cosine {q(d['hc'], 50):.3f})")
        out_json[lk] = rec

    F = np.array([x[0] for x in S["0 s"]["first"]]), np.array([x[1] for x in S["0 s"]["first"]])
    print("\nAt the 0 s look the arm stands in the start pose. First command minus the reading, median, units "
          "(pan, lift, elbow, wrist flex, wrist roll, gripper):")
    print("   A inference   " + "  ".join(f"{q(F[0][:, j], 50):+5.1f}" for j in range(6)))
    print("   B dropout on  " + "  ".join(f"{q(F[1][:, j], 50):+5.1f}" for j in range(6)))
    out_json["start_pose_first_command"] = dict(A=[q(F[0][:, j], 50) for j in range(6)],
                                                B=[q(F[1][:, j], 50) for j in range(6)])

    if own:
        print("\nJaw error at the teacher's close frame, median (25-75 %), mm")
        for lk in ("3.33 s", "close-0.33 s"):
            d = S[lk]
            hv = np.array(d["half"])
            fixed, factors = np.zeros(len(hv)), []
            for side in (0, 1):                                    # factor from one half, applied to the other half
                s_fit = stretch([x[5] for x, h in zip(d["last"], hv) if h != side],
                                [x[4] for x, h in zip(d["last"], hv) if h != side], ARM)
                factors.append(s_fit)
                for idx in np.flatnonzero(hv == side):
                    grip_pt, grip_cmd, tgt, p_last = d["last"][idx][:4]
                    fixed[idx] = np.linalg.norm(jaw(np.r_[p_last[:5] / s_fit + pivot[:5], grip_cmd], grip_pt) - tgt)
            eA, eB = np.array(d["errA"]), np.array(d["errB"])
            print(f"   plan made at {lk} (n {len(eA)}):")
            print(f"      A inference                         {q(eA, 50):5.1f} ({q(eA, 25):4.1f}-{q(eA, 75):4.1f})   within 6 mm in {int((eA <= 6).sum())}/{len(eA)}")
            print(f"      B dropout on, mean of {K} answers     {q(eB, 50):5.1f} ({q(eB, 25):4.1f}-{q(eB, 75):4.1f})   within 6 mm in {int((eB <= 6).sum())}/{len(eB)}"
                  f"      (one answer alone: {q(d['oneB'], 50):.1f})")
            print(f"      A with its stretch taken out         {q(fixed, 50):5.1f} ({q(fixed, 25):4.1f}-{q(fixed, 75):4.1f})   within 6 mm in {int((fixed <= 6).sum())}/{len(fixed)}"
                  f"      (factors {factors[0]:.3f}, {factors[1]:.3f}, each found on the other half)")
            if lk == "3.33 s":
                print(f"      C random code, D dropout + random code: {q(CD['errC'], 50):.1f}, {q(CD['errD'], 50):.1f}")
            out_json[f"err/{lk}"] = dict(A=q(eA, 50), B=q(eB, 50), B_one=q(d["oneB"], 50), A_unstretched=q(fixed, 50),
                                         factors=factors, within6=[int((eA <= 6).sum()), int((eB <= 6).sum()),
                                                                   int((fixed <= 6).sum())], n=len(eA),
                                         C=q(CD["errC"], 50) if lk == "3.33 s" else None,
                                         D=q(CD["errD"], 50) if lk == "3.33 s" else None)
    tag = "" if a.dataset == "blush_sim_teacher_v5" else f"_on_{a.dataset}"
    json.dump(out_json, open(OUT / f"offline_train_vs_inference_{a.run}_{a.step}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
