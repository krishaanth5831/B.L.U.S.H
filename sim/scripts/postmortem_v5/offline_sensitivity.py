#!/usr/bin/env python3
"""Post-mortem 2026-11-19: which inputs decide WHERE the policy puts the jaw? No simulator.

    conda activate lerobot && python offline_sensitivity.py [--step 100000] [--episodes 50]

For clean training episodes the recorded frame at 0 s and at 3.33 s goes through the policy as in eval_truss.py, once
unchanged and once per change below. What is compared is where the predicted chunk commands the jaw at the teacher's
close frame (chunk from 3.33 s) or at the chunk's last step (chunk from 0 s):

  cue +30u / cue -30v      the 6 cue numbers moved 30 px (box centre and cut pixel together)
  cue other scene          the cue numbers of another training scene in a different slot
  cue zero                 the cue numbers set to 0 (what a "nothing ripe" scene gives)
  base +30u / claw +30u    that camera's frame moved 30 px to the right (black where empty)
  base black / claw black  that camera's frame all black
  state +3                 every arm joint's measured position +3 norm units

"Follows" = how far the commanded jaw moves, and for the cue shifts how that compares with where a target seen 30 px
further over would really be (from the dataset's own pixel-to-mm scale in that slot).
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
FPS, W, H = 30, 640, 480


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def shift_img(img, dx):
    out = np.zeros_like(img)
    if dx >= 0:
        out[:, dx:] = img[:, :img.shape[1] - dx]
    else:
        out[:, :dx] = img[:, -dx:]
    return out


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
    z = np.load(OUT / "analyse_dataset_episodes.npz", allow_pickle=True)
    # the dataset's own pixel -> mm scale per slot: (u, v) px = A [x, z, 1] over the clean episodes of the slot
    scale = {}
    for s in set(z["slot"]):
        m = (z["slot"] == s) & (~z["noisy"])
        if m.sum() >= 5:
            X = np.c_[z["cut_xyz"][m][:, [0, 2]], np.ones(m.sum())]
            A, *_ = np.linalg.lstsq(X, z["env"][m][:, 4:] * [W, H], rcond=None)
            scale[s] = np.linalg.inv(A[:2].T)                 # d(x, z) mm per d(u, v) px

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

    recs = []
    for n, seed in enumerate(seeds):
        i = by_seed[seed]
        r, g = kept[i], eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])
        other = kept[by_seed[seeds[(n + 7) % len(seeds)]]]
        j = n
        while other["target"] == r["target"]:                  # a scene in another slot
            j += 1
            other = kept[by_seed[seeds[(j + 7) % len(seeds)]]]
        oe = np.asarray(other["env"], np.float32)
        other_xyz = z["cut_xyz"][list(z["seed"]).index(other["seed"])]
        own_xyz = z["cut_xyz"][list(z["seed"]).index(seed)]
        for name, k in (("0s", 0), ("3.33s", 100)):
            if k >= kc:
                continue
            h = min(kc - k, 99)                                 # the teacher's close frame, or the chunk's end
            base, claw = frame_rgb(i, "base", k), frame_rgb(i, "claw", k)
            f0 = {"observation.state": st[k], OBS_ENV_STATE: env[k], "observation.images.base": base,
                  "observation.images.claw": claw}
            p0 = predict(f0)
            j0 = jaw(p0[h], grip_pt)
            du, dv = np.array([30 / W, 0, 0, 0, 30 / W, 0], np.float32), np.array([0, -30 / H, 0, 0, 0, -30 / H],
                                                                                 np.float32)
            var = {"cue +30u": {**f0, OBS_ENV_STATE: env[k] + du}, "cue -30v": {**f0, OBS_ENV_STATE: env[k] + dv},
                   "cue other scene": {**f0, OBS_ENV_STATE: oe}, "cue zero": {**f0, OBS_ENV_STATE: env[k] * 0},
                   "base +30u": {**f0, "observation.images.base": shift_img(base, 30)},
                   "claw +30u": {**f0, "observation.images.claw": shift_img(claw, 30)},
                   "base black": {**f0, "observation.images.base": base * 0},
                   "claw black": {**f0, "observation.images.claw": claw * 0},
                   "state +3": {**f0, "observation.state": st[k] + np.array([3, 3, 3, 3, 3, 0], np.float32)}}
            rec = dict(seed=seed, slot=r["target"], at=name, h=h, base_err=float(np.linalg.norm(j0 - jaw(ac[k + h], grip_pt))),
                       travel=float(np.linalg.norm(j0 - jaw(p0[0], grip_pt))), grip_close=float(p0[h, 5]))
            S = scale.get(r["target"])
            for vn, fr in var.items():
                p = predict(fr)
                d = jaw(p[h], grip_pt) - j0
                rec[vn] = dict(move=float(np.linalg.norm(d)), d=d.round(1).tolist(), grip=float(p[h, 5]),
                               travel=float(np.linalg.norm(jaw(p[h], grip_pt) - jaw(p[0], grip_pt))))
                if vn in ("cue +30u", "cue -30v") and S is not None:
                    want = S @ (np.array([30.0, 0.0]) if vn == "cue +30u" else np.array([0.0, -30.0]))   # (dx, dz) mm
                    got = np.array([d[0], d[2]])
                    rec[vn].update(want=want.round(1).tolist(),
                                   follow=float(got @ want / (want @ want)))   # 1 = moved as far as the target would
                if vn == "cue other scene":
                    want = other_xyz - own_xyz
                    rec[vn].update(want_mm=float(np.linalg.norm(want)), follow=float(d @ want / (want @ want)))
            recs.append(rec)
    json.dump(recs, open(OUT / f"offline_sensitivity_{a.step // 1000}k.json", "w"))
    print(f"checkpoint {a.step // 1000}k, {len(seeds)} clean training episodes: how far the commanded jaw moves when one "
          f"input is changed (mm, median and 90 %)")
    for name in ("0s", "3.33s"):
        R = [x for x in recs if x["at"] == name]
        print(f"\nchunk predicted from {name} ({len(R)} episodes); unchanged input: {q([x['base_err'] for x in R], 50):.1f} mm "
              f"from the teacher, chunk travels {q([x['travel'] for x in R], 50):.0f} mm")
        for vn in ("cue +30u", "cue -30v", "cue other scene", "cue zero", "base +30u", "claw +30u", "base black",
                   "claw black", "state +3"):
            mv = [x[vn]["move"] for x in R]
            line = f"   {vn:16s} jaw moves {q(mv, 50):5.1f} mm ({q(mv, 90):5.1f})"
            if "follow" in R[0][vn]:
                fo = [x[vn]["follow"] for x in R if "follow" in x[vn]]
                line += f" | follows the cue by {q(fo, 50):4.2f} of the way (10-90 %: {q(fo, 10):.2f} to {q(fo, 90):.2f})"
            if vn in ("cue +30u", "cue -30v"):
                line += f" | a real target shift would be {q([np.linalg.norm(x[vn]['want']) for x in R if 'want' in x[vn]], 50):.0f} mm"
            if vn == "cue other scene":
                line += f" | the other target is {q([x[vn]['want_mm'] for x in R], 50):.0f} mm away"
            if vn in ("cue zero", "base black", "claw black"):
                line += (f" | chunk travel {q([x[vn]['travel'] for x in R], 50):.0f} mm, gripper at the close "
                         f"{q([x[vn]['grip'] for x in R], 50):.0f} (unchanged: {q([x['grip_close'] for x in R], 50):.0f})")
            print(line)


if __name__ == "__main__":
    main()
