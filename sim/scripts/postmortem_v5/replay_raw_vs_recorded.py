#!/usr/bin/env python3
"""Post-mortem 2026-11-20: through a whole training episode, how do the pictures the policy was TRAINED on (decoded
from the dataset's video) differ from the pictures a TEST gives it (the raw render), and what does that do to its plan?

    conda activate lerobot && python replay_raw_vs_recorded.py [--step 100000] [--episodes 8]

Why: claw_frame_leak.py showed that the recorded claw start picture tells the policy how the teacher is about to
move, although the raw picture is the same wall in every scene. This replays the recorded commands of clean training
episodes in the simulator (same seed, so the same world; the replayed joint readings are checked against the recorded
ones), keeps the raw pictures, and compares at several moments of the episode:
  1. raw picture vs recorded picture, per camera
  2. the plan from the recorded frame vs the plan from the same frame with the raw pictures (numbers unchanged), each
     against the teacher's own next commands, and with only one camera's picture swapped
  3. where the difference comes from: the raw frames from a moment on are encoded as the dataset encodes its video
     (lerobot encode_video_frames, RGBEncoderConfig(vcodec="libsvtav1"): AV1, crf 30, a key frame every 2 frames) as
     clips of 1, 2, 3, 5, 9, 17 and 33 frames, and the FIRST frame of each clip is decoded. If that frame changes with
     the number of frames that follow it, the encoder writes the future into the present frame.

Plans are compared by the jaw point their commands ask for (forward kinematics), in mm.
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
import arm_map  # noqa: E402
import expert  # noqa: E402
import truss_ik  # noqa: E402

REPO = "krish5831/blush_sim_teacher_v5"
SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
TIMES = (0, 16, 30, 60, 90, 100, 116, 130, 150, 180)   # frames of the episode to look at (100 = 3.33 s, where a test
                                                       # makes its second plan; the teacher closes the jaw near 135)
LOOK = (9, 29, 99)                                     # commands 10, 30 and 100 of a plan
CLIPS = (1, 2, 3, 5, 9, 17, 33)                        # clip lengths for part 3 (frames, starting at the moment looked at)
CLIP_AT = (0, 100)                                     # moments for part 3 (even frames: key frames, as in the dataset)
CAMS = ("base", "claw")


def q(a, p=50):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()
    import cv2
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES
    from lerobot.utils.import_utils import register_third_party_plugins
    sys.argv = sys.argv[:1]
    import eval_truss as et
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

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

    def first_of_clip(frames):
        """The first frame of a clip after the dataset's video encoding and decoding."""
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
        mean_t = post(torch.zeros(1, 6, device=device))[0].to(device).float()          # un-normalising: x * std + mean
        std_t = post(torch.ones(1, 6, device=device))[0].to(device).float() - mean_t

    def plan(frame):
        """(100, 6) commands in joint units, the plan as it runs in a test (dropout off)."""
        x = pre(prepare_observation_for_inference(dict(frame), device, TASK, "so101_mujoco"))
        x[OBS_IMAGES] = [x[key] for key in policy.config.image_features]
        policy.eval()
        with torch.inference_mode():
            out = policy.model(x)[0]
        return (out * std_t + mean_t)[0].cpu().numpy().astype(float)

    ik, fm = expert.ArmIK(n_samples=10), arm_map.FollowerMap()

    def jaw(norm6, grip_pt):
        o, R = ik.fk(fm.norm_to_rad(np.asarray(norm6, float))[:5])
        return (o + R @ grip_pt) * 1000

    def jaws(p, grip_pt):
        return np.array([jaw(p[j], grip_pt) for j in LOOK])

    _, rig = et.trained_on(ck)
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=seeds[0], viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, yolo_weights=None, state_log_dir=None, rig=rig))
    robot.connect()

    K = max(max(TIMES) + 1, max(CLIP_AT) + max(CLIPS))
    px = {t: {c: [] for c in CAMS} for t in TIMES}
    M = {t: {k: [] for k in ("raw_vs_rec", "rec_vs_teacher", "raw_vs_teacher", "rawclaw_vs_rec", "rawbase_vs_rec",
                             "trav_teacher", "trav_rec", "trav_raw")} for t in TIMES}
    clip = {t: {n: {"px_base": [], "px_claw": [], "plan": []} for n in CLIPS + ("still",)} for t in CLIP_AT}
    fidelity = []
    for n_scene, seed in enumerate(seeds):
        i = by_seed[seed]
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(kept[i]["plan"]["stalk_mm"] / 2000)
        k_end = min(K, len(ac))
        robot.new_scene(seed, log=False, cut_u=kept[i]["plan"].get("cut_u", 0.5))   # as the recording pass
        obs = robot.get_observation()
        raw, qs = [], []
        for k in range(k_end):                                  # the recorded commands again, raw pictures kept
            raw.append({c: np.array(obs[c]) for c in CAMS})
            qs.append([obs[f"{j}.pos"] for j in arm_map.JOINTS])
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, ac[k])})
            obs = robot.get_observation()
        fidelity.append(float(np.abs(np.array(qs) - st[:k_end]).max()))
        for t in TIMES:
            if t >= k_end:
                continue
            rec = {"observation.state": st[t], OBS_ENV_STATE: env[t],
                   **{f"observation.images.{c}": frame_rgb(i, c, t) for c in CAMS}}
            rawf = {**rec, **{f"observation.images.{c}": raw[t][c] for c in CAMS}}
            for c in CAMS:
                d = np.abs(raw[t][c].astype(np.int16) - rec[f"observation.images.{c}"].astype(np.int16))
                px[t][c].append(d.mean())
            tea = np.concatenate([ac[t:t + 100], np.repeat(ac[-1:], max(0, t + 100 - len(ac)), 0)])
            j_tea, here = jaws(tea, grip_pt), jaw(st[t], grip_pt)
            j_rec, j_raw = jaws(plan(rec), grip_pt), jaws(plan(rawf), grip_pt)
            j_rc = jaws(plan({**rec, "observation.images.claw": raw[t]["claw"]}), grip_pt)
            j_rb = jaws(plan({**rec, "observation.images.base": raw[t]["base"]}), grip_pt)
            m = M[t]
            m["raw_vs_rec"].append(np.linalg.norm(j_raw - j_rec, axis=1))
            m["rec_vs_teacher"].append(np.linalg.norm(j_rec - j_tea, axis=1))
            m["raw_vs_teacher"].append(np.linalg.norm(j_raw - j_tea, axis=1))
            m["rawclaw_vs_rec"].append(np.linalg.norm(j_rc - j_rec, axis=1))
            m["rawbase_vs_rec"].append(np.linalg.norm(j_rb - j_rec, axis=1))
            m["trav_teacher"].append(np.linalg.norm(j_tea - here, axis=1))
            m["trav_rec"].append(np.linalg.norm(j_rec - here, axis=1))
            m["trav_raw"].append(np.linalg.norm(j_raw - here, axis=1))
            if t in CLIP_AT:
                variants = {n: raw[t:t + n] for n in CLIPS if t + n <= k_end}
                variants["still"] = [raw[t]] * 17
                for n, frames in variants.items():
                    dec = {c: first_of_clip([f[c] for f in frames]) for c in CAMS}
                    for c in CAMS:
                        clip[t][n][f"px_{c}"].append(np.abs(dec[c].astype(np.int16) - rec[f"observation.images.{c}"].astype(np.int16)).mean())
                    j = jaws(plan({**rec, **{f"observation.images.{c}": dec[c] for c in CAMS}}), grip_pt)
                    clip[t][n]["plan"].append(np.linalg.norm(j - j_rec, axis=1))
        print(f"scene {n_scene + 1}/{len(seeds)} seed {seed}: replayed {k_end} frames, joint readings within {fidelity[-1]:.3f} units of the "
              f"recorded ones", flush=True)
    robot.disconnect()

    f3 = lambda v: "  ".join(f"{q(np.array(v)[:, n]):6.1f}" for n in range(len(LOOK)))   # noqa: E731
    print(f"\ncheckpoint {a.step // 1000}k, {len(seeds)} clean training episodes replayed; joint readings of the replay within "
          f"{max(fidelity):.3f} units of the recorded ones (largest over all frames and episodes)")
    print("\n1+2. By moment of the episode (medians over episodes; plan columns = at commands 10, 30, 100 after that moment, mm)")
    print("   frame (clock)   raw vs recorded picture      plan(raw) vs plan(recorded)   plan(recorded) vs teacher   plan(raw) vs teacher     "
          "only the claw picture raw    only the base picture raw")
    for t in TIMES:
        m = M[t]
        if not m["raw_vs_rec"]:
            continue
        print(f"   {t:3d} ({t / FPS:4.2f} s)    base {q(px[t]['base']):4.2f}  claw {q(px[t]['claw']):4.2f}      {f3(m['raw_vs_rec'])}        "
              f"{f3(m['rec_vs_teacher'])}      {f3(m['raw_vs_teacher'])}      {f3(m['rawclaw_vs_rec'])}       {f3(m['rawbase_vs_rec'])}")
    print("\n   how far the jaw travels in the next 10, 30, 100 commands, mm, median: the teacher | plan from the recorded frame | plan from the raw pictures")
    for t in TIMES:
        m = M[t]
        if m["trav_teacher"]:
            print(f"   {t:3d} ({t / FPS:4.2f} s)    {f3(m['trav_teacher'])}   |   {f3(m['trav_rec'])}   |   {f3(m['trav_raw'])}")
    print("\n3. The first frame of a clip of raw frames after the dataset's video encoding, against the RECORDED frame of that moment "
          "(mean |difference| of 255, median over episodes), and the plan from it against the plan from the recorded frame (mm at commands 10, 30, 100)")
    for t in CLIP_AT:
        print(f"   clip starting at frame {t}:")
        for n in CLIPS + ("still",):
            c = clip[t][n]
            if c["plan"]:
                name = f"{n:2d} frame{'s' if n != 1 else ' '}" if n != "still" else "the same frame 17 times"
                print(f"      {name:24s} base {q(c['px_base']):5.2f}   claw {q(c['px_claw']):5.2f}     plan {f3(c['plan'])}")
    out = OUT / f"replay_raw_vs_recorded_{a.step // 1000}k.json"
    out.write_text(json.dumps(dict(times=TIMES, look=[j + 1 for j in LOOK],
                                   M={str(t): {k: np.array(v).tolist() for k, v in M[t].items()} for t in TIMES},
                                   px={str(t): {c: [float(x) for x in v] for c, v in px[t].items()} for t in TIMES},
                                   clip={str(t): {str(n): {k: np.array(v).tolist() for k, v in c.items()} for n, c in clip[t].items()}
                                         for t in CLIP_AT})))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
