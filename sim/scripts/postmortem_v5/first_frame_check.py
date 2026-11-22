#!/usr/bin/env python3
"""Post-mortem 2026-11-20: is the FIRST observation of a test the same as the first frame the policy was trained on?

    conda activate lerobot && python first_frame_check.py [--step 100000] [--episodes 12] [--save 2]

Why: on training scenes the policy's rollout does not follow the plan the policy makes offline from the RECORDED first
frame of the same scene (offline_first_plan_turn.py: 126 mm apart after 1 s, while the teacher's recorded commands and
its rollout agree within 1 mm). So either the test's first observation differs from the recorded one, or the test loop
does something else. This builds those training scenes in the simulator exactly as eval_truss.py does (same robot
settings, new_scene, get_observation, env_state; no physics rollout) and compares the first observation with the
recorded frame 0 of the same episode: the joint readings, the 6 cue numbers, both camera images, and the 100-step plan
the policy makes from each. Then one part at a time is swapped (the simulator's images with the recorded numbers, and
so on) to see which part moves the plan, and the simulator's images are passed through the dataset's video compression
(AV1, the settings in the dataset's info.json) to see whether the compression alone explains a difference.

Plans are compared by the jaw point their commands ask for (forward kinematics), in mm, at the commands in STEPS.
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

RUNS = ROOT / "outputs" / "diag" / "runs"                       # traced rollouts (eval_truss.py --trace)
REPO = "krish5831/blush_sim_teacher_v5"
SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
FPS = 30
STEPS = (0, 29, 59, 99)     # commands of the plan to look at: its first, after 1 and 2 s, and its last (3.33 s)
CAMS = ("base", "claw")
BIG = 16                    # a pixel "differs clearly" above this many grey levels of 255 (a choice, for the printout)


def q(a, p=50):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--save", type=int, default=2, help="save recorded | simulator | difference pictures of this many scenes")
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

    def av1(img):
        """The image after the dataset's own video compression: a 4-frame clip of it, encoded as generate_truss.py
        encodes (RGBEncoderConfig(vcodec="libsvtav1"): crf 30, GOP 2, yuv420p), first frame decoded as training does."""
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            d = pathlib.Path(td) / "imgs"
            d.mkdir()
            for k in range(4):
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

    def jaws(p, grip_pt):
        out = []
        for j in STEPS:
            o, R = ik.fk(fm.norm_to_rad(p[j])[:5])
            out.append((o + R @ grip_pt) * 1000)
        return np.array(out)

    def rollout_jaw(run):
        out = {}
        f = RUNS / run / "trace.jsonl"
        if not f.exists():
            return out
        want = {round((j + 1) / FPS, 2): n for n, j in enumerate(STEPS)}
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                n = want.get(r["t"])
                if n is not None:
                    out.setdefault(r["seed"], {})[n] = np.array(r["jaw_mm"])
        return out

    roll = rollout_jaw("train_clean")

    # the robot exactly as eval_truss.py builds it for these tests (--headless --no-dr --no-yolo)
    train_ds, rig = et.trained_on(ck)
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    _, _, rop = make_default_processors()
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=seeds[0], viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, yolo_weights=None, state_log_dir=None, rig=rig))
    robot.connect()

    names = ("simulator's first observation", "simulator's images + recorded numbers", "recorded images + simulator's numbers",
             "simulator's base image, rest recorded", "simulator's claw image, rest recorded",
             "simulator's images after AV1, rest recorded", "the teacher's recorded commands")
    D = {n: [] for n in names}
    R_rec, R_sim = [], []
    px = {c: [] for c in CAMS}
    px_av1 = {c: [] for c in CAMS}
    st_d, env_d = [], []
    for n_scene, seed in enumerate(seeds):
        i = by_seed[seed]
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        rec = {"observation.state": st[0], OBS_ENV_STATE: env[0],
               **{f"observation.images.{c}": frame_rgb(i, c, 0) for c in CAMS}}
        robot.new_scene(seed, log=False)                       # eval_truss.py, policy only: the cue at u = 0.5
        obs = robot.get_observation()
        env_sim = np.asarray(robot.env_state(), np.float32)
        sim = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
        sim[OBS_ENV_STATE] = env_sim
        st_d.append(np.abs(np.asarray(sim["observation.state"], float) - st[0]))
        env_d.append(np.abs(env_sim - env[0]))
        comp = {c: av1(np.asarray(sim[f"observation.images.{c}"])) for c in CAMS}
        for c in CAMS:
            k = f"observation.images.{c}"
            d = np.abs(np.asarray(sim[k]).astype(np.int16) - rec[k].astype(np.int16))
            px[c].append((d.mean(), np.percentile(d, 99), (d.max(-1) > BIG).mean() * 100))
            d2 = np.abs(comp[c].astype(np.int16) - rec[k].astype(np.int16))
            px_av1[c].append((d2.mean(), np.percentile(d2, 99), (d2.max(-1) > BIG).mean() * 100))
            if n_scene < a.save:
                row = np.concatenate([rec[k], np.asarray(sim[k]), np.clip(d * 8, 0, 255).astype(np.uint8)], axis=1)
                cv2.imwrite(str(OUT / f"first_frame_{seed}_{c}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        img = lambda f: {k: f[k] for k in f if "images" in k}          # noqa: E731
        num = lambda f: {k: f[k] for k in f if "images" not in k}      # noqa: E731
        variants = {
            names[0]: sim,
            names[1]: {**img(sim), **num(rec)},
            names[2]: {**img(rec), **num(sim)},
            names[3]: {**rec, "observation.images.base": sim["observation.images.base"]},
            names[4]: {**rec, "observation.images.claw": sim["observation.images.claw"]},
            names[5]: {**rec, **{f"observation.images.{c}": comp[c] for c in CAMS}},
        }
        grip_pt = truss_ik.cut_point_in_gripper(kept[i]["plan"]["stalk_mm"] / 2000)
        p_rec = plan(rec)
        j_rec = jaws(p_rec, grip_pt)
        for name, f in variants.items():
            D[name].append(np.linalg.norm(jaws(plan(f), grip_pt) - j_rec, axis=1))
        D[names[6]].append(np.linalg.norm(jaws(ac[:100], grip_pt) - j_rec, axis=1))
        if seed in roll and all(n in roll[seed] for n in range(len(STEPS))):
            r = np.array([roll[seed][n] for n in range(len(STEPS))])
            R_rec.append(np.linalg.norm(j_rec - r, axis=1))
            R_sim.append(np.linalg.norm(jaws(plan(sim), grip_pt) - r, axis=1))
        print(f"scene {n_scene + 1}/{len(seeds)} seed {seed} (cut_u {kept[i]['plan'].get('cut_u', 0.5)}): joints differ by at most "
              f"{st_d[-1].max():.3f} units, cue numbers by {env_d[-1].max():.4f}; base image mean |difference| {px['base'][-1][0]:.2f} of 255 "
              f"({px['base'][-1][2]:.1f} % of pixels above {BIG}), claw {px['claw'][-1][0]:.2f} ({px['claw'][-1][2]:.1f} %); plan from the "
              f"simulator's observation vs plan from the recorded frame: " + ", ".join(f"{v:.1f}" for v in D[names[0]][-1]) + " mm",
              flush=True)
    robot.disconnect()

    print(f"\ncheckpoint {a.step // 1000}k, {len(seeds)} clean training scenes, first observation only (no rollout)")
    print(f"joint readings, simulator vs recorded: largest difference {np.max(st_d):.3f} units (median over scenes {q(np.max(st_d, 1)):.3f})")
    print(f"cue numbers, simulator vs recorded: largest difference per number {np.max(env_d, 0).round(4).tolist()}")
    for c in CAMS:
        b, c2 = np.array(px[c]), np.array(px_av1[c])
        print(f"{c} image, simulator vs recorded: mean |difference| {q(b[:, 0]):.2f} of 255 (median over scenes), 99 % of pixels within "
              f"{q(b[:, 1]):.0f}, pixels differing by more than {BIG}: {q(b[:, 2]):.2f} %.  After passing the simulator's image through AV1: "
              f"{q(c2[:, 0]):.2f}, {q(c2[:, 1]):.0f}, {q(c2[:, 2]):.2f} %")
    print("\nJaw point asked for by each plan, distance from the plan made from the RECORDED frame, mm, median (largest), at commands "
          + ", ".join(str(j + 1) for j in STEPS) + ":")
    for name in names:
        d = np.array(D[name])
        print(f"   {name:42s} " + "   ".join(f"{q(d[:, n]):6.1f} ({d[:, n].max():6.1f})" for n in range(len(STEPS))))
    if R_rec:
        rr, rs = np.array(R_rec), np.array(R_sim)
        print(f"\nAgainst the traced rollout on the same scene (train_clean, {len(rr)} scenes), distance between the jaw point the plan asks "
              "for and the jaw point the rollout had at that moment, mm, median (largest):")
        print("   plan from the recorded frame               " + "   ".join(f"{q(rr[:, n]):6.1f} ({rr[:, n].max():6.1f})" for n in range(len(STEPS))))
        print("   plan from the simulator's first observation " + "   ".join(f"{q(rs[:, n]):6.1f} ({rs[:, n].max():6.1f})" for n in range(len(STEPS))))
    if a.save:
        print(f"pictures (recorded | simulator | difference x8) in {OUT}/first_frame_<seed>_<camera>.png")


if __name__ == "__main__":
    main()
