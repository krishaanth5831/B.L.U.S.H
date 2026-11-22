#!/usr/bin/env python3
"""Post-mortem 2026-11-20: does the RECORDED start frame of the claw camera tell the policy what the teacher is
about to do? No simulator, no training: recorded frames through the trained policy.

    conda activate lerobot && python claw_frame_leak.py [--step 100000] [--episodes 50] [--donors 6]

Why: first_frame_check.py found that the simulator's first observation and the recorded first frame of the same
training scene have the same joint readings and cue numbers and almost the same pictures, yet the policy's plans from
the two are about 110 mm apart after 1 s, and that the CLAW picture alone causes it. At the start pose the claw camera
looks at a wall: its raw picture is the same in every scene (checked: the simulator's start pictures of three scenes
differ by at most 1 grey level of 255). The recorded start pictures of those scenes differ from each other (a quarter
to a third of the pixels). The scene cannot be the reason, so the video encoding must be: the encoder looks at the
frames that FOLLOW when it encodes a frame, so a recorded frame can carry traces of the motion that comes next.

The test: keep everything of training episode A (joint readings, cue numbers, base picture) and swap in the claw start
picture of another episode B. If the plan now moves like B's teacher in the first second, the policy reads the coming
motion out of the recorded claw picture, which a live camera can never give it.

"Travel" = how far the jaw's cut point has moved from the start pose after n commands of a plan (forward kinematics).
The teacher starts in two ways in this dataset: a slow start (about 40 mm after 10 commands) and a fast swing (107 mm).
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
N10, N30, N100 = 9, 29, 99   # commands 10, 30 and 100 of a plan (0.33 s, 1 s, 3.33 s)
FAST_MM = 80.0               # teacher travel after 10 commands above this = a fast swing (the two groups sit at about
                             # 40 and 107 mm; read off the traced teacher runs, 2026-11-20)


def q(a, p=50):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--donors", type=int, default=6, help="claw pictures of this many other episodes swapped into each episode")
    ap.add_argument("--raw-claw", type=pathlib.Path, default=OUT / "first_frame_10000001_claw.png",
                    help="a picture saved by first_frame_check.py: its middle third is the simulator's raw claw start picture")
    a = ap.parse_args()
    import os
    import tempfile

    import cv2
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames
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

    def first_of_clip(frames):
        """The first frame of a clip after the dataset's video encoding and decoding (as generate_truss.py encodes)."""
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

    E = []
    for seed in seeds:
        i = by_seed[seed]
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(g["action"].to_numpy()).astype(np.float32)
        env = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        grip_pt = truss_ik.cut_point_in_gripper(kept[i]["plan"]["stalk_mm"] / 2000)
        start = jaw(st[0], grip_pt)
        f0 = {"observation.state": st[0], OBS_ENV_STATE: env[0], "observation.images.base": frame_rgb(i, "base", 0),
              "observation.images.claw": frame_rgb(i, "claw", 0)}
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        kc = int(np.flatnonzero(opened & (ac[:, 5] < 10))[0])       # the teacher's close frame
        E.append(dict(seed=seed, f0=f0, ac=ac, grip=grip_pt, start=start, goal=float(ac[kc, 0]),
                      t10=float(np.linalg.norm(jaw(ac[N10], grip_pt) - start)),
                      t30=float(np.linalg.norm(jaw(ac[N30], grip_pt) - start)), end=jaw(ac[N100], grip_pt)))
    n = len(E)
    fast = np.array([e["t10"] > FAST_MM for e in E])
    print(f"checkpoint {a.step // 1000}k, {n} clean training episodes; the teacher starts with a fast swing in {int(fast.sum())}, slowly in "
          f"{int((~fast).sum())} (travel after 10 commands: fast {q([e['t10'] for e, f in zip(E, fast) if f]):.0f} mm, slow "
          f"{q([e['t10'] for e, f in zip(E, fast) if not f]):.0f} mm)")

    # 1. the recorded claw start pictures: do they differ, and does the difference go with what the teacher does next?
    C = np.stack([e["f0"]["observation.images.claw"] for e in E]).astype(np.int16)
    pair = [np.abs(C[i] - C[j]).mean() for i in range(n) for j in range(i + 1, n)]
    same = [np.abs(C[i] - C[j]).mean() for i in range(n) for j in range(i + 1, n) if fast[i] == fast[j]]
    other = [np.abs(C[i] - C[j]).mean() for i in range(n) for j in range(i + 1, n) if fast[i] != fast[j]]
    print("\n1. The recorded claw start pictures (the raw picture is the same wall in every scene):")
    print(f"   two episodes' pictures differ by {q(pair):.2f} grey levels of 255 on average (median pair; range {min(pair):.2f} to {max(pair):.2f})")
    print(f"   pairs whose teachers start the same way: {q(same):.2f}; pairs whose teachers start differently: {q(other):.2f}")
    raw = None
    if a.raw_claw.exists():
        raw = cv2.cvtColor(cv2.imread(str(a.raw_claw))[:, 640:1280], cv2.COLOR_BGR2RGB)
        d_raw = np.abs(C - raw.astype(np.int16)).mean((1, 2, 3))
        print(f"   distance from the simulator's raw picture: fast starts {q(d_raw[fast]):.2f}, slow starts {q(d_raw[~fast]):.2f} "
              f"(correlation with the teacher's travel after 10 commands {corr(d_raw, [e['t10'] for e in E]):+.2f})")

    # 2. plans: own picture, another episode's picture, the raw picture, the average recorded picture
    def travel(p, e):
        return (float(np.linalg.norm(jaw(p[N10], e["grip"]) - e["start"])), float(np.linalg.norm(jaw(p[N30], e["grip"]) - e["start"])))

    own = [plan(e["f0"]) for e in E]
    own_t = np.array([travel(p, e) for p, e in zip(own, E)])
    tea_t = np.array([(e["t10"], e["t30"]) for e in E])
    print("\n2. Jaw travel after 10 and 30 commands, mm, median:")
    print(f"   the teacher                                           fast starts {q(tea_t[fast, 0]):5.0f} {q(tea_t[fast, 1]):5.0f}    slow starts {q(tea_t[~fast, 0]):5.0f} {q(tea_t[~fast, 1]):5.0f}")
    print(f"   plan from the episode's own recorded frame            fast starts {q(own_t[fast, 0]):5.0f} {q(own_t[fast, 1]):5.0f}    slow starts {q(own_t[~fast, 0]):5.0f} {q(own_t[~fast, 1]):5.0f}"
          f"    (correlation with the teacher, after 10: {corr(own_t[:, 0], tea_t[:, 0]):+.2f}, after 30: {corr(own_t[:, 1], tea_t[:, 1]):+.2f})")
    extra = {}
    if raw is not None:
        extra["same frame, claw picture = the simulator's raw one"] = raw
    extra["same frame, claw picture = average of the recorded ones"] = np.clip(np.rint(C.mean(0)), 0, 255).astype(np.uint8)
    for name, img in extra.items():
        t = np.array([travel(plan({**e["f0"], "observation.images.claw": img}), e) for e in E])
        print(f"   {name:53s} fast starts {q(t[fast, 0]):5.0f} {q(t[fast, 1]):5.0f}    slow starts {q(t[~fast, 0]):5.0f} {q(t[~fast, 1]):5.0f}"
              f"    (correlation with the teacher, after 10: {corr(t[:, 0], tea_t[:, 0]):+.2f}, after 30: {corr(t[:, 1], tea_t[:, 1]):+.2f})")

    # 3. the swap: episode A's frame with episode B's claw start picture
    rng = np.random.default_rng(0)
    S = []
    for ia, ea in enumerate(E):
        for ib in rng.choice([j for j in range(n) if j != ia], size=min(a.donors, n - 1), replace=False):
            eb = E[int(ib)]
            p = plan({**ea["f0"], "observation.images.claw": eb["f0"]["observation.images.claw"]})
            t10, t30 = travel(p, ea)
            S.append(dict(a_fast=bool(fast[ia]), b_fast=bool(fast[int(ib)]), t10=t10, t30=t30, a10=ea["t10"], b10=eb["t10"],
                          a30=ea["t30"], b30=eb["t30"], own10=own_t[ia, 0], own30=own_t[ia, 1],
                          end_shift=float(np.linalg.norm(jaw(p[N100], ea["grip"]) - jaw(own[ia][N100], ea["grip"]))),
                          end_err=float(np.linalg.norm(jaw(p[N100], ea["grip"]) - ea["end"])),
                          own_end_err=float(np.linalg.norm(jaw(own[ia][N100], ea["grip"]) - ea["end"]))))
    print(f"\n3. The swap ({len(S)} plans): episode A's joint readings, cue numbers and base picture, with episode B's recorded claw start picture")
    print("   jaw travel of the plan after 10 and 30 commands, mm, median:")
    for af in (True, False):
        for bf in (True, False):
            v = [s for s in S if s["a_fast"] == af and s["b_fast"] == bf]
            if v:
                print(f"      A's teacher starts {'fast' if af else 'slow'}, B's teacher starts {'fast' if bf else 'slow'} (n {len(v):3d}): "
                      f"{q([s['t10'] for s in v]):5.0f} {q([s['t30'] for s in v]):5.0f}     (A's own plan: {q([s['own10'] for s in v]):5.0f} {q([s['own30'] for s in v]):5.0f})")
    print(f"   correlation of the plan's travel after 10 commands with B's teacher (whose claw picture it is): {corr([s['t10'] for s in S], [s['b10'] for s in S]):+.2f}; "
          f"with A's teacher (whose scene it is): {corr([s['t10'] for s in S], [s['a10'] for s in S]):+.2f}")
    print(f"   the same after 30 commands: with B's teacher {corr([s['t30'] for s in S], [s['b30'] for s in S]):+.2f}; with A's teacher {corr([s['t30'] for s in S], [s['a30'] for s in S]):+.2f}")
    print(f"   where the plan ENDS (command 100): it moves {q([s['end_shift'] for s in S]):.1f} mm when the claw picture is swapped (median; 90 %: "
          f"{q([s['end_shift'] for s in S], 90):.1f}); distance from the teacher's command 100 in the same scene: {q([s['end_err'] for s in S]):.1f} mm "
          f"with B's picture, {q([s['own_end_err'] for s in S]):.1f} mm with A's own")
    # 4. the first plan with each start picture a test could give the claw camera: where the plan ends
    if raw is not None:
        medoid = int(np.argmin([np.abs(C - C[i]).mean() for i in range(n)]))
        pics = {"the episode's own recorded picture (training input)": None,
                "the simulator's raw picture (what every test so far gave)": raw,
                "raw picture through the video encoder, 1 frame": first_of_clip([raw]),
                "raw picture through the video encoder, 17 copies": first_of_clip([raw] * 17),
                "average of the recorded start pictures": np.clip(np.rint(C.mean(0)), 0, 255).astype(np.uint8),
                "one typical recorded start picture (the most central)": E[medoid]["f0"]["observation.images.claw"]}
        print("\n4. The first plan by the start picture the claw camera gives (base picture and numbers as recorded), medians over episodes:")
        print("   start picture                                                 jaw point vs the plan from the own recorded frame      "
              "turn left in the LAST shoulder command, deg     travel after")
        print("                                                                 at commands 10, 30, 100 (mm)                          "
              "(25-75 %; + = short of the teacher's turn)       10, 30 commands")
        UNIT_DEG = 1.116            # deg of shoulder_pan per norm unit (joint range +-111.6 deg = +-100 units; model jnt_range)
        res = {}
        for name, img in pics.items():
            ps = own if img is None else [plan({**e["f0"], "observation.images.claw": img}) for e in E]
            d = np.array([[np.linalg.norm(jaw(p[j], e["grip"]) - jaw(o[j], e["grip"])) for j in (N10, N30, N100)]
                          for p, o, e in zip(ps, own, E)])
            turn = np.array([(e["goal"] - p[N100, 0]) * UNIT_DEG for p, e in zip(ps, E)])
            tr = np.array([travel(p, e) for p, e in zip(ps, E)])
            res[name] = dict(dist=d.tolist(), turn=turn.tolist())
            print(f"   {name:62s} {q(d[:, 0]):6.1f} {q(d[:, 1]):6.1f} {q(d[:, 2]):6.1f}                  {q(turn):+6.2f} ({q(turn, 25):+6.2f} to {q(turn, 75):+6.2f}); "
                  f"more than 2 deg short in {int((turn > 2).sum()):2d}/{n}       {q(tr[:, 0]):4.0f} {q(tr[:, 1]):4.0f}")
        tt = np.array([(e["goal"] - e["ac"][N100, 0]) * UNIT_DEG for e in E])
        print(f"   {'the teacher (its own command 100)':62s}                                           {q(tt):+6.2f} ({q(tt, 25):+6.2f} to {q(tt, 75):+6.2f})")
    out = OUT / f"claw_frame_leak_{a.step // 1000}k.json"
    out.write_text(json.dumps(dict(swaps=S, teacher=tea_t.tolist(), own=own_t.tolist(), fast=fast.tolist())))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
