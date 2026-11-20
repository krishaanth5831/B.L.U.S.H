#!/usr/bin/env python3
"""Post-mortem 2026-11-20: two facts about the CELL, read from dataset v5 with forward kinematics only.
No physics stepping, no policy, CPU, about 1 minute.

    conda activate lerobot && python cell_geometry_and_claw_view.py

A. Where the arm's joints are when the teacher cuts, against each joint's end of range (normalised units,
   -100..100), and how far the cut point is from the shoulder: is the cut made stretched or folded, and how much
   room is left? Also the same margins for the start pose.
B. What the wrist ("claw") camera has of the cut point in the last 3 s before the jaw shuts: is the point inside
   its 640x480 frame, how far away is it, and how many pixels per mm. ⚠️ "Inside the frame" only: whether a finger
   or a fruit hides the stalk is not checked here.
"""
import glob
import json
import pathlib
import sys

import mujoco
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE.parents[1]
OUT = HERE.parents[2] / "outputs" / "notes" / "2026-11-19_v5_postmortem"   # generated files (git-ignored)
sys.path.insert(0, str(SIM))
import arm_map  # noqa: E402
import expert  # noqa: E402
import scene  # noqa: E402
import truss_ik  # noqa: E402
from lens import LENSES  # noqa: E402

SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
FPS = 30
W, H = 640, 480
LOOK2 = 100                 # frame of the policy's second look with 100-command chunks (3.33 s)
BEFORE_S = (3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
FULL_REACH_MM = 400.0       # shoulder to jaw tip at full stretch (scene.REACH_M comment)


def q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                    for f in sorted(glob.glob(f"{SNAP}/data/*/*.parquet"))])
    df = df.sort_values(["episode_index", "frame_index"])
    rows = []
    for f in LOGS:
        rows += [json.loads(ln) for ln in open(f) if ln.strip()]
    kept = [r for r in rows if r.get("kept") is True]
    eps = {int(k): g for k, g in df.groupby("episode_index")}
    assert len(eps) == len(kept), (len(eps), len(kept))     # same order check as analyse_dataset.py did

    ik, fm = expert.ArmIK(n_samples=10), arm_map.FollowerMap()
    cam = mujoco.mj_name2id(ik.m, mujoco.mjtObj.mjOBJ_CAMERA, "arm/claw")
    lens = LENSES["claw"]
    yaw = np.radians(scene.ARM_YAW_DEG)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    shoulder = (np.asarray(scene.ARM_BASE) + Rz @ scene.SHOULDER_IN_BASE) * 1000      # mm, world

    def view(state, grip_pt, target_mm):
        """jaw point (mm), claw pixel of the target, camera-to-target distance (mm), px per mm there."""
        o, R = ik.fk(fm.norm_to_rad(state)[:5])
        mujoco.mj_camlight(ik.m, ik.d)
        jaw = (o + R @ grip_pt) * 1000
        C, cp = ik.d.cam_xmat[cam].reshape(3, 3), ik.d.cam_xpos[cam] * 1000
        pts = np.stack([target_mm, target_mm + C[:, 0], target_mm + C[:, 1]])       # +1 mm along image x and y
        c = (pts - cp) @ C
        uv = lens.project(np.c_[c[:, 0], -c[:, 1], -c[:, 2]])
        scale = float(np.mean([np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[2] - uv[0])]))
        return jaw, uv[0], float(np.linalg.norm(target_mm - cp)), scale

    A, B, L2 = [], {s: [] for s in BEFORE_S}, []
    for i, r in enumerate(kept):
        if r.get("hold"):
            continue
        g = eps[i]
        st = np.stack(g["observation.state"].to_numpy()).astype(float)
        ac = np.stack(g["action"].to_numpy()).astype(float)
        opened = np.maximum.accumulate(ac[:, 5] > 20)
        shut = np.flatnonzero(opened & (st[:, 5] < 10))
        if not len(shut):
            continue
        ks = int(shut[0])                                         # first frame the jaw IS nearly shut
        grip_pt = truss_ik.cut_point_in_gripper(r["plan"]["stalk_mm"] / 2000)
        target, _, _, _ = view(st[ks], grip_pt, np.zeros(3))      # the cut point = where the jaw point is then
        A.append(dict(slot=r["target"], q=st[ks, :5], reach=float(np.linalg.norm(target - shoulder)),
                      noisy=r.get("noise") is not None))
        for s in BEFORE_S:
            k = ks - int(round(s * FPS))
            if k < 0:
                continue
            jaw, uv, dist, scale = view(st[k], grip_pt, target)
            inside = bool(np.all(np.isfinite(uv)) and 0 <= uv[0] < W and 0 <= uv[1] < H)
            B[s].append((inside, dist, scale if inside else np.nan, float(np.linalg.norm(jaw - target)), *uv))
        if ks > LOOK2:
            jaw, uv, dist, scale = view(st[LOOK2], grip_pt, target)
            inside = bool(np.all(np.isfinite(uv)) and 0 <= uv[0] < W and 0 <= uv[1] < H)
            L2.append((inside, dist, scale if inside else np.nan, float(np.linalg.norm(jaw - target)),
                       (ks - LOOK2) / FPS))

    out = {}
    Q = np.array([a["q"] for a in A])
    print(f"cut episodes: {len(A)}\n\nA. Arm joints when the teacher's jaw shuts (normalised units, each joint's range is -100..100)")
    for j, name in enumerate(JOINTS):
        m = 100 - np.abs(Q[:, j])
        print(f"   {name:14s} 5-50-95 %: {q(Q[:, j], 5):7.1f} {q(Q[:, j], 50):7.1f} {q(Q[:, j], 95):7.1f}   "
              f"min {Q[:, j].min():7.1f} max {Q[:, j].max():7.1f}   room to the nearer end: median {q(m, 50):5.1f}, "
              f"smallest {m.min():5.1f}; within 10 units of an end in {int((m < 10).sum())} episodes")
        out[f"A_{name}"] = dict(p5=q(Q[:, j], 5), p50=q(Q[:, j], 50), p95=q(Q[:, j], 95), min=float(Q[:, j].min()),
                                max=float(Q[:, j].max()), within10=int((m < 10).sum()))
    near = (100 - np.abs(Q)).min(axis=1)
    print(f"   any arm joint within 10 units of an end of its range: {int((near < 10).sum())}/{len(A)} episodes; "
          f"within 5: {int((near < 5).sum())}")
    reach = np.array([a["reach"] for a in A])
    print(f"   shoulder joint to the cut point: 5-50-95 % {q(reach, 5):.0f} {q(reach, 50):.0f} {q(reach, 95):.0f} mm "
          f"(min {reach.min():.0f}, max {reach.max():.0f}); full stretch to the jaw tip is about {FULL_REACH_MM:.0f} mm")
    out["A_reach_mm"] = dict(p5=q(reach, 5), p50=q(reach, 50), p95=q(reach, 95), min=float(reach.min()),
                             max=float(reach.max()))
    s0 = np.asarray(scene.START_NORM[:5])
    print("   start pose " + ", ".join(f"{n} {v:.0f} (room {100 - abs(v):.0f})" for n, v in zip(JOINTS, s0)))
    print(f"   shoulder joint in the world: x {shoulder[0]:.0f}, y {shoulder[1]:.0f}, z {shoulder[2]:.0f} mm; "
          f"the row hangs at y {1000 * (scene.TARGET_ROW - scene.KEYHOLE_OFF):.0f} mm")

    print("\nB. The cut point in the wrist camera's 640x480 frame, before the jaw shuts (teacher episodes)")
    print("   time before   inside the frame   camera to cut point   pixels per mm   jaw point to cut point   pixel (u, v) median")
    for s in BEFORE_S:
        b = np.array(B[s], float)
        if not len(b):
            continue
        print(f"   {s:4.1f} s        {int(b[:, 0].sum()):4d}/{len(b):<4d} ({100 * b[:, 0].mean():3.0f} %)   "
              f"{q(b[:, 1], 50):6.0f} mm            {q(b[:, 2], 50):5.1f}           {q(b[:, 3], 50):6.0f} mm"
              f"               ({q(b[:, 4], 50):4.0f}, {q(b[:, 5], 50):4.0f})")
        out[f"B_{s}"] = dict(n=len(b), inside=int(b[:, 0].sum()), cam_mm=q(b[:, 1], 50), px_per_mm=q(b[:, 2], 50),
                             jaw_mm=q(b[:, 3], 50), u=q(b[:, 4], 50), v=q(b[:, 5], 50))
    b = np.array(L2, float)
    print(f"\n   At the policy's second look (frame {LOOK2} = {LOOK2 / FPS:.2f} s), on the teacher's own path: the cut point is "
          f"inside the wrist frame in {int(b[:, 0].sum())}/{len(b)} episodes ({100 * b[:, 0].mean():.0f} %), "
          f"{q(b[:, 1], 50):.0f} mm from the camera, {q(b[:, 2], 50):.1f} px per mm; the jaw point is still "
          f"{q(b[:, 3], 50):.0f} mm from the cut point and the jaw shuts {q(b[:, 4], 50):.2f} s later (median)")
    out["B_look2"] = dict(n=len(b), inside=int(b[:, 0].sum()), cam_mm=q(b[:, 1], 50), px_per_mm=q(b[:, 2], 50),
                          jaw_mm=q(b[:, 3], 50), s_before_shut=q(b[:, 4], 50))
    json.dump(out, open(OUT / "cell_geometry_and_claw_view.json", "w"), indent=1)
    print(f"\nsaved {OUT / 'cell_geometry_and_claw_view.json'}")


if __name__ == "__main__":
    main()
