"""check_dataset.py: sanity checks on a recorded dataset before training (lerobot env).

    python harvest/scripts/check_dataset.py krish5831/blush_pick_v1

Checks the episode count, the camera keys, and that joint values are in the +-100 normalised
range rather than degrees. That confirms use_degrees=false actually stuck while recording.
⚠️ The import path moves on lerobot main. If it fails: grep -rn "class LeRobotDataset" src/
"""
import sys

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MIN_EPISODES = 100   # vault note 03: plan 150, keep >= 100 after deleting bad ones
NORM_LIMIT = 100.5   # use_degrees=false -> range-normalised to [-100, 100] (gripper [0, 100])

d = LeRobotDataset(sys.argv[1])
print(f"episodes: {d.num_episodes}   frames: {d.num_frames}   fps: {d.fps}")
print("camera keys:", [k for k in d.meta.features if k.startswith("observation.images")])

s = torch.stack([d[i]["observation.state"] for i in range(0, len(d), 200)])
lo, hi = s.min(0).values, s.max(0).values
print("state min:", lo.tolist())
print("state max:", hi.tolist())

ok = True
if d.num_episodes < MIN_EPISODES:
    print(f"⚠️ only {d.num_episodes} episodes (< {MIN_EPISODES})"); ok = False
if (lo < -NORM_LIMIT).any() or (hi > NORM_LIMIT).any():
    print("⚠️ state outside +-100: this looks like DEGREES. Recorded with use_degrees=true? Don't train on it."); ok = False
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
