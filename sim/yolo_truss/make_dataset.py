#!/usr/bin/env python3
"""Birdseye frames of truss scenes with YOLO labels from the sim's truth, for the truss YOLO (K7, 2026-11-03).

    python yolo_truss/make_dataset.py                     # default splits into ~/.cache/blush_sim/yolo_truss
    python yolo_truss/make_dataset.py --train 20 --val 4 --test 4 --out /tmp/yt   # a quick look

Per scene seed: truss_scene.build, domain randomisation (randomize.py, as the episodes use), the vines settle
1 s, then FRAMES_PER_SCENE birdseye frames with the arm in different poses (start pose jittered, or reaching
for a truss's cut pose, so the arm hides different parts of the vines as it will mid-episode). Labels
(truss_target.yolo_labels): truss_ready / truss_not_ready boxes around each visible truss (fruit + stalk) and
a cut_point square at each truss's cut zone.

Split BY SEED, never by frame: train, val and a frozen test split live on disjoint seed ranges
(SEED_BASE), so no scene leaks between them. ⚠️ Sim-only: there are no real truss photos labelled yet, so
this model says nothing about the real camera until it's checked on one.

Run in the lerobot env (MuJoCo); training runs in the ripeness env (train.sh).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import arm_map  # noqa: E402
import randomize  # noqa: E402
import scene  # noqa: E402
import truss_ik  # noqa: E402
import truss_scene as ts  # noqa: E402
import truss_target as ttg  # noqa: E402
from cameras import SimCameras  # noqa: E402

OUT = pathlib.Path.home() / ".cache" / "blush_sim" / "yolo_truss"
# Disjoint seed ranges per split (and far from the episode seeds, which start at 0 and 1.5e9 for tests).
SEED_BASE = {"train": 3_000_000, "val": 3_100_000, "test": 3_200_000}
FRAMES_PER_SCENE = 2             # frames per scene, arm posed differently (design choice)
SETTLE_S = 1.0                   # as the episodes' settle_s
JITTER_NORM = 12.0               # +- start-pose jitter per arm joint, normalised units (design choice)
REACH_P = 0.5                    # share of frames with the arm at a truss's cut pose instead


def arm_pose(rng, info, fm, ik):
    if rng.random() < REACH_P and info["trusses"]:
        t = info["trusses"][int(rng.integers(len(info["trusses"])))]
        p, s = ts.cut_target(t)
        x = ik.solve(p, s, t["r"])                  # the cut pose itself
        if x is not None:
            return np.r_[x, np.radians(rng.uniform(5, 30))]
    n = np.asarray(scene.START_NORM, float) + np.r_[rng.uniform(-JITTER_NORM, JITTER_NORM, 5), 0]
    return fm.norm_to_rad(n)


def write(out, split, name, img, rows):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    W, H = ttg.W, ttg.H
    lines = [f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} "
             f"{(b[3] - b[1]) / H:.6f}" for c, b in rows]
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=800, help="scenes")
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    fm, ik = arm_map.FollowerMap(), truss_ik.truss_ik()
    counts = {}
    t0 = time.time()
    total = a.train + a.val + a.test
    done = 0
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        c = counts.setdefault(split, {"images": 0, **{v: 0 for v in ttg.CLASSES.values()}})
        for i in range(n):
            seed = SEED_BASE[split] + i
            m, d, info = ts.build(seed)
            randomize.apply(m, seed, info["fruit"])
            qidx, _ = scene.arm_ids(m)
            for _ in range(int(SETTLE_S / m.opt.timestep)):
                mujoco.mj_step(m, d)
            cams = SimCameras(m)
            rng = np.random.default_rng(seed)
            for f in range(FRAMES_PER_SCENE):
                q = arm_pose(rng, info, fm, ik)
                d.qpos[qidx] = q                      # kinematic: posed, not driven (images only)
                mujoco.mj_forward(m, d)
                img = cams.rgb(d, "birdseye")
                rows = ttg.yolo_labels(cams, d, m, info)
                write(a.out, split, f"s{seed}_f{f}", img, rows)
                c["images"] += 1
                for cls, _ in rows:
                    c[ttg.CLASSES[cls]] += 1
            cams.close()
            done += 1
            if done % 20 == 0 or done == total:
                el = time.time() - t0
                print(f"{done}/{total} scenes ({el / 60:.1f} min, ~{el / done * (total - done) / 60:.0f} min left) "
                      f"{json.dumps(counts)}", flush=True)
    (a.out / "truss.yaml").write_text(
        f"path: {a.out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {k}: {v}\n" for k, v in ttg.CLASSES.items()))
    (a.out / "counts.json").write_text(json.dumps(dict(counts=counts, seed_base=SEED_BASE,
                                                       frames_per_scene=FRAMES_PER_SCENE), indent=2))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
