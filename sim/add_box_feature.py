#!/usr/bin/env python3
"""Copy a scripted sim dataset with the target box as numbers: observation.environment_state (K7, 2026-11-01).

    python add_box_feature.py                     # blush_sim_scripted_v1 -> blush_sim_scripted_v1_box (local)

Why: ACT trained on v1 follows the green box only to ~1 tomato width (sim/analyze_policy.py, 2026-11-01:
jaw closes a median 37 mm from new targets, neighbours hang 21-30 mm apart). The box is ~25 px in a
640x480 frame and ResNet18 sees 32 px cells. Handing it the box as 4 numbers removes the guesswork. ACT
reads observation.environment_state as its own encoder token. The real rig has the same numbers: the
birdseye YOLO box at the scene's first frame (ripeness/gate/birdseye.py).

Values: [cx, cy, w, h] of the green box, divided by the birdseye width/height (0..1). The box is picked on
the first frame and held for the whole episode (sim/target.py), so every frame of an episode has the same
4 numbers; they come from the recording session's session.json (scene "box", keyed by episode_index).
The frames, videos and everything else are copied unchanged. Nothing is pushed.
⚠️ Single-fruit task only. Its one source, blush_sim_scripted_v1 (and so v1_box), was deleted locally and on the
Hub 2026-11-03 (K7): main() has nothing to read. box_vector() is still imported by eval_policy / analyze_policy.
The truss datasets get their box (+ cut point) from truss_target.env_vector at record time instead.
"""
import argparse
import json
import pathlib

import numpy as np

BIRDSEYE_WH = (640, 480)          # observation.images.birdseye shape in blush_sim_scripted_v1 (info.json)
SESSIONS = {                      # recording session per dataset (its kept scenes carry episode_index)
    "krish5831/blush_sim_scripted_v1": pathlib.Path.home() / ".cache/blush_sim/states/20261029_173515",
}
NAMES = ["box_cx", "box_cy", "box_w", "box_h"]


def box_vector(box):
    """[x0, y0, x1, y1] in birdseye pixels -> [cx, cy, w, h] / image size. None -> zeros (no box drawn)."""
    if box is None:
        return np.zeros(4, np.float32)
    x0, y0, x1, y1 = box
    W, H = BIRDSEYE_WH
    return np.array([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="krish5831/blush_sim_scripted_v1")
    a = ap.parse_args()
    new_id = a.repo_id + "_box"
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features, recompute_stats
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE

    ds = LeRobotDataset(a.repo_id, root=HF_LEROBOT_HOME / a.repo_id)
    scenes = json.loads((SESSIONS[a.repo_id] / "session.json").read_text())["scenes"]
    by_ep = {s["episode_index"]: s for s in scenes if "episode_index" in s}
    if sorted(by_ep) != list(range(ds.meta.total_episodes)):
        raise SystemExit(f"session.json has episodes {sorted(by_ep)[:5]}..., the dataset {ds.meta.total_episodes}")
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    per_ep = {e: box_vector(by_ep[e]["box"]["box"]) for e in by_ep}
    values = np.stack([per_ep[int(e)] for e in ep_col])
    print(f"{len(values)} frames, {len(per_ep)} episodes; box cx {values[:, 0].min():.2f}-{values[:, 0].max():.2f}, "
          f"cy {values[:, 1].min():.2f}-{values[:, 1].max():.2f}, w px {values[:, 2].min() * BIRDSEYE_WH[0]:.0f}-"
          f"{values[:, 2].max() * BIRDSEYE_WH[0]:.0f}", flush=True)
    new = add_features(ds, {OBS_ENV_STATE: (values, {"dtype": "float32", "shape": [4], "names": NAMES})},
                       output_dir=HF_LEROBOT_HOME / new_id, repo_id=new_id)
    # ⚠️ add_features writes no stats for the new feature (lerobot 5aa74557), and ACT normalises it with them.
    new = recompute_stats(LeRobotDataset(new_id, root=HF_LEROBOT_HOME / new_id))
    st = new.meta.stats.get(OBS_ENV_STATE)
    print(f"wrote {new_id}: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames; "
          f"stats for the box: {'yes' if st else 'MISSING'}", flush=True)
    if st:
        print({k: np.round(np.asarray(v), 3).tolist() for k, v in st.items() if k in ("mean", "std", "min", "max")})


if __name__ == "__main__":
    main()
