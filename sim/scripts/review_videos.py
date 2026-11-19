#!/usr/bin/env python3
"""Review videos of a recorded dataset before training (K7 approves 5 per dataset; dataset v5, 2026-11-19).
Base camera left, claw camera right, one mp4 per episode in outputs/videos/review_<dataset>/. Picks 3 clean cuts
(short, middle, the longest), 1 noisy cut and 1 hold, from the generate_truss.py attempt logs of the parts
(merged index = 25 * part + local index: merge_parts.py merges the parts in name order, 25 episodes each).
⚠️ One look only (K7 2026-11-19): delete the folder once the videos are approved or rejected.
    python sim/scripts/review_videos.py krish5831/blush_sim_teacher_v5      (lerobot env; the merged local copy)"""
import glob
import json
import pathlib
import subprocess
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
repo_id = sys.argv[1]
D = pathlib.Path.home() / ".cache/huggingface/lerobot" / repo_id
LOGS = sorted(glob.glob(str(pathlib.Path.home() / f".cache/blush_sim/truss/{repo_id.split('/')[1]}_p*.jsonl")))
OUT = REPO / "outputs" / "videos" / f"review_{repo_id.split('/')[1]}"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for f in LOGS:
    for line in open(f):
        r = json.loads(line)
        if r.get("reason") == "ok" and "part" in r:
            P = int(r["part"][-2:])
            rows.append(dict(ep=25 * P + r["episode_index"], part=P, seed=r["seed"], hold=bool(r.get("hold")),
                             noisy="noise" in r, target=r.get("target")))
lg = pd.DataFrame(rows).set_index("ep")
meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(D / "meta/episodes/*/*.parquet"))))
lg = lg.join(meta.set_index("episode_index")["length"])
clean = lg[~lg.hold & ~lg.noisy].sort_values("length")
noisy, hold = lg[lg.noisy & ~lg.hold], lg[lg.hold]
picks = {"clean_short": clean.index[len(clean) // 10], "clean_mid": clean.index[len(clean) // 2],
         "clean_longest": clean.index[-1], "noisy": noisy.index[len(noisy) // 2], "hold": hold.index[len(hold) // 2]}

for kind, ep in picks.items():
    m = meta[meta.episode_index == ep].iloc[0]
    src = []
    for cam in ("base", "claw"):
        k = f"videos/observation.images.{cam}"
        src += ["-ss", f"{m[k + '/from_timestamp']:.3f}", "-to", f"{m[k + '/to_timestamp']:.3f}", "-i",
                str(D / k / f"chunk-{int(m[k + '/chunk_index']):03d}/file-{int(m[k + '/file_index']):03d}.mp4")]
    out = OUT / f"review_videos_ep{ep:03d}_{kind}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-filter_complex",
                    "[0:v][1:v]scale2ref=oh*mdar:ih[a][b];[a][b]hstack", "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    r = lg.loc[ep]
    print(f"{out}: seed {r.seed}, part {r.part}, target {r.target}, {r.length / 30:.1f} s, "
          f"{'hold' if r.hold else 'noisy' if r.noisy else 'clean'}")
