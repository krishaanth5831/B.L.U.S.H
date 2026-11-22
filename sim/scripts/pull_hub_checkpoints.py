#!/usr/bin/env python3
"""Pull a Colab run's checkpoints from its Hub repo into outputs/train/<run>/, the way a local training run lays them
out, so pick_checkpoint.py --watch (train_and_test.sh's testers) tests them unchanged (K7 2026-11-22: training moves
to Colab, tests stay on the laptop).
    python sim/scripts/pull_hub_checkpoints.py krish5831/blush_sim_act_v5b64 act_teacher_v5b64 --until 100000
Every POLL_S: lists checkpoints/<step>/ on the Hub (sim/colab_train_act.ipynb pushes one every 5k steps), downloads
each new step's pretrained_model/ only (the weights, ~0.2 GB; the optimizer state stays on the Hub and Drive), then
points checkpoints/last at it. The watcher only tests steps at or below `last`, so a half-downloaded checkpoint is
never tested. Stops once --until is pulled. Needs the lerobot-mj4 env (huggingface_hub) and the HF token on the laptop.
"""
import argparse
import os
import pathlib
import time

from huggingface_hub import HfApi, snapshot_download

REPO = pathlib.Path(__file__).resolve().parents[2]
POLL_S = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hub_repo", help="the Hub model repo the run pushes to, e.g. krish5831/blush_sim_act_v5b64")
    ap.add_argument("run", help="local run name: outputs/train/<run>/checkpoints/<step>/pretrained_model")
    ap.add_argument("--until", type=int, default=100000, help="stop after this step is pulled")
    a = ap.parse_args()
    ck = REPO / "outputs" / "train" / a.run / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    while True:
        try:
            files = api.list_repo_files(a.hub_repo)
        except Exception as e:                        # repo not created yet (first push at 5k) or a network blip
            print(f"{time.strftime('%H:%M')} waiting for {a.hub_repo}: {type(e).__name__}", flush=True)
            files = []
        steps = sorted({f.split("/")[1] for f in files
                        if f.startswith("checkpoints/") and f.split("/")[1].isdigit() and "/pretrained_model/" in f})
        for s in steps:
            dst = ck / s / "pretrained_model"
            if (dst / "model.safetensors").exists():
                continue
            t = time.time()
            snapshot_download(a.hub_repo, allow_patterns=[f"checkpoints/{s}/pretrained_model/*"],
                              local_dir=ck.parent)
            assert (dst / "model.safetensors").exists(), f"{s}: no model.safetensors after download"
            last = ck / "last"
            cur = int(pathlib.Path(os.readlink(last)).name) if last.is_symlink() else -1
            if int(s) > cur:
                tmp = ck / "last.tmp"
                tmp.unlink(missing_ok=True)
                tmp.symlink_to(s)
                tmp.replace(last)                     # atomic: the watcher never sees a missing `last`
            print(f"{time.strftime('%H:%M')} pulled {s} in {time.time() - t:.0f} s", flush=True)
        if steps and int(steps[-1]) >= a.until and (ck / steps[-1] / "pretrained_model" / "model.safetensors").exists():
            print(f"{time.strftime('%H:%M')} {steps[-1]} pulled, done", flush=True)
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
