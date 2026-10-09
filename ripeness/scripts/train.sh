#!/usr/bin/env bash
# train.sh: two-stage YOLO11n training for the ripeness gate. Run from the repo root
# in the `ripeness` env, after build_split.py.
#   ripeness/scripts/train.sh pretrain           public data only
#   ripeness/scripts/train.sh finetune v1        own + public, starting from the pretrain weights
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
YOLO_DIR=$REPO/ripeness/data/yolo
RUNS=$REPO/runs/ripeness    # gitignored. Release a model by copying best.pt into ripeness/models/
# ⚠️ RUNS must be absolute: Ultralytics puts a RELATIVE project under its global runs_dir setting
#    (here a stale ~/Desktop/personal_projects/vinea/runs), not under the cwd.

BATCH=${BATCH:-16}          # ⚠️ the 6 GB laptop GPU may OOM at 16. BATCH=8 ripeness/scripts/train.sh ...
IMGSZ=640                   # matches the 640x480 camera, so going higher buys nothing
HSV_H=0.005                 # ⚠️ Ultralytics default 0.015 shifts hue, and hue is the signal. Keep it small.

case "${1:-}" in
  pretrain)
    yolo detect train model=yolo11n.pt data=$YOLO_DIR/public.yaml imgsz=$IMGSZ \
         epochs=100 batch=$BATCH patience=25 hsv_h=$HSV_H project=$RUNS name=pretrain ;;
  finetune)
    # Starts from the committed pretrain (ripeness/models/pretrain_laboro_v1.pt). After a new pretrain run,
    # copy its best.pt into ripeness/models/ under a new version and change this line.
    yolo detect train model=$REPO/ripeness/models/pretrain_laboro_v1.pt data=$YOLO_DIR/ripeness.yaml imgsz=$IMGSZ \
         epochs=150 batch=$BATCH patience=30 hsv_h=$HSV_H project=$RUNS name=finetune_${2:?version, e.g. v1} ;;
  *) echo "usage: $0 pretrain | finetune <version>"; exit 1 ;;
esac
