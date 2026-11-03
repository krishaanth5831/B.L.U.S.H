#!/usr/bin/env bash
# train.sh: the truss YOLO (YOLO11n) on make_dataset.py's sim birdseye frames. Run in the `ripeness` env.
#   sim/yolo_truss/train.sh v1            -> runs/yolo_truss/truss_v1/weights/best.pt
#   sim/yolo_truss/train.sh eval v1       -> precision/recall on the frozen test split (eval.py)
# Classes: 0 truss_ready, 1 truss_not_ready, 2 cut_point (sim/truss_target.py). ⚠️ Sim-only weights.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA=${DATA:-$HOME/.cache/blush_sim/yolo_truss/truss.yaml}
RUNS=$REPO/runs/yolo_truss     # ⚠️ absolute: Ultralytics puts a relative project under its stale global runs_dir
BATCH=${BATCH:-16}             # ⚠️ the 6 GB laptop GPU may OOM at 16
HSV_H=0.005                    # ⚠️ hue is the ripeness signal (ripeness/scripts/train.sh): keep the hue jitter small
case "${1:-}" in
  eval)
    python "$REPO/sim/yolo_truss/eval.py" "$RUNS/truss_${2:?version}/weights/best.pt" "$DATA" ;;
  v*)
    yolo detect train model=$REPO/yolo11n.pt data=$DATA imgsz=640 epochs=${EPOCHS:-80} batch=$BATCH patience=20 \
         hsv_h=$HSV_H project=$RUNS name=truss_$1 exist_ok=true ;;
  *) echo "usage: $0 <version, e.g. v1> | eval <version>"; exit 1 ;;
esac
