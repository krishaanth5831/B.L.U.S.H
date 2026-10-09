#!/usr/bin/env bash
# train_act.sh: ACT baseline (lerobot env).
#   harvest/scripts/train_act.sh smoke   500 steps on the laptop: data loads, both camera keys listed, loss falls
#   harvest/scripts/train_act.sh full    the real run, meant for the rented 4090
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../config/hardware.env"
source "$HERE/../harvest.env"
OUT=outputs/train   # gitignored. Checkpoints go to the Hub, not git.

case "${1:-}" in
  smoke)
    lerobot-train --dataset.repo_id="$DATASET_CLEAN" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false \
      --batch_size=8 --steps=500 --save_freq=500 \
      --output_dir=$OUT/act_smoke --job_name=act_smoke --wandb.enable=false ;;
  full)
    # batch 32 fits a 24 GB card (use 8 on the laptop). Saving every 20k steps lets you eval an early
    # and a late checkpoint: ACT can overfit, so the last one isn't automatically the best.
    # ⚠️ AV1 decoding is CPU-bound. If nvidia-smi shows <~80 % GPU use, raise num_workers.
    lerobot-train --dataset.repo_id="$DATASET_CLEAN" --policy.type=act --policy.device=cuda \
      --policy.repo_id="$ACT_POLICY" \
      --batch_size=32 --steps=80000 --save_freq=20000 --num_workers=8 \
      --output_dir=$OUT/act_blush_v1 --job_name=act_blush_v1 --wandb.enable=true ;;
  *) echo "usage: $0 smoke | full"; exit 1 ;;
esac
