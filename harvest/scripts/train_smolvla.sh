#!/usr/bin/env bash
# train_smolvla.sh: SmolVLA fine-tune from lerobot/smolvla_base (lerobot env).
#   harvest/scripts/train_smolvla.sh smoke   on the laptop first. It catches most problems for free.
#   harvest/scripts/train_smolvla.sh full    on the rented 4090
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../../config/hardware.env"
source "$HERE/../harvest.env"
OUT=outputs/train

# ⚠️ SmolVLA's base config expects its own camera keys. Print them (vault note 05, A3), map
#    birdseye -> overview-style key and claw -> wrist-style key, and pass the SAME map to rollout.sh.
RENAME_MAP=''   # e.g. '{"observation.images.birdseye": "<base key 1>", "observation.images.claw": "<base key 2>"}'
[[ -n "$RENAME_MAP" ]] || { echo "set RENAME_MAP first (see comment)"; exit 1; }

case "${1:-}" in
  smoke)
    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id="$DATASET_CLEAN" \
      --policy.device=cuda --policy.push_to_hub=false --rename_map="$RENAME_MAP" \
      --batch_size=2 --steps=50 --save_freq=50 \
      --output_dir=$OUT/smolvla_smoke --job_name=smolvla_smoke --wandb.enable=false ;;
  full)
    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id="$DATASET_CLEAN" \
      --policy.device=cuda --policy.repo_id="$SMOLVLA_POLICY" --rename_map="$RENAME_MAP" \
      --batch_size=64 --steps=20000 --save_freq=5000 --num_workers=12 \
      --output_dir=$OUT/smolvla_blush_v1 --job_name=smolvla_blush_v1 --wandb.enable=true ;;
  *) echo "usage: $0 smoke | full"; exit 1 ;;
esac
# Values from vault note 05 (lerobot's SmolVLA guide: batch 64, 20k steps). If 64 OOMs on 24 GB,
# use 32. Smoke needs `pip install -e ".[smolvla]"` in the lerobot checkout first.
