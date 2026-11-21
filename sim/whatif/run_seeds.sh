#!/bin/bash
# Teacher v5 dry runs over a seed file, clean (no noise), as the census: run_seeds.sh TAG SEEDFILE [RETRIES]
# Kept rows: ~/.cache/blush_sim/truss/blush_eval_v5_<TAG>_dry_*.jsonl; stdout in logs/<TAG>.log
TAG=$1; SEEDS=$(realpath $2); RETRIES=${3:-4}
SIM=$(cd "$(dirname "$0")/.." && pwd)
OUT=$SIM/../outputs/whatif_logs; mkdir -p $OUT
systemd-run --user --unit=blush-v5-$TAG --collect -q -p MemoryMax=5G -p MemorySwapMax=0 \
  -E OMP_NUM_THREADS=1 -E MKL_NUM_THREADS=1 -E OPENBLAS_NUM_THREADS=1 \
  bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4 && cd $SIM && python -u generate_truss.py --teacher --no-dr --no-yolo --dry --noise-share 0 --retries $RETRIES --seeds $SEEDS --episodes 100000 --repo-id krish5831/blush_eval_v5_$TAG > $OUT/$TAG.log 2>&1"
