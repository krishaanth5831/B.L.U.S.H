#!/usr/bin/env bash
# record_parts.sh: one worker of a dataset recorded as parts of 25 episodes (how dataset v5 was recorded,
# 2026-11-18: 3 workers, 32 parts, 7.4 h). One process per part, because a recorder's memory is only released when
# its process ends and a lerobot dataset is only readable after its finalize; a part killed before that (no parquet
# footer) is redone with the same seeds, a finished one is kept. Run W workers as parts w, w+W, w+2W, ...:
#   for w in 0 1 2; do nohup sim/scripts/record_parts.sh krish5831/blush_sim_teacher_v6 11000000 $w 3 32 & done
#   python sim/merge_parts.py krish5831/blush_sim_teacher_v6 krish5831/blush_sim_teacher_v6_p   # then the Hub
# Seeds of part P: SEED_BASE + 1000 P (pick a range nothing else uses; sim/README.md lists the used ones).
# ⚠️ RAM (K7 2026-11-18): each part peaked at ~4.4 GB; at most 3 workers, keep 8 GB free (`free -g` first).
# Extra generate_truss.py flags: GEN_ARGS (default: dataset v5's recipe).
# SEEDS_DIR: a folder of seed plans, one per part (p00.jsonl ..., sim/scripts/v6_seed_plan.py): part N then tries
# the seeds of its plan in order, up to the plan's quota per cell, instead of SEED_BASE + 1000 N upward.
# Dataset v6's 600 new episodes (K7 2026-11-26; NOT RUN YET, needs K7's go):
#   for w in 0 1 2; do GEN_ARGS="--teacher --no-dr --no-yolo --rig trio --layout-rig base --world crate23 \
#       --start-jitter-deg 3 --noise-share 0.5 --retries 6" SEEDS_DIR=sim/seed_plan_v6 \
#       nohup sim/scripts/record_parts.sh krish5831/<name>_new 10100000 $w 3 24 & done
#   ⚠️ --world crate23 (the dataset's world is v5's crate, not the code default) and --layout-rig base (the scenes
#   come from the generator v5's 800 came from; only the cameras are the new rig's). No --balance, no --hold-share:
#   the seed plan says which seeds and how many per cell.
set -uo pipefail
REPO_ID=${1:?repo id, e.g. krish5831/blush_sim_teacher_v6}; SEED_BASE=${2:?first seed}
P=${3:?first part}; STRIDE=${4:?workers}; END=${5:?parts}
SIM=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$SIM/../outputs/record_$(basename "$REPO_ID"); mkdir -p "$LOGS"
GEN_ARGS=${GEN_ARGS:---teacher --no-dr --no-yolo --balance --hold-share 0.10 --noise-share 0.5 --retries 6 --max-attempts 900}
SEEDS_DIR=${SEEDS_DIR:-}
[ -n "$SEEDS_DIR" ] && SEEDS_DIR=$(cd "$SEEDS_DIR" && pwd)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4   # MuJoCo 4.0.1 since 2026-11-21 (K7); dataset v5 was recorded on 3.8.1 (env lerobot)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN="systemd-inhibit --what=sleep:idle --why=blush_record systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0"
while [ "$P" -lt "$END" ]; do
  N=$(printf %02d "$P")
  D=~/.cache/huggingface/lerobot/${REPO_ID}_p$N
  if [ -d "$D" ]; then
    if python -c "import pyarrow.parquet as pq,glob,sys; [pq.ParquetFile(f) for f in glob.glob(sys.argv[1]+'/data/*/*.parquet')] or sys.exit(1)" "$D" 2>/dev/null
    then P=$((P + STRIDE)); continue; else rm -rf "$D"; fi
  fi
  cd "$SIM"
  PLAN=""
  if [ -n "$SEEDS_DIR" ]; then
    [ -f "$SEEDS_DIR/p$N.jsonl" ] || { echo "part $N: no seed plan $SEEDS_DIR/p$N.jsonl" >> "$LOGS/workers.log"; P=$((P + STRIDE)); continue; }
    PLAN="--seeds $SEEDS_DIR/p$N.jsonl"
  fi
  # shellcheck disable=SC2086
  $RUN python -u generate_truss.py $GEN_ARGS $PLAN --seed-start $((SEED_BASE + P * 1000)) --episodes 25 \
    --repo-id "${REPO_ID}_p$N" > "$LOGS/p$N.log" 2>&1
  echo "part $N exit $? at $(date +%H:%M): $(grep -c '✅ episode' "$LOGS/p$N.log") kept" >> "$LOGS/workers.log"
  P=$((P + STRIDE))
done
