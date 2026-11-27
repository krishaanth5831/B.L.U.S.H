#!/usr/bin/env bash
# remake_parts.sh: one worker of re-making dataset v5's 800 episodes with another rig's cameras, part by part
# (sim/remake_pictures.py part: the logged simulator state of every frame is set, the new cameras rendered, the joint
# readings and commands copied from the old dataset). Parts are v5's own 32 parts of 25 episodes.
# ⚠️ NOT RUN YET (2026-11-27): it writes a dataset, so it needs K7's go. Only `remake_pictures.py dry` has run.
# Run W workers as parts w, w+W, w+2W, ... (as record_parts.sh):
#   for w in 0 1 2; do nohup sim/scripts/remake_parts.sh krish5831/<name>_old $w 3 32 & done
#   python sim/merge_parts.py krish5831/<name> krish5831/<name>_old krish5831/<name>_new   # old + new, then the Hub
# A finished part is kept, one killed before its finalize (no parquet footer) is removed and redone, as
# record_parts.sh does. Needs ~/.cache/blush_sim/states (the only copy of the old episodes' states) and the old
# dataset's joint data (one 30 MB file, in the Hugging Face cache since the dry runs of 2026-11-26).
# ⚠️ RAM: not measured for a re-make part; the cap is the recorder's (7 GB a part). Start one worker, look at
# `systemctl --user status` for its peak, then the others; keep 8 GB free (`free -g`).
# ⚠️ v5 episode 353 (seed 10014003) does not rebuild as recorded and is left out (remake_pictures.py logs it), so
# its part has 24 episodes and the whole re-make 799. Check the count when all parts are done:
#   grep -h "re-made" outputs/remake_<name>_old/workers.log     # 31 parts of 25 and part 14 with 24
set -uo pipefail
REPO_ID=${1:?repo id of the re-made parts, e.g. krish5831/<name>_old}
P=${2:?first part}; STRIDE=${3:?workers}; END=${4:?parts (32 for all of v5)}
RIG=${RIG:-trio}
SIM=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$SIM/../outputs/remake_$(basename "$REPO_ID"); mkdir -p "$LOGS"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4   # MuJoCo 4.0.1 (K7 2026-11-21)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="systemd-inhibit --what=sleep:idle --why=blush_remake systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0"
while [ "$P" -lt "$END" ]; do
  N=$(printf %02d "$P")
  D=~/.cache/huggingface/lerobot/${REPO_ID}_p$N
  if [ -d "$D" ]; then
    if python -c "import pyarrow.parquet as pq,glob,sys; [pq.ParquetFile(f) for f in glob.glob(sys.argv[1]+'/data/*/*.parquet')] or sys.exit(1)" "$D" 2>/dev/null
    then P=$((P + STRIDE)); continue; else rm -rf "$D"; fi
  fi
  cd "$SIM"
  $RUN python -u remake_pictures.py part --part "$P" --repo-id "$REPO_ID" --rig "$RIG" > "$LOGS/p$N.log" 2>&1
  RC=$?
  echo "part $N exit $RC at $(date +%H:%M): $(grep -c '✅ episode' "$LOGS/p$N.log") re-made" >> "$LOGS/workers.log"
  # ⚠️ a part that stopped on an episode whose scene did not come back (exit != 0) is still finalized, with the
  # episodes before the stop, so a restart takes it for finished: say so here and in workers.log. Once the cause is
  # fixed, delete that part's folder ($D) and run the worker again.
  if [ "$RC" -ne 0 ]; then
    echo "⚠️ part $N STOPPED EARLY (exit $RC), its folder $D is short: $(grep -v '^$' "$LOGS/p$N.log" | tail -1 | cut -c1-200)" | tee -a "$LOGS/workers.log"
  fi
  P=$((P + STRIDE))
done
