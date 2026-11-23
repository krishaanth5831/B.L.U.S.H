#!/usr/bin/env bash
# test_all_checkpoints.sh: the full test of every checkpoint of a finished run (K7 2026-11-23: "stop training and run
# 100 scene tests on each check point", "do it both on my cpu and gpu"). The physics runs on the CPU (MuJoCo has no GPU
# path), the pictures are drawn on the GPU (EGL), and the policy runs on the GPU too (no --cpu), so the CPU cores are
# left to the physics.
#   sim/scripts/test_all_checkpoints.sh act_teacher_v5b64 070000 065000 ... 005000
# Splits the given checkpoints round-robin over WORKERS systemd user units (default 3: memory feedback_memory_cap,
# never more than 3 simulator jobs at once). Each unit tests its share in the order given, in two passes:
#   1. policy alone: SCENES held-out scenes (heldout_v5.jsonl) + HOLDS nothing-ripe scenes
#   2. hybrid: the same scenes, the teacher steps in only when the policy is failing
# Logs: outputs/full_<run>_w<i>_{alone,hybrid}.log. Tables: python3 sim/scripts/checkpoint_table.py <run> --scenes 100
set -euo pipefail
RUN=${1:?run name, e.g. act_teacher_v5b64}; shift
CKS=("$@"); [[ ${#CKS[@]} -gt 0 ]] || { echo "give the checkpoints, e.g. 070000 065000"; exit 1; }
WORKERS=${WORKERS:-3}; SCENES=${SCENES:-100}; HOLDS=${HOLDS:-20}; KINDS=${KINDS:-alone hybrid}   # KINDS=hybrid: that pass only
REPO=$(cd "$(dirname "$0")/../.." && pwd)
U=blush-$(echo "$RUN" | tr '_' '-')-full${TAG:+-$TAG}   # TAG: a second batch of units next to a first
ENV="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4"   # MuJoCo 4.0.1 (K7 2026-11-21)
COMMON="--headless --no-dr --no-yolo --n-action-steps 100 --seeds-from heldout_v5.jsonl --scenes $SCENES"
for ((w = 0; w < WORKERS; w++)); do
  mine=()
  for ((i = w; i < ${#CKS[@]}; i += WORKERS)); do mine+=("${CKS[i]}"); done
  [[ ${#mine[@]} -gt 0 ]] || continue
  # one pick_checkpoint call per checkpoint, in the order given (it would otherwise go oldest first)
  PC="python -u pick_checkpoint.py $REPO/outputs/train/$RUN --only \$ck --"
  systemd-run --user --unit="$U-w$w" --collect -q -p MemoryMax=7G -p MemorySwapMax=0 \
    systemd-inhibit --what=sleep:idle --why="B.L.U.S.H full tests $RUN" \
    bash -c "$ENV && cd $REPO/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && \
      if [[ \" $KINDS \" == *' alone '* ]]; then for ck in ${mine[*]}; do $PC $COMMON --holds $HOLDS >> $REPO/outputs/full_${RUN}_w${w}_alone.log 2>&1; done; fi; \
      if [[ \" $KINDS \" == *' hybrid '* ]]; then for ck in ${mine[*]}; do $PC $COMMON --hybrid --rejoin --trigger failing >> $REPO/outputs/full_${RUN}_w${w}_hybrid.log 2>&1; done; fi"
  echo "$U-w$w: ${mine[*]}"
done
