#!/usr/bin/env bash
# train_and_test.sh: ACT training (train_sim.sh act-teacher) plus a test of every checkpoint while it trains
# (K7 2026-11-19: "every 5k steps after 20k steps run tests to see how the policy is performing").
#   sim/scripts/train_and_test.sh <run> <dataset> [steps] [first tested step]
#   sim/scripts/train_and_test.sh act_teacher_v5 krish5831/blush_sim_teacher_v5 100000 20000
# Starts three systemd user units (they survive a Claude Code restart; the laptop doesn't sleep while training):
#   blush-<run>         training on the GPU, a checkpoint every 5k (12 GB RAM fuse; step 3 peaked at 8.4 GB)
#   blush-<run>-alone   pick_checkpoint.py --watch on the CPU: the policy alone, TEST_SCENES held-out scenes + 5 holds
#   blush-<run>-hybrid  the same scenes, the policy drives and teacher v5 steps in only when it is failing
#                       (--hybrid --rejoin --trigger failing, rescue.Gate; K7 2026-11-19: "the teacher should only
#                       jump in if the robot is failing ... basically like dagger")
# ONLY=hybrid (or alone) starts just that tester for a run already training.
# Both watchers keep going after training ends until the last checkpoint is tested. Rankings:
# outputs/eval/<run>_ranking.json; logs: outputs/train_<run>.log, outputs/watch_<run>_{alone,hybrid}.log.
# ⚠️ A test of 20 scenes + 5 holds took ~40 min on the CPU (step 3, 2026-11-14), the same as the gap between two
# checkpoints (~0.47 s per step), so TEST_SCENES stays small; the best checkpoints get all of heldout_v5.jsonl after.
# ⚠️ RAM (K7 2026-11-18): training ~8.4 GB + two tests ~3-4 GB each; check `free -g` first, keep 8 GB free.
set -euo pipefail
RUN=${1:?run name, e.g. act_teacher_v5}; DS=${2:?dataset, e.g. krish5831/blush_sim_teacher_v5}
STEPS=${3:-100000}; FROM=${4:-20000}
SEEDS=${SEEDS:-heldout_v5.jsonl}; N=${TEST_SCENES:-25}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
U=blush-$(echo "$RUN" | tr '_' '-')
ENV="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4"   # MuJoCo 4.0.1 since 2026-11-21 (K7); act_teacher_v5 ran on 3.8.1 (env lerobot)
if [[ -z ${ONLY:-} ]]; then
  [[ -e $REPO/outputs/train/$RUN ]] && { echo "$REPO/outputs/train/$RUN exists"; exit 1; }
  systemd-run --user --unit="$U" --collect -q -p MemoryMax=12G -p MemorySwapMax=0 \
    systemd-inhibit --what=sleep:idle --why="B.L.U.S.H training $RUN" \
    bash -c "$ENV && cd $REPO && env RUN=$RUN STEPS=$STEPS SAVE=5000 sim/scripts/train_sim.sh act-teacher $DS \
      > outputs/train_$RUN.log 2>&1"
fi
COMMON="--headless --no-dr --no-yolo --n-action-steps 100 --seeds-from $SEEDS --scenes $N"
for KIND in ${ONLY:-alone hybrid}; do
  if [[ $KIND == alone ]]; then ARGS="$COMMON --holds 5"; else ARGS="$COMMON --hybrid --rejoin --trigger failing"; fi
  systemd-run --user --unit="$U-$KIND" --collect -q -p MemoryMax=7G -p MemorySwapMax=0 \
    bash -c "$ENV && cd $REPO/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && python -u pick_checkpoint.py \
      $REPO/outputs/train/$RUN --watch --from-step $FROM --until $STEPS --cpu -- $ARGS \
      > $REPO/outputs/watch_${RUN}_$KIND.log 2>&1"
done
echo "started ${ONLY:+only }$( [[ -z ${ONLY:-} ]] && echo "$U, ")$(for K in ${ONLY:-alone hybrid}; do printf "%s " "$U-$K"; done); logs in $REPO/outputs/"
