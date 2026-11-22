#!/usr/bin/env bash
# One standard 100-scene test of the 100k checkpoint with a changed inference (post-mortem 2026-11-20), as the
# earlier variant tests were run: held-out scenes heldout_v5.jsonl, log in outputs/eval (checkpoint_table.py reads it).
#   run_std_test.sh alone  <name> <extra eval_truss flags...>    policy only, with the 20 nothing-ripe scenes, traced
#   run_std_test.sh hybrid <name> <extra eval_truss flags...>    the teacher steps in only when the policy is failing
# ⚠️ Never more than 3 simulator jobs at once: the caller counts.
set -u
REPO=/home/krishaanth/Desktop/personal_projects/github/vinea-so101/B.L.U.S.H
CK=$REPO/outputs/train/act_teacher_v5/checkpoints/100000/pretrained_model
kind=$1; name=$2; shift 2
if [[ $kind == alone ]]; then
  mkdir -p "$REPO/outputs/diag/runs/$name"
  ARGS="--holds 20 --trace $REPO/outputs/diag/runs/$name/trace.jsonl"
else
  ARGS="--hybrid --rejoin --trigger failing"
fi
LOG=$REPO/outputs/full_act_teacher_v5_${kind}_100000_$name.log
systemd-run --user --unit="blush-act-teacher-v5-full-$kind-100000-$name" --collect -q --wait -p MemoryMax=7G -p MemorySwapMax=0 \
  bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd $REPO/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && python -u eval_truss.py $CK --headless --no-dr --no-yolo --n-action-steps 100 --seeds-from heldout_v5.jsonl --scenes 100 $ARGS $* > $LOG 2>&1"
echo "$(date +%H:%M) $kind $name done: $(grep -E 'successes' "$LOG" | cut -c1-220)"
