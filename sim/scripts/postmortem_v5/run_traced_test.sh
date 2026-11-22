#!/usr/bin/env bash
# Post-mortem 2026-11-20: rollouts that test the picture finding (the policy was trained on frames decoded from
# video; a test gives it the raw render; the plan made at the start frame depends on the difference).
#   run_traced_test.sh <unit> <output folder> <seeds file> <scenes> <extra eval_truss flags...>
# One traced run per call, under the usual memory fuse. Never more than 3 simulator jobs at once: the caller decides.
set -u
REPO=/home/krishaanth/Desktop/personal_projects/github/vinea-so101/B.L.U.S.H
CK=$REPO/outputs/train/act_teacher_v5/checkpoints/100000/pretrained_model
unit=$1; out=$2; seeds=$3; scenes=$4; shift 4
mkdir -p "$out"; rm -f "$out"/done
systemd-run --user --unit="$unit" --collect -q --wait -p MemoryMax=7G -p MemorySwapMax=0 \
  bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd $REPO/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && python -u eval_truss.py $CK --headless --no-dr --no-yolo --n-action-steps 100 --seeds-from $seeds --scenes $scenes --log-dir $out --trace $out/trace.jsonl $* > $out/run.log 2>&1"
touch "$out/done"
echo "$(date +%H:%M) $unit done: $(grep -E 'successes' "$out/run.log" | cut -c1-200)"
