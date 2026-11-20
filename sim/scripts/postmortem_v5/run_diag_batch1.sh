#!/usr/bin/env bash
# Post-mortem batch 1 (2026-11-19 night), one eval after another on the GPU. Policy: act_teacher_v5 100k.
# Every run writes its result log + a per-frame trace into outputs/diag/runs/<name>/ (never outputs/eval).
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd)     # the repo root
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd $R/sim; export MUJOCO_GL=egl OMP_NUM_THREADS=2
CK=$R/outputs/train/act_teacher_v5/checkpoints/100000/pretrained_model
BASE="--headless --no-dr --no-yolo --n-action-steps 100"
run() {  # name, seeds file, scenes, extra args...
  local name=$1 seeds=$2 n=$3; shift 3
  local out=$R/outputs/diag/runs/$name; mkdir -p $out
  if ls $out/*.jsonl >/dev/null 2>&1 && grep -q '"summary": true' $out/act_*.jsonl 2>/dev/null; then echo "skip $name (done)"; return; fi
  echo "=== $name $(date +%H:%M:%S)"
  python -u eval_truss.py $CK $BASE --seeds-from $seeds --scenes $n --log-dir $out --trace $out/trace.jsonl "$@" > $out/run.log 2>&1
  grep -E "successes" $out/run.log | cut -c1-200
}
H=heldout_v5.jsonl; T=$R/sim/scripts/postmortem_v5/train_clean_50.jsonl
run base_a        $H 50
run teacher       $H 50 --hybrid --rejoin --trigger failing --teacher-drives
run cue_both_u30  $H 25 --cue-shift 30 0
run img_base_u30  $H 25 --img-shift base 30 0
run img_claw_u30  $H 25 --img-shift claw 30 0
run cue_num_u30   $H 25 --cue-shift 30 0 --cue-shift-what numbers
run cue_ovl_u30   $H 25 --cue-shift 30 0 --cue-shift-what overlay
run train_clean   $T 50
run train_teacher $T 50 --hybrid --rejoin --trigger failing --teacher-drives
run cue_both_u100 $H 25 --cue-shift 100 0
run cue_both_v-30 $H 25 --cue-shift 0 -30
run base_b        $H 50
echo "=== batch 1 done $(date +%H:%M:%S)"
