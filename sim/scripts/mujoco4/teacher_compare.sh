#!/usr/bin/env bash
# MuJoCo 4.0.1 port (K7, 2026-11-21: "we will do everything with mujoco 4 from now on", "use 4.0.1").
# The teacher alone on the same 100 scenes it scored 86 on under MuJoCo 3.8.1 (post-mortem runs `teacher` = the first
# 50 of heldout_v5.jsonl: 44/50, and `train_teacher` = train_clean_50.jsonl: 42/50; both from run_diag_batch1.sh), with
# the same flags, in the env lerobot-mj4 (a clone of lerobot with mujoco==4.0.1). Results go to
# outputs/diag/runs/<name>_mj4/ (log + per-frame trace); compare with compare_versions.py.
# ⚠️ Two simulator jobs at once; with anything else running keep the total at 3 or fewer.
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd)     # the repo root
CK=$R/outputs/train/act_teacher_v5/checkpoints/100000/pretrained_model   # loaded but not used: the teacher drives
BASE="--headless --no-dr --no-yolo --n-action-steps 100"
run() {  # name, seeds file, scenes
  local name=$1 seeds=$2 n=$3
  local out=$R/outputs/diag/runs/${name}_mj4; mkdir -p $out
  systemd-run --user --unit="blush-mj4-$name" --collect -q --wait -p MemoryMax=7G -p MemorySwapMax=0 \
    bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4 && cd $R/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && python -c 'import mujoco; assert mujoco.__version__ == \"4.0.1\", mujoco.__version__' && python -u eval_truss.py $CK $BASE --seeds-from $seeds --scenes $n --log-dir $out --trace $out/trace.jsonl --hybrid --rejoin --trigger failing --teacher-drives > $out/run.log 2>&1"
  echo "$(date +%H:%M) ${name}_mj4: $(grep -E 'successes' $out/run.log | cut -c1-200)"
}
run teacher heldout_v5.jsonl 50 &
run train_teacher $R/sim/scripts/postmortem_v5/train_clean_50.jsonl 50 &
wait
