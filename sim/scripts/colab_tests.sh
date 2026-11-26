#!/usr/bin/env bash
# colab_tests.sh: the laptop half of a Colab training run (K7 2026-11-22: train on Colab, a test every 5k steps).
# Colab (sim/colab_train_act.ipynb) pushes a checkpoint to a private Hub repo every 5k steps; this starts three
# systemd user units, all holding off sleep (a sleeping laptop also drops the Colab tab, which ends the session):
#   blush-<run>-pull     pull_hub_checkpoints.py: each new checkpoint's weights into outputs/train/<run>/
#   blush-<run>-alone    pick_checkpoint.py --watch: the policy alone, TEST_SCENES held-out scenes + 5 holds
#   blush-<run>-hybrid   the same scenes, the teacher steps in only when the policy is failing (only with
#                        KINDS="alone hybrid": K7 2026-11-23, policy-only tests from now on)
# The testers are the same as train_and_test.sh's (so the tables compare with act_teacher_v5's), from FROM on.
#   sim/scripts/colab_tests.sh act_teacher_v5b64 krish5831/blush_sim_act_v5b64 [steps] [first tested step]
# A v6 run (dataset = v5's 800 re-made + 600 new, rig trio; NOT RUN YET):
#   EXTRA="--world crate23 --layout-rig base --start-guard --clip-actions" sim/scripts/colab_tests.sh act_teacher_v6b64 \
#       krish5831/blush_sim_act_v6b64
#   ⚠️ --world crate23: the dataset's world is v5's 23 x 28 cm crate, not the code default (truss_scene.WORLDS).
#   ⚠️ --layout-rig base: dataset v6's scenes are decided with the old camera bracket standing in them and shown with
#   the new ones (own_sim_measurements.md section 21), so the held-out lists (heldout_v5*.jsonl) keep their targets.
#   --start-guard / --clip-actions: the two guards outside the policy (post-mortem 5.9), opt-in.
#   The rig, the picture path and the encoder come from the checkpoint and its dataset by themselves.
# Tables: python3 sim/scripts/checkpoint_table.py <run> [--wait <step>]
# ⚠️ Never more than 3 simulator jobs at once (memory feedback_memory_cap): these are 2, so one slot is left.
set -euo pipefail
RUN=${1:?run name, e.g. act_teacher_v5b64}; HUB=${2:?Hub repo, e.g. krish5831/blush_sim_act_v5b64}
STEPS=${3:-100000}; FROM=${4:-5000}
SEEDS=${SEEDS:-heldout_v5.jsonl}; N=${TEST_SCENES:-25}
KINDS=${KINDS:-alone}; EXTRA=${EXTRA:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
U=blush-$(echo "$RUN" | tr '_' '-')
ENV="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4"   # MuJoCo 4.0.1 (K7 2026-11-21)
NOSLEEP=(systemd-inhibit --what=sleep:idle --why="B.L.U.S.H Colab run $RUN")
systemd-run --user --unit="$U-pull" --collect -q -p MemoryMax=2G \
  "${NOSLEEP[@]}" bash -c "$ENV && cd $REPO && python -u sim/scripts/pull_hub_checkpoints.py $HUB $RUN --until $STEPS \
    > outputs/pull_$RUN.log 2>&1"
COMMON="--headless --no-dr --no-yolo --n-action-steps 100 --seeds-from $SEEDS --scenes $N $EXTRA"
for KIND in $KINDS; do
  if [[ $KIND == alone ]]; then ARGS="$COMMON --holds 5"; else ARGS="$COMMON --hybrid --rejoin --trigger failing"; fi
  systemd-run --user --unit="$U-$KIND" --collect -q -p MemoryMax=7G -p MemorySwapMax=0 \
    "${NOSLEEP[@]}" bash -c "$ENV && cd $REPO/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && python -u pick_checkpoint.py \
      $REPO/outputs/train/$RUN --watch --from-step $FROM --until $STEPS --cpu -- $ARGS \
      > $REPO/outputs/watch_${RUN}_$KIND.log 2>&1"
done
echo "started $U-pull and the testers ($KINDS); logs in $REPO/outputs/{pull,watch}_${RUN}*.log"
