#!/usr/bin/env bash
# rollout.sh: run a trained policy on the follower (lerobot env).
#   harvest/scripts/rollout.sh eval <policy_repo> <eval_name>   episodic: 20 saved episodes with resets, for the eval grid
#   harvest/scripts/rollout.sh run  <policy_repo>               one 40 s run, nothing saved
#
# ⚠️ Start with the arm in the rest pose: with no teleop, each reset returns to the startup joints.
# Keep a hand near the follower's power switch for the first trials.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODE=${1:?eval|run}; POLICY=${2:?policy repo, e.g. \$ACT_POLICY}
SEATBELT=--robot.max_relative_target=10   # caps per-step joint jump for first runs. Remove once it behaves (it caps speed).
EXTRA=()
# SmolVLA: uncomment both, with the SAME rename map used in training.
# EXTRA+=(--inference.type=rtc --rename_map='...')

case "$MODE" in
  eval)
    lerobot-rollout --strategy.type=episodic "${ROBOT_FLAGS[@]}" $SEATBELT \
      --policy.path="$POLICY" --task="$TASK" \
      --dataset.repo_id="$HF_USER/rollout_${3:?eval name, e.g. act_v1_40k}" --dataset.no_stamp=true \
      --dataset.single_task="$TASK" --dataset.num_episodes=20 \
      --dataset.episode_time_s=40 --dataset.reset_time_s=40 \
      --dataset.rgb_encoder.vcodec="$VCODEC" --dataset.push_to_hub=false \
      --display_data=true "${EXTRA[@]}" ;;
  run)
    lerobot-rollout --strategy.type=base "${ROBOT_FLAGS[@]}" $SEATBELT \
      --policy.path="$POLICY" --task="$TASK" --duration=40 "${EXTRA[@]}" ;;
  *) echo "usage: $0 eval <policy> <name> | run <policy>"; exit 1 ;;
esac
