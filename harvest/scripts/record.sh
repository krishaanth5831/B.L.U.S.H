#!/usr/bin/env bash
# record.sh: record one session of teleop picking demonstrations (lerobot env).
#   harvest/scripts/record.sh            first session (creates the dataset)
#   harvest/scripts/record.sh --resume   later sessions (appends; every other flag must stay identical)
#
# Keys while recording:  → end the episode early   ← discard + re-record it   Esc stop the session
# Follow harvest/plan/episode_plan.csv row by row, and log the session in harvest/logs/session_log.md.
set -euo pipefail
source "$(dirname "$0")/common.sh"

EXTRA=()
[[ "${1:-}" == "--resume" ]] && EXTRA+=(--resume=true)

# Flags checked against lerobot commit 5aa74557 (vault note 03). After a `git pull` in the
# lerobot checkout, re-check with `lerobot-record --help`.
lerobot-record "${ROBOT_FLAGS[@]}" "${TELEOP_FLAGS[@]}" \
  --dataset.repo_id="$DATASET" \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$SESSION_EPISODES" \
  --dataset.episode_time_s="$EPISODE_S" \
  --dataset.reset_time_s="$RESET_S" \
  --dataset.rgb_encoder.vcodec="$VCODEC" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --display_data=true \
  "${EXTRA[@]}"
# Local copy: ~/.cache/huggingface/lerobot/$DATASET. Push to the Hub after the last session.
