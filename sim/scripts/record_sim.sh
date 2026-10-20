#!/usr/bin/env bash
# record_sim.sh: record teleop demos in the sim (lerobot env). Same flags as harvest/scripts/record.sh.
#   sim/scripts/record_sim.sh            first session (creates $SIM_DATASET)
#   sim/scripts/record_sim.sh --resume   later sessions (appends; every other flag must stay identical)
#
# Keys (terminal focus):  → end the episode early   ← discard + re-record it   Esc stop the session
# MuJoCo window:          N during the reset period = the next episode gets a new vine layout
# ⚠️ Not before the gates in the vault note "(C) BLUSH Sim — Before Each Step" (step 5).
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
need_leader
EXTRA=()
[[ "${1:-}" == "--resume" ]] && { EXTRA+=(--resume=true); shift; }
lerobot-record "${SIM_ROBOT_FLAGS[@]}" "${TELEOP_FLAGS[@]}" \
  --dataset.repo_id="$SIM_DATASET" \
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
  "${EXTRA[@]}" "$@"
# Local copy: ~/.cache/huggingface/lerobot/$SIM_DATASET. Push to the Hub after the last session.
