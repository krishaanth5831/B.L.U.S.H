#!/usr/bin/env bash
# replay_sim.sh: play a recorded episode's actions back on the sim arm (checks the data, no leader).
#   sim/scripts/replay_sim.sh <episode> [dataset]     default dataset: $SIM_DATASET
# Works on real-arm datasets too: same units, so a real episode replays on the sim arm.
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
EP=${1:?episode index}; DS=${2:-$SIM_DATASET}
lerobot-replay "${SIM_ROBOT_FLAGS[@]}" --robot.snap_to_first_action=true \
  --dataset.repo_id="$DS" --dataset.episode="$EP"
