#!/usr/bin/env bash
# teleop_sim.sh: the real leader drives the sim follower. Like ~/teleop.sh, with the MuJoCo arm.
#   sim/scripts/teleop_sim.sh                MuJoCo window + rerun with the claw + birdseye footage
#   SEED=7 sim/scripts/teleop_sim.sh         a specific vine layout
#   OVERLAY=none sim/scripts/teleop_sim.sh   no target box (default: YOLO picks it, true box fills in)
# MuJoCo window: N = new layout, Ctrl + right-drag = pull a tomato. Ctrl+C here stops.
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
need_leader
lerobot-teleoperate "${SIM_ROBOT_FLAGS[@]}" "${TELEOP_FLAGS[@]}" \
  --fps="$CAM_FPS" \
  --display_data=true \
  "$@"
