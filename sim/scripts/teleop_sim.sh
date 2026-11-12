#!/usr/bin/env bash
# teleop_sim.sh: the real leader drives the sim follower. Like ~/teleop.sh, with the MuJoCo arm.
#   sim/scripts/teleop_sim.sh                MuJoCo window + rerun with the birdseye + claw footage (a truss scene)
#   RIG=base sim/scripts/teleop_sim.sh       the base + claw cameras instead
#   SEED=7600000 sim/scripts/teleop_sim.sh   a specific truss layout
#   OVERLAY=none sim/scripts/teleop_sim.sh   no target box (default: the truss YOLO picks it if trained, else the
#                                            true box; the true box fills in when YOLO misses)
# MuJoCo window: N = new layout, Ctrl + right-drag = pull a tomato. Ctrl+C here stops.
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
need_leader
lerobot-teleoperate "${SIM_ROBOT_FLAGS[@]}" "${TELEOP_FLAGS[@]}" \
  --fps="$CAM_FPS" \
  --display_data=true \
  "$@"
