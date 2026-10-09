#!/usr/bin/env bash
# teleop.sh: pre-flight teleop with both cameras in the rerun viewer. Ctrl+C to exit.
# Check before recording: wrist roll moves through its range with no runaway, both
# camera streams are live, and the claw camera cable is free at full range.
set -euo pipefail
source "$(dirname "$0")/common.sh"
lerobot-teleoperate "${ROBOT_FLAGS[@]}" "${TELEOP_FLAGS[@]}" --display_data=true
