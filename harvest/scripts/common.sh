# common.sh: sourced by the harvest scripts. Loads the rig + pipeline config and builds
# the flag groups every lerobot command shares, so record, rollout and teleop can't drift apart.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../config/hardware.env"
source "$HERE/../harvest.env"

# ⚠️ Camera NAMES and order must match between recording and rollout, or the policy gets the
#    wrong image in each slot.
CAMERAS="{ birdseye: {type: opencv, index_or_path: $BIRDSEYE_CAM, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS}, claw: {type: opencv, index_or_path: $CLAW_CAM, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS} }"

ROBOT_FLAGS=(
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_PORT"
  --robot.id="$FOLLOWER_ID"
  --robot.use_degrees="$USE_DEGREES"
  --robot.cameras="$CAMERAS"
)
TELEOP_FLAGS=(
  --teleop.type=so101_leader
  --teleop.port="$LEADER_PORT"
  --teleop.id="$LEADER_ID"
  --teleop.use_degrees="$USE_DEGREES"
)

for p in "$FOLLOWER_PORT" "$LEADER_PORT"; do
  [[ -e "$p" ]] || { echo "⚠️ arm not found: $p (powered and plugged in?)"; exit 1; }
done
