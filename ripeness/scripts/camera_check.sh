#!/usr/bin/env bash
# camera_check.sh: what can the claw + birdseye cameras do, and lock their settings.
#   ripeness/scripts/camera_check.sh          list devices, formats and controls
#   ripeness/scripts/camera_check.sh --lock   lock WB + exposure on the claw camera
# Needs v4l-utils (sudo apt install v4l-utils).
set -euo pipefail
source "$(dirname "$0")/../../config/hardware.env"

# Values from the vault's ripeness note, step 2. ⚠️ Control names and values differ per camera.
# Use what --list-ctrls prints, then write the final numbers here AND in ripeness/data/own/README.md.
WB_TEMP=4600        # K, unverified starting point
EXPOSURE=150        # exposure_time_absolute units (100 µs on most UVC cams), unverified
AUTO_EXPOSURE_MANUAL=1   # 1 = manual on most UVC cams

if [[ "${1:-}" == "--lock" ]]; then
  v4l2-ctl -d /dev/video"$CLAW_CAM" -c white_balance_automatic=0 -c white_balance_temperature=$WB_TEMP
  v4l2-ctl -d /dev/video"$CLAW_CAM" -c auto_exposure=$AUTO_EXPOSURE_MANUAL -c exposure_time_absolute=$EXPOSURE
  v4l2-ctl -d /dev/video"$CLAW_CAM" --get-ctrl=white_balance_automatic,auto_exposure
  exit 0
fi

v4l2-ctl --list-devices
for cam in "$CLAW_CAM" "$BIRDSEYE_CAM"; do
  echo "=== /dev/video$cam"
  # Check: max resolution (the birdseye is a "1080P" part, so 1920x1080 may be on offer),
  # and MJPG. ⚠️ Two cameras in raw YUYV on one USB bus drop frames.
  v4l2-ctl -d /dev/video"$cam" --list-formats-ext
  # Check: white_balance_automatic / auto_exposure exist, so step 2 can lock them.
  v4l2-ctl -d /dev/video"$cam" --list-ctrls
done
