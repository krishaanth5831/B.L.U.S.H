#!/usr/bin/env bash
# camera_check.sh: what can the claw + birdseye cameras do, and lock their settings.
#   ripeness/scripts/camera_check.sh          list devices, formats and controls
#   ripeness/scripts/camera_check.sh --lock   lock WB + exposure on the claw camera
# Needs v4l-utils (sudo apt install v4l-utils).
set -euo pipefail
source "$(dirname "$0")/../../config/hardware.env"

# Control names checked on the InnoMaker U20CAM-1080P, 2026-10-09 (v4l2-ctl --list-ctrls-menus).
# Write the final numbers here AND in ripeness/data/own/README.md once the gripper LED is fitted,
# because that light, not the room, sets the exposure.
WB_TEMP=4600        # K. Camera default; range 2800-6500. Starting point, not tuned.
EXPOSURE=150        # exposure_time_absolute, range 1-5000. In room light 150 gave mean brightness 98/255
                    # vs 112 on auto (2026-10-09), so it's a sane start. 300 -> 139, 600 -> 178.
AUTO_EXPOSURE_MANUAL=1   # this camera's menu: 1 = Manual, 3 = Aperture Priority (its default)

if [[ "${1:-}" == "--lock" ]]; then
  v4l2-ctl -d /dev/video"$CLAW_CAM" -c white_balance_automatic=0 -c white_balance_temperature=$WB_TEMP
  v4l2-ctl -d /dev/video"$CLAW_CAM" -c auto_exposure=$AUTO_EXPOSURE_MANUAL -c exposure_time_absolute=$EXPOSURE
  # ⚠️ Ships with exposure_dynamic_framerate=1, which lets fps drop in dim light. Pin it to 30 fps.
  v4l2-ctl -d /dev/video"$CLAW_CAM" -c exposure_dynamic_framerate=0
  v4l2-ctl -d /dev/video"$CLAW_CAM" --get-ctrl=white_balance_automatic,auto_exposure,exposure_dynamic_framerate
  # ⚠️ UVC cameras usually reset these on replug (not yet checked on this one). Re-run --lock every session.
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
