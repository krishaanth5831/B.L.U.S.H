"""capture_claw.py: save claw-camera frames for labelling. SPACE saves a frame, q quits.

Run from the repo root with the camera_check.sh --lock settings applied:
    python ripeness/scripts/capture_claw.py [camera_index]

Each run is one session: ripeness/data/own/images/sYYYYMMDD_HHMM/. Sessions are the
unit of the train/val/test split, so start a new run whenever the vine, day or angle changes.
"""
import pathlib
import sys
import time

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 2   # claw camera = index 2 (config/hardware.env); recheck per boot
W, H = 640, 480                                       # match what the gate runs on, so no resize at inference

SESSION = time.strftime("s%Y%m%d_%H%M")
OUT = ROOT / "ripeness/data/own/images" / SESSION
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # ⚠️ YUYV + 2 cams can starve the USB bus
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed: wrong index, or the camera is held by another process")
        break
    cv2.imshow(f"claw  [{SESSION}]  saved={n}  SPACE=save  q=quit", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        path = OUT / f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(path), frame)
        n += 1
        print("saved", path.relative_to(ROOT))
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
print(f"{n} frames in {OUT.relative_to(ROOT)}. Log the session in ripeness/data/own/README.md")
