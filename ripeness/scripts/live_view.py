"""live_view.py: run any ripeness weights on a live camera and draw every class. A sanity
check of model + camera, NOT the gate (the gate only picks `ripe` above t_ripe, see gate/pick_target.py).

Run from the repo root in the `ripeness` env:
    python ripeness/scripts/live_view.py [camera] [weights] [conf]
      camera   index (2) or a /dev/v4l/by-path/... path. Default 2
      weights  default ripeness/models/pretrain_laboro_v1.pt
      conf     lowest confidence drawn. Default 0.25
Keys: s saves the annotated frame to runs/ripeness/live_view/ (gitignored), q quits.
"""
import pathlib
import sys
import time

import cv2
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAM = sys.argv[1] if len(sys.argv) > 1 else "2"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "ripeness/models/pretrain_laboro_v1.pt")
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
W, H = 640, 480                       # same as capture_claw.py and the gate
COLOURS = {0: (0, 200, 0), 1: (0, 165, 255), 2: (0, 0, 255)}   # BGR: unripe green, turning orange, ripe red
OUT = ROOT / "runs/ripeness/live_view"

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(int(CAM) if CAM.isdigit() else CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # ⚠️ YUYV is slow on this camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

t_prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed: wrong index/path, or the camera is held by another process")
        break
    r = model.predict(frame, conf=CONF, verbose=False)[0]
    for (x0, y0, x1, y1), c, conf in zip(r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        cv2.rectangle(frame, (x0, y0), (x1, y1), COLOURS[c], 2)
        cv2.putText(frame, f"{model.names[c]} {conf:.2f}", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOURS[c], 2)
    t = time.time()
    cv2.putText(frame, f"{1 / max(t - t_prev, 1e-6):.0f} fps  conf>={CONF}", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    t_prev = t
    cv2.imshow(f"ripeness live view  [{pathlib.Path(WEIGHTS).name}]  s=save q=quit", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("s"):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{int(t * 1000)}.jpg"
        cv2.imwrite(str(path), frame)
        print("saved", path.relative_to(ROOT))
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
