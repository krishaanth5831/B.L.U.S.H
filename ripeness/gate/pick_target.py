"""pick_target.py: the runtime ripeness gate. Claw frame in, pick target (or None) out.

    from ripeness.gate.pick_target import Gate
    gate = Gate()                 # reads ripeness/models/gate.yaml
    target = gate(frame)          # None = nothing safely ripe, don't pick

Run directly for a live view on the claw camera:
    python ripeness/gate/pick_target.py [camera_index]

Rule (vault note 02, step 8a):
  1. keep `ripe` boxes with conf >= t_ripe (t_ripe comes from eval_threshold.py, never set by eye)
  2. drop boxes touching the image edge (partly out of view, so ripeness is uncertain)
  3. of the rest, take the LARGEST box: usually the closest, least hidden fruit
"""
import pathlib
import sys

import yaml
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[2]
RIPE = 2   # class id, LABELLING.md


class Gate:
    def __init__(self, cfg_path=ROOT / "ripeness/models/gate.yaml"):
        cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
        if not cfg.get("weights") or cfg.get("t_ripe") is None:
            raise RuntimeError("gate.yaml has no released weights/t_ripe yet; see ripeness/models/README.md")
        self.model = YOLO(str(ROOT / cfg["weights"]))
        self.t_ripe = float(cfg["t_ripe"])
        self.edge = int(cfg.get("edge_px", 4))

    def __call__(self, frame):
        r = self.model.predict(frame, conf=0.05, verbose=False)[0]
        h, w = r.orig_shape
        cands = []
        for (x0, y0, x1, y1), c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(c) != RIPE or conf < self.t_ripe:
                continue
            if x0 < self.edge or y0 < self.edge or x1 > w - self.edge or y1 > h - self.edge:
                continue
            cands.append(((x1 - x0) * (y1 - y0), (x0, y0, x1, y1), conf))
        if not cands:
            return None
        _, box, conf = max(cands)
        return {"box": box, "conf": conf, "centre": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)}


if __name__ == "__main__":
    import cv2

    gate = Gate()
    cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 2)   # claw = 2, config/hardware.env
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = gate(frame)
        if t:
            x0, y0, x1, y1 = map(int, t["box"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, f"PICK {t['conf']:.2f}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "no safe ripe fruit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("ripeness gate (q quits)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
