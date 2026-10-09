"""eval_threshold.py: precision / recall of the `ripe` call vs confidence threshold,
on the FROZEN test set only.

    python ripeness/scripts/eval_threshold.py runs/ripeness/finetune_v1/weights/best.pt

Decide the precision target BEFORE reading this output (e.g. "at most 1 in 20 picks not
ripe"), then take the LOWEST threshold that meets it. Put that in models/gate.yaml.

This is a simplified matcher: one prediction can match a box another prediction already
matched. That's good enough to pick a threshold. Use `yolo detect val ... split=test` for mAP.
"""
import pathlib
import sys

import numpy as np
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEST_LIST = ROOT / "ripeness/data/yolo/test.txt"
RIPE = 2          # class id, LABELLING.md
IOU_MATCH = 0.5   # standard detection match threshold


def load_gt(lbl, w, h):
    if not lbl.exists():
        return []
    out = []
    for line in lbl.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        out.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


model = YOLO(sys.argv[1])
imgs = [pathlib.Path(p) for p in TEST_LIST.read_text().split()]
assert imgs, "test.txt is empty: set `test:` sessions in data/splits.yaml and run build_split.py"

preds, n_true_ripe = [], 0   # preds: (conf, correct) for every `ripe` prediction
for img in imgs:
    r = model.predict(str(img), conf=0.05, verbose=False)[0]
    h, w = r.orig_shape
    lbl = pathlib.Path("/labels/".join(img.as_posix().rsplit("/images/", 1))).with_suffix(".txt")
    gt = load_gt(lbl, w, h)
    n_true_ripe += sum(g[0] == RIPE for g in gt)
    for box, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        if int(c) != RIPE:
            continue
        best = max(gt, key=lambda g: iou(box, g[1:]), default=None)
        preds.append((conf, best is not None and iou(box, best[1:]) >= IOU_MATCH and best[0] == RIPE))

print(f"{len(imgs)} test images, {n_true_ripe} truly ripe fruits")
for t in np.arange(0.30, 0.96, 0.05):
    hits = [ok for conf, ok in preds if conf >= t]
    prec = sum(hits) / len(hits) if hits else float("nan")
    rec = sum(hits) / n_true_ripe if n_true_ripe else float("nan")
    print(f"t={t:.2f}  precision={prec:.3f}  recall={rec:.3f}  n_called_ripe={len(hits)}")
