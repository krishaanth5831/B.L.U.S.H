#!/usr/bin/env python3
"""Truss YOLO on the FROZEN test split: per-class P/R/mAP, and precision/recall of truss_ready per threshold.

    python sim/yolo_truss/eval.py <best.pt> <truss.yaml>          (ripeness env)

The error that matters is the same as the ripeness gate's (ripeness/README.md): calling a truss ready that
isn't (it gets cut too early) costs more than missing a ready one (it's still there next pass). So the
threshold is read off this table for a precision target picked BEFORE looking at it.
"""
import pathlib
import sys

import numpy as np
from ultralytics import YOLO

IOU_MATCH = 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main():
    weights, data = sys.argv[1], sys.argv[2]
    model = YOLO(weights)
    r = model.val(data=data, split="test", imgsz=640, batch=8, verbose=False, plots=False)
    print("test split, per class:")
    for i, c in enumerate(r.box.ap_class_index):
        p, rec, ap50, ap = r.box.class_result(i)
        print(f"  {model.names[int(c)]:16s} P {p:.3f}  R {rec:.3f}  mAP50 {ap50:.3f}  mAP50-95 {ap:.3f}")
    # truss_ready: precision / recall vs confidence threshold, matched to truth boxes at IoU 0.5
    root = pathlib.Path(data).parent
    imgs = sorted((root / "images" / "test").glob("*.jpg"))
    rows = []                      # (conf, is_true_positive), plus the number of true ready boxes
    n_true = 0
    # ⚠️ one frame at a time: a list of 300 paths went through as one batch and ran the 6 GB GPU out of memory
    for res in (r for p in imgs for r in model.predict(str(p), conf=0.05, verbose=False, save=False)):
        p = pathlib.Path(res.path)
        H, W = res.orig_shape
        truth = []
        for line in (root / "labels" / "test" / f"{p.stem}.txt").read_text().split("\n"):
            if line.strip() and int(line.split()[0]) == 0:
                _, cx, cy, w, h = map(float, line.split())
                truth.append((cx * W - w * W / 2, cy * H - h * H / 2, cx * W + w * W / 2, cy * H + h * H / 2))
        n_true += len(truth)
        used = set()
        dets = [(float(c), b.tolist()) for b, k, c in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf) if int(k) == 0]
        for conf, b in sorted(dets, reverse=True):
            best = max(((iou(b, t), j) for j, t in enumerate(truth) if j not in used), default=(0, None))
            tp = best[0] >= IOU_MATCH
            if tp:
                used.add(best[1])
            rows.append((conf, tp))
    rows.sort(reverse=True)
    print(f"\ntruss_ready on {len(imgs)} test frames ({n_true} ready trusses):  threshold  precision  recall")
    for t in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sel = [tp for c, tp in rows if c >= t]
        tp = sum(sel)
        print(f"  {t:.2f}  {tp / max(len(sel), 1):.3f}  {tp / max(n_true, 1):.3f}")


if __name__ == "__main__":
    main()
