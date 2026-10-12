"""eval_negatives.py: does a model still find tomatoes, and does it leave non-tomatoes alone?

    python ripeness/scripts/eval_negatives.py ripeness/models/pretrain_laboro_v1.pt ripeness/models/pretrain_v2.pt

For each weights file:
  (a) Laboro val ONLY: per-class P / R / mAP50. Compare with the pretrain's numbers in models/README.md.
      ⚠️ Not the training-time val: that mixes in negatives, which drag precision down, so its
         numbers can't be compared with a model trained without them.
  (b) negatives_oi/images/test (never trained or tuned on): share of non-tomato images with ANY
      box at conf >= 0.25 and >= 0.5, plus the count of boxes called `ripe`. Lower is better.

Neither says anything about our claw camera. The real check is own claw frames of red/yellow/green
non-tomatoes (data/own/README.md) and live_view.py.
"""
import pathlib
import sys

import yaml
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[2]
YOLO_DIR = ROOT / "ripeness/data/yolo"
NEG_TEST = ROOT / "ripeness/data/public/negatives_oi/images/test"
LABORO_VAL = ROOT / "ripeness/data/public/laboro_yolo/images/val"
RUNS = ROOT / "runs/ripeness"   # ⚠️ absolute, see train.sh
RIPE = 2                        # class id, LABELLING.md

(YOLO_DIR / "laboro_val.txt").write_text("".join(f"{p}\n" for p in sorted(LABORO_VAL.glob("*.jpg"))))
(YOLO_DIR / "laboro_val.yaml").write_text(yaml.safe_dump(
    {"path": str(YOLO_DIR), "train": "laboro_val.txt", "val": "laboro_val.txt",
     "names": {0: "unripe", 1: "turning", 2: "ripe"}}, sort_keys=False))
negs = sorted(NEG_TEST.glob("*.jpg"))
assert negs, f"{NEG_TEST} is empty: run prepare_negatives.py"

for w in sys.argv[1:]:
    model = YOLO(w)
    # ⚠️ workers=0: with dataloader workers this val hung for 20+ min at 0 % CPU (2026-10-12).
    m = model.val(data=str(YOLO_DIR / "laboro_val.yaml"), split="val", imgsz=640, batch=16, plots=False, workers=0,
                  verbose=False, project=str(RUNS), name="eval_laboro_val", exist_ok=True)
    print(f"\n=== {w}\n(a) Laboro val ({len(list(LABORO_VAL.glob('*.jpg')))} images)")
    print(f"    {'class':8s}  P     R     mAP50")
    print(f"    {'all':8s}  {m.box.mp:.2f}  {m.box.mr:.2f}  {m.box.map50:.2f}")
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, _ = m.box.class_result(i)
        print(f"    {model.names[int(c)]:8s}  {p:.2f}  {r:.2f}  {ap50:.2f}")

    hit25 = hit50 = ripe50 = 0
    for img in negs:
        b = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0].boxes
        hit25 += len(b) > 0
        hit50 += bool((b.conf >= 0.5).any())
        ripe50 += int(((b.conf >= 0.5) & (b.cls == RIPE)).sum())
    print(f"(b) {len(negs)} held-out non-tomato images")
    print(f"    any box conf>=0.25: {hit25:3d} ({hit25 / len(negs):.0%})")
    print(f"    any box conf>=0.50: {hit50:3d} ({hit50 / len(negs):.0%})")
    print(f"    `ripe` boxes conf>=0.50: {ripe50}")
