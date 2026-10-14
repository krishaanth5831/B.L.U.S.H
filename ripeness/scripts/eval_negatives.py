"""eval_negatives.py: does a model still find tomatoes, and does it tell look-alikes apart from them?

    python ripeness/scripts/eval_negatives.py ripeness/models/pretrain_v2.pt ripeness/models/pretrain_v3.pt

For each weights file (3-class v1/v2 or 4-class v3+):
  (a) Laboro val ONLY: per-class P / R / mAP50. Compare with the numbers in models/README.md.
      ⚠️ Not the training-time val: that mixes in the look-alike images, so its numbers can't be
         compared with a model trained without them.
      4-class models: how many `not_tomato` boxes land on Laboro val. Laboro has only tomatoes and
      leaves, so every one of them is a real tomato (or leaf) called "not a tomato".
  (b) negatives_oi/images/test (never trained or tuned on): share of look-alike images with ANY
      TOMATO-class box (0-2) at conf >= 0.25 and >= 0.5, and the count of `ripe` boxes. Lower is
      better. Class-wise NMS, so a tomato box counts even if a `not_tomato` box sits on the same object.
      4-class models: `not_tomato` P / R / mAP50 against the Open Images boxes, on all of test and on the
      images without fruit piles (test_no_piles.txt). Piles are unlabelled, so the first understates P.
      Open Images labels are incomplete either way, so P is a lower bound.

Neither says anything about our claw camera. The real check is own claw frames of red/yellow/green
look-alikes (LABELLING.md → "Non-tomato sessions") and live_view.py.
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
RIPE, NOT_TOMATO = 2, 3         # class ids, LABELLING.md
TOMATO = (0, 1, 2)


def val(model, name, images):
    # Names come from the model, so 3-class weights validate without an nc mismatch.
    (YOLO_DIR / f"{name}.txt").write_text("".join(f"{p}\n" for p in images))
    (YOLO_DIR / f"{name}.yaml").write_text(yaml.safe_dump(
        {"path": str(YOLO_DIR), "train": f"{name}.txt", "val": f"{name}.txt", "names": dict(model.names)}, sort_keys=False))
    # ⚠️ workers=0: with dataloader workers this val hung for 20+ min at 0 % CPU (2026-10-12).
    return model.val(data=str(YOLO_DIR / f"{name}.yaml"), split="val", imgsz=640, batch=16, plots=False, workers=0,
                     verbose=False, project=str(RUNS), name=f"eval_{name}", exist_ok=True)


def table(model, m, classes):
    print(f"    {'class':10s}  P     R     mAP50")
    for i, c in enumerate(m.box.ap_class_index):
        if int(c) in classes:
            p, r, ap50, _ = m.box.class_result(i)
            print(f"    {model.names[int(c)]:10s}  {p:.2f}  {r:.2f}  {ap50:.2f}")


laboro = sorted(LABORO_VAL.glob("*.jpg"))
negs = sorted(NEG_TEST.glob("*.jpg"))
assert negs, f"{NEG_TEST} is empty: run prepare_negatives.py"

for w in sys.argv[1:]:
    model = YOLO(w)
    four = NOT_TOMATO in model.names
    m = val(model, "laboro_val", laboro)
    print(f"\n=== {w}\n(a) Laboro val ({len(laboro)} images)   mAP50 all tomato classes: "
          f"{sum(m.box.class_result(i)[2] for i, c in enumerate(m.box.ap_class_index) if int(c) in TOMATO) / 3:.2f}")
    table(model, m, TOMATO)
    if four:
        n25 = n50 = 0
        for img in laboro:
            b = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0].boxes
            n25 += int((b.cls == NOT_TOMATO).sum())
            n50 += int(((b.cls == NOT_TOMATO) & (b.conf >= 0.5)).sum())
        print(f"    `not_tomato` boxes on real tomatoes/leaves: {n25} at conf>=0.25, {n50} at conf>=0.50")

    hit25 = hit50 = ripe50 = 0
    for img in negs:
        b = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0].boxes
        tom = b.cls < NOT_TOMATO
        hit25 += bool(tom.any())
        hit50 += bool((tom & (b.conf >= 0.5)).any())
        ripe50 += int(((b.cls == RIPE) & (b.conf >= 0.5)).sum())
    print(f"(b) {len(negs)} held-out look-alike images")
    print(f"    any tomato box conf>=0.25: {hit25:3d} ({hit25 / len(negs):.0%})")
    print(f"    any tomato box conf>=0.50: {hit50:3d} ({hit50 / len(negs):.0%})")
    print(f"    `ripe` boxes conf>=0.50: {ripe50}")
    if four:
        table(model, val(model, "negatives_test", negs), (NOT_TOMATO,))
        no_piles = set((NEG_TEST.parents[1] / "test_no_piles.txt").read_text().split())
        print(f"    on the {len(no_piles)} test images without fruit piles:")
        table(model, val(model, "negatives_test_no_piles", [p for p in negs if p.stem in no_piles]), (NOT_TOMATO,))
