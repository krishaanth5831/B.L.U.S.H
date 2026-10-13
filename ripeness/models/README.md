# Released ripeness models

Only **released** weights live here (YOLO11n `best.pt` is a few MB). Training runs stay
in `runs/ripeness/` (gitignored).

To release:

1. `cp runs/ripeness/finetune_vN/weights/best.pt ripeness/models/ripeness_vN.pt`
2. Add a row to `results.csv`: which sessions trained it, which test sessions, which
   public sets, the precision target (decided before the curve), the chosen `t_ripe`,
   and the precision and recall at that threshold.
3. Point `gate.yaml` at the new weights and threshold.

⚠️ `commercial_ok` is `false` if Laboro or tomatOD were in training. Those weights are
for the demo only and must never ship in a product.

## Pretrain checkpoints (fine-tune starting points, NOT gates)

These are never pointed at by `gate.yaml`. They exist so `train.sh finetune` has a
fixed, versioned starting point.

### `pretrain_laboro_v1.pt` (2026-10-09)

- YOLO11n from `yolo11n.pt`, `train.sh pretrain`: imgsz 640, batch 16, hsv_h 0.005,
  ultralytics 8.4.174, RTX 1000 Ada laptop. 100 epochs in 9 min, best at epoch 75.
- Data: Laboro Tomato only, cherry + regular tomatoes, 643 train / 161 val (Laboro's own
  test split), via `prepare_laboro.py`. **commercial_ok: false** (CC BY-NC-SA).
- Val (Laboro test split, NOT our camera, so it says nothing about the SO-101 yet):

  | Class | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | all | 0.86 | 0.77 | 0.86 | 0.71 |
  | unripe | 0.91 | 0.79 | 0.90 | 0.72 |
  | turning | 0.79 | 0.74 | 0.81 | 0.69 |
  | ripe | 0.88 | 0.77 | 0.86 | 0.73 |

- Per-epoch curves: `pretrain_laboro_v1_curves.csv`.
- Spot check on an unseen Mendeley photo (orange cherry variety): fruit found, green →
  `unripe`, but orange fruit → `turning` rather than `ripe`. Laboro's "fully ripened" is deep
  red, so other varieties read as less ripe. That's the safe direction, and it's what fine-tuning
  on our own vines has to fix.

### `pretrain_v2.pt` (2026-10-12): v1 + non-tomato backgrounds

Why: v1 boxed almost anything red, yellow/orange or green as a tomato (balls, peppers, citrus,
apples). Laboro has a tomato in nearly every frame, so colour alone was enough to score well.

- Same recipe as v1 (`train.sh pretrain v2`, from `yolo11n.pt`, imgsz 640, batch 16, hsv_h 0.005,
  ultralytics 8.4.174). 100 epochs in ~25 min, best at epoch 88. Per-epoch curves: `pretrain_v2_curves.csv`.
- Data: Laboro (643 train / 161 val) + `negatives_oi` (299 train / 72 val background images, empty labels,
  `prepare_negatives.py` as of commit 852f5d7). 3 classes. **commercial_ok: false** (Laboro).
- `eval_negatives.py ripeness/models/pretrain_laboro_v1.pt ripeness/models/pretrain_v2.pt`:

  | | v1 | v2 |
  |---|---|---|
  | Laboro val mAP50 (all) | 0.86 | 0.86 |
  | Laboro val P / R `unripe` | 0.91 / 0.80 | 0.90 / 0.83 |
  | Laboro val P / R `turning` | 0.79 / 0.74 | 0.84 / 0.71 |
  | Laboro val P / R `ripe` | 0.87 / 0.78 | 0.80 / 0.80 |
  | 312 held-out non-tomato images with any box, conf ≥ 0.25 | 214 (69 %) | 21 (7 %) |
  | … with any box, conf ≥ 0.50 | 179 (57 %) | 13 (4 %) |
  | `ripe` boxes at conf ≥ 0.50 on them | 314 | 16 |

  P / R above are at Ultralytics' max-F1 confidence, which differs per model. `ripe` on Laboro val at FIXED
  thresholds (eval_threshold.py's matcher), precision / recall:

  | t | v1 | v2 |
  |---|---|---|
  | 0.5 | 0.82 / 0.79 | 0.79 / 0.85 |
  | 0.7 | 0.88 / 0.69 | 0.86 / 0.73 |
  | 0.8 | 0.91 / 0.62 | 0.91 / 0.64 |

  v2 is more confident on ripe fruit, so at a given threshold it calls more of it, with slightly lower
  precision at low t. At equal precision (t 0.8) it has equal or better recall. None of this sets `t_ripe`.
  That still comes from `eval_threshold.py` on own frames.
- What v2 still boxes: fruit hanging among leaves (mangoes, peaches, apples and citrus on trees, grape
  bunches, rose hips) and baubles on Christmas trees. Those are the scenes closest to a vine. Own claw-camera
  negatives sessions (`LABELLING.md` → "Non-tomato sessions") are what fixes them for our camera.

### `pretrain_v4.pt` (2026-10-13): 4 classes, look-alikes boxed as `not_tomato`. Default for live_view + finetune

Why: the user wants to SEE what the model thinks a look-alike is, not just get no box. Class 3 `not_tomato`
is appended, so ids 0-2 and the gate (`RIPE = 2`) are unchanged. No box still means "not a tomato" too.

- Same recipe (`train.sh pretrain v4`, from `yolo11n.pt`). Per-epoch curves: `pretrain_v4_curves.csv`.
- Data: Laboro (643 / 161) + `negatives_oi` with Open Images' own boxes as `not_tomato`: 633 train / 146 val
  images (`prepare_negatives.py`, second draw added). Train/val images with group-of (pile) boxes, or drawn
  for a boxed class but left with no box, are dropped, so a pile isn't background in one image and
  `not_tomato` in the next. The 312 held-out test images are exactly v2's. **commercial_ok: false** (Laboro).
- `eval_negatives.py` (v3 = first 4-class run on v2's 299 train images, NOT released; its numbers are why
  the data was grown):

  | | v2 | v3 | **v4** |
  |---|---|---|---|
  | Laboro val mAP50 (tomato classes) | 0.86 | 0.84 | 0.85 |
  | Laboro val P / R `unripe` | 0.90 / 0.83 | 0.89 / 0.81 | 0.90 / 0.79 |
  | Laboro val P / R `turning` | 0.84 / 0.71 | 0.80 / 0.70 | 0.79 / 0.76 |
  | Laboro val P / R `ripe` | 0.80 / 0.80 | 0.78 / 0.83 | 0.87 / 0.77 |
  | `not_tomato` boxes on Laboro val (real tomatoes), conf ≥ 0.25 / 0.5 | n/a | 0 / 0 | 2 / 0 |
  | Held-out look-alikes with any TOMATO box, conf ≥ 0.25 | 7 % | 4 % | 2 % |
  | … conf ≥ 0.5 | 4 % | 3 % | 1 % |
  | `ripe` boxes on them, conf ≥ 0.5 | 16 | 4 | 6 |
  | `not_tomato` P / R / mAP50, all 312 test | n/a | 0.47 / 0.28 / 0.27 | 0.31 / 0.44 / 0.24 |
  | `not_tomato` P / R / mAP50, 195 test without piles | n/a | 0.64 / 0.35 / 0.37 | 0.73 / 0.47 / 0.52 |

  ⚠️ On all 312, `not_tomato` precision is understated: test keeps its pile images, whose piles have no
  labels, so v4's boxes on single fruit in a pile score as wrong. The pile-free row is the fair one. Open
  Images labels are incomplete everywhere, so P is a lower bound in both rows.
- `ripe` on Laboro val at fixed thresholds (`eval_threshold.py <weights> ripeness/data/yolo/laboro_val.txt`),
  precision / recall:

  | t | v2 | v3 | v4 |
  |---|---|---|---|
  | 0.5 | 0.79 / 0.85 | 0.79 / 0.84 | 0.87 / 0.73 |
  | 0.7 | 0.86 / 0.73 | 0.85 / 0.75 | 0.93 / 0.60 |
  | 0.8 | 0.91 / 0.64 | 0.88 / 0.67 | 0.95 / 0.50 |

  v4 is more cautious about `ripe`: higher precision, lower recall at the same t. At equal precision
  (v4 t 0.5 vs v2 t 0.7) recall is about equal. Precision is the error that matters (../README.md).
- Acceptance rule, written before v4's numbers: replace v3 only if tomato mAP50 ≥ 0.84 and `not_tomato`
  recall clearly beats 0.28. Met (0.85, 0.44).
- `not_tomato` recall is still under half, so many look-alikes get no box at all (still the right answer
  for the gate). `live_view.py 2 ripeness/models/pretrain_v4.pt 0.15` shows weaker calls. Own claw-cam
  look-alike sessions, boxed as class 3, are the fix for our camera.
