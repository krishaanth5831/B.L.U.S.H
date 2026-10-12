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
  `prepare_negatives.py`). Still 3 classes, no `not_tomato` class. **commercial_ok: false** (Laboro).
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
