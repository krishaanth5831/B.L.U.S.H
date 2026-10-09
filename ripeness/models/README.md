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
