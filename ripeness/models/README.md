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
