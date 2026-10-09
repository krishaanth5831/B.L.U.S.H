# Ripeness gate

For **every cherry tomato in the claw camera's view**, find it and call it `unripe`,
`turning` or `ripe`. The arm only picks fruits called `ripe` with enough confidence.

The birdseye camera only steers the arm. At 640×480 through a 130° lens a cherry
tomato is a few pixels wide there, which is too small to judge ripeness. The claw
camera, close to the vine, makes the call.

## The error that matters: precision on `ripe`

| Model says | Truth | Cost |
|---|---|---|
| not ripe | ripe | Missed pick. The fruit is still there next pass. **Cheap.** |
| ripe | unripe | Wasted fruit, and lost grower trust. **Expensive.** |

Tune for precision on `ripe` and accept lower recall to get it. The pick threshold
comes from the measured precision–recall curve on the frozen test set, never by eye.

## Layout

```
ripeness/
├── LABELLING.md            6 colour stages → 3 classes, labelling rules
├── data/
│   ├── own/images/<session>/   claw-camera frames, one folder per capture session  (committed)
│   ├── own/labels/<session>/   YOLO .txt per frame, same names                      (committed)
│   ├── stages.csv              6-stage label per box (sidecar)                       (committed)
│   ├── splits.yaml             which sessions are val / FROZEN test                 (committed)
│   ├── public/                 Laboro, Mendeley, tomatOD: download notes only       (gitignored)
│   └── yolo/                   image lists + dataset yamls, built by build_split.py  (gitignored)
├── scripts/                capture → convert → remap → split → train → eval
├── gate/pick_target.py     the runtime gate: frame in, pick target (or None) out
└── models/                 released weights, gate.yaml (threshold), results.csv
```

## Steps

All in the `ripeness` env (`conda env create -f envs/ripeness.yml`). Run from the repo root.

| # | Do | With |
|---|---|---|
| 1 | Check both cameras: max resolution, MJPG, manual WB/exposure, claw-cam pixels per fruit (aim ≥ 30 px) | `scripts/camera_check.sh` |
| 2 | Lock white balance + exposure, add the gripper LED. ⚠️ Capture, label and run with the same settings. | `scripts/camera_check.sh --lock` |
| 3 | Get vines with green, turning **and** red fruit (supermarket vines are all red, so precision can't be measured on them) | ask a grower |
| 4 | Capture 300–500 claw frames, several short sessions, one thing changed per session | `scripts/capture_claw.py` |
| 5 | Label in Label Studio / CVAT, export YOLO, fill `stages.csv` | `LABELLING.md` |
| 6 | Download + convert + remap public sets | `scripts/convert_coco.py`, `scripts/remap_classes.py` |
| 7 | Freeze the test sessions in `data/splits.yaml`, build the lists | `scripts/build_split.py` |
| 8 | Pretrain on public, fine-tune on own + public | `scripts/train.sh` |
| 9 | Precision/recall vs threshold on the test set, set `t_ripe` | `scripts/eval_threshold.py` |
| 10 | Release: copy `best.pt` into `models/`, write `gate.yaml` + a `results.csv` row | `models/README.md` |
| 11 | Run the gate live | `gate/pick_target.py` |

## Done when

- The frozen test set has a precision–recall curve, and the chosen threshold meets the
  precision target. The target is decided **before** looking at the curve.
- `pick_target()` runs live on the claw camera at the approach pose and picks a
  sensible fruit on 10 different vine setups.
- The "how does the policy know which fruit" decision is written in `harvest/README.md`.

⚠️ Every script here was seeded from the vault notes and **has not been run yet**.
