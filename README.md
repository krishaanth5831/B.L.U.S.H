# B.L.U.S.H (Berry Localization Using Selective Harvesting)

Vinea's harvest module, built on an SO-101 leader/follower pair in a printed test
greenhouse. The arm looks at a cherry tomato vine, decides which fruits are ripe,
and picks **single ripe fruits** into a crate. It leaves everything else on the vine.

Everything for that lives here: the ripeness model and its data, the picking policy
and its recording plan, the eval logs, and the greenhouse CAD.

## The pipeline

```
birdseye cam ──► coarse: "red fruit over there" ──► arm moves to the approach pose
                                                             │
claw cam ──► ripeness gate (YOLO11n, per fruit) ─────────────┤
             unripe / turning / ripe + confidence            ▼
                                       pick target = largest confident `ripe` fruit
                                                             │  none → skip, log it
                                                             ▼
claw + birdseye + joint state ──► picking policy (ACT → SmolVLA) ──► fruit in crate
                                                             │
                                                             ▼
                                                  logs/harvest_log.csv
```

The gate decides **whether** to pick and the policy decides **how**. They are
trained, tested and debugged separately, so a policy mistake can never turn
into picking a green fruit.

## Layout

| Folder | What's in it |
|---|---|
| [`config/`](config/) | Shared hardware facts: arm ports, calibration ids, camera indices, codec. Every script reads these. |
| [`ripeness/`](ripeness/) | **Ripeness gate.** Capture, labelling guide, own claw-camera data, split-by-session build, YOLO training, threshold eval, `pick_target()`. |
| [`harvest/`](harvest/) | **Picking pipeline.** Episode plan, record / train / rollout scripts, dataset manifest, gated harvest loop, eval logs. |
| [`hardware/`](hardware/) | Greenhouse STLs + FeatureScript, the picking end effector, camera mounts. |
| [`envs/`](envs/) | Conda env for the ripeness model (kept apart from `lerobot`). |

## Where the data lives

| Data | Where | Why |
|---|---|---|
| Own claw-camera frames + YOLO labels + 6-stage sidecar | **this repo**, `ripeness/data/` | A few hundred JPGs, so plain git copes |
| Frozen test split | **this repo**, `ripeness/data/splits.yaml` | Must never change once made |
| Released ripeness weights (`best.pt`, a few MB) + results | **this repo**, `ripeness/models/` | Small, and they need to be versioned with the data they came from |
| Public pretraining sets (Laboro, Mendeley, tomatOD) | local only, gitignored, `ripeness/data/public/` | Two are non-commercial. The repo holds the download notes, not the images. |
| LeRobot episode datasets (video) | **Hugging Face Hub**, listed in [`harvest/datasets.md`](harvest/datasets.md) | Hundreds of MB of video. GitHub rejects files over 100 MB. |
| ACT / SmolVLA checkpoints | **Hugging Face Hub**, same manifest | Same reason |
| Training runs (`runs/`, `outputs/`) | local only, gitignored | Scratch |

⚠️ Git LFS is not installed on the laptop. If the claw-camera data outgrows plain
git, install `git-lfs` and move `ripeness/data/raw/**` to LFS then. Don't do it
half-way.

## Branches

- **`dev`**: work in progress. Push here.
- **`main`**: final, working code only. It's protected: no direct pushes, even
  for admins. Changes land by PR from `dev`, and you merge them yourself.

## Order of work

From the vault's *SO-ARM101 Training* plan (`03 Projects/Vinea/08 SOARM101 Training/`).
Each step's "done when" gates the next one.

1. Picking end effector + hanging-vine rig: [`hardware/end_effector/`](hardware/end_effector/), [`hardware/greenhouse/`](hardware/greenhouse/)
2. Ripeness gate: [`ripeness/`](ripeness/)
3. Record picking episodes: [`harvest/`](harvest/)
4. ACT baseline
5. SmolVLA fine-tune (rented GPU)
6. Gated end-to-end runs, success and false-pick rate measured
