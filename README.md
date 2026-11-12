# B.L.U.S.H (Berry Localization Using Selective Harvesting)

Vinea's harvest module, built on an SO-101 leader/follower pair in a laser-cut test
greenhouse. The arm looks at a cherry tomato vine, decides what is ready, harvests it
into a crate and leaves everything else on the vine.

What it harvests changed on 2026-11-03 (K7): from **single ripe fruits** to **whole
trusses**. One close of the jaw cuts the truss's stalk with blades at the fingertips,
spring pads hold it, and the arm carries the whole truss to the crate. The truss work
runs in the MuJoCo sim ([`sim/`](sim/)) so far. The ripeness gate, the real-rig scripts
in `harvest/` and the diagram below still describe single-fruit picking.

Everything for that lives here: the ripeness model and its data, the picking policy
and its recording plan, the sim, the eval logs, and the greenhouse CAD.

## Watch it

```bash
conda activate lerobot && cd sim
python generate_truss.py --dry --watch --seed-start 200007 --episodes 1   # MuJoCo window: the scripted arm harvests one truss
```

More in [`sim/README.md`](sim/README.md#watch-it), including a trained policy in the sim (none right now: all
were deleted on 2026-11-08 and ACT is to be retrained).

## The pipeline

```
birdseye cam ──► ripeness gate SELECTS (YOLO, recall-tuned) ──► target box
                 unripe / turning / ripe / not_tomato              │  none → skip, log it
                                                                   ▼
             target box drawn onto the birdseye frames (overlay, fixed for the episode)
                                                                   │
claw + birdseye (with overlay) + joint state ──► picking policy (ACT; π0.5 planned)
                                                                   │
claw cam ──► ripeness gate CONFIRMS just before the close ─────────┤
             (precision-tuned)   not ripe → abort, log claw_veto   ▼
                                                             fruit in crate
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
| [`sim/`](sim/) | **MuJoCo sim.** The SO-101 as a lerobot robot plugin, truss scenes, scripted expert, episode generator, DAgger, policy eval, training recipes (`sim/scripts/train_sim.sh`). |
| [`hardware/`](hardware/) | Greenhouse laser-cut generator + STLs + FeatureScript, the fin-ray end effector, camera mounts. |
| [`envs/`](envs/) | Conda env for the ripeness model (kept apart from `lerobot`). |

## Where the data lives

| Data | Where | Why |
|---|---|---|
| Own claw-camera frames + YOLO labels + 6-stage sidecar | **this repo**, `ripeness/data/` | A few hundred JPGs, so plain git copes |
| Frozen test split | **this repo**, `ripeness/data/splits.yaml` | Must never change once made |
| Released ripeness weights (`best.pt`, a few MB) + results | **this repo**, `ripeness/models/` | Small, and they need to be versioned with the data they came from |
| Public pretraining sets (Laboro, Mendeley, tomatOD) | local only, gitignored, `ripeness/data/public/` | Two are non-commercial. The repo holds the download notes, not the images. |
| LeRobot episode datasets (video) | **Hugging Face Hub**, listed in [`harvest/datasets.md`](harvest/datasets.md) | Hundreds of MB of video. GitHub rejects files over 100 MB. |
| Sim episode datasets (`blush_sim_truss_v2`) | `~/.cache/huggingface/lerobot/krish5831/` + the Hub, same manifest | Same reason |
| Policy checkpoints | `outputs/train/` (local) or the Hub, same manifest | Same reason |
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
3. Record picking episodes: [`harvest/`](harvest/) on the real rig; scripted truss episodes in [`sim/`](sim/)
4. ACT, trained on the laptop. In the sim this is the next run (`sim/scripts/train_sim.sh act-truss` on
   `blush_sim_truss_v2`, not started); SmolVLA was tried and removed on 2026-11-08
5. The original plan's π0.5 fine-tune from `lerobot/pi05_base` (rented GPU or the hosted π API, since the
   laptop's 6 GB GPU can't run it). Not started
6. Gated end-to-end runs, success and false-pick rate measured
