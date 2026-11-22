# B.L.U.S.H (Berry Localization Using Selective Harvesting)

Vinea's harvest module, built on an SO-101 leader/follower arm pair. The arm looks at a cherry tomato vine,
decides what is ready, harvests it into a crate and leaves everything else on the vine.

What it harvests changed on 2026-11-03 (K7): from **single ripe fruits** to **whole trusses**. One close of the
jaw cuts the truss's stalk with blades at the fingertips, spring pads hold it, and the arm carries the whole truss
to the crate. The truss work runs in the MuJoCo sim ([`sim/`](sim/)) so far. The ripeness gate, the real-rig scripts
in `harvest/` and the real-rig diagram below still describe single-fruit picking.

Everything for that lives here: the ripeness model and its data, the picking policy and its recording plan, the
sim, the eval logs, and the test-rig CAD.

Source-available, not open source: see [License](#license).

## Status (2026-11-22)

| Part | State |
|---|---|
| Sim: truss scenes, cut-and-hold fingers, scripted teacher | ✅ working; the teacher alone solves 86 of 100 held-out scenes on MuJoCo 3.8.1, 87 on 4.0.1 |
| Sim datasets | ✅ teacher v5 dataset, 800 episodes, on the Hub (private); list in [`harvest/datasets.md`](harvest/datasets.md) |
| Picking policy (ACT, trained in the sim) | ⚠️ `act_teacher_v5`: 19/100 alone, 60/100 with the teacher stepping in. Post-mortem in progress ([`sim/scripts/postmortem_v5/`](sim/scripts/postmortem_v5/README.md)); nothing trains or records until it is done |
| Simulator | MuJoCo 4.0.1 since 2026-11-21 (env `lerobot-mj4`); everything before that ran on 3.8.1 |
| Ripeness gate (YOLO, real camera) | pretrained on public data only; no own claw-camera frames captured yet ([`ripeness/`](ripeness/README.md)) |
| Real-rig recording / rollout scripts | written, **not yet run on the real arm** ([`harvest/`](harvest/README.md)) |
| Cut-and-hold fingers | sim only; no printable CAD yet. The printable fin-ray gripper is for single fruit |
| Test rig | wooden pipe-rail rig for two arms, designed ([`hardware/greenhouse/pipe_rail_rig/`](hardware/greenhouse/pipe_rail_rig/README.md)) |

## Setup

Linux laptop, conda (miniforge). Three envs, because they pin different torch / opencv / MuJoCo versions:

| Env | For |
|---|---|
| `lerobot-mj4` | the sim, recording, training, rollout. lerobot (editable checkout) + MuJoCo **4.0.1** (never 4.0.0, it has a solver bug) |
| `lerobot` | MuJoCo 3.8.1. Only for the v5 post-mortem scripts in `sim/scripts/postmortem_v5/`. Don't upgrade it in place |
| `ripeness` | YOLO (ultralytics) for the ripeness gate and the sim's YOLO worker |

How `lerobot-mj4` was made on the laptop (a clone of the `lerobot` env, which has lerobot installed editable from
`~/Desktop/personal_projects/github/lerobot`):

```bash
conda create -n lerobot-mj4 --clone lerobot
conda activate lerobot-mj4
pip uninstall -y mujoco && pip install mujoco==4.0.1    # the 4.0 release notes ask for the uninstall first
pip install -e sim/plugin                               # registers --robot.type=so101_mujoco for every lerobot command
sim/fetch_so101.sh                                      # SO-101 MJCF + meshes into sim/assets/so101 (gitignored)
conda env create -f envs/ripeness.yml                   # the ripeness env
```

The real arms are addressed by `/dev/serial/by-id/...` and calibration ids `follower_white` / `leader_black`, all in
[`config/hardware.env`](config/hardware.env). Several sim checks read `follower_white.json` from lerobot's
calibration folder (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`).

## Watch it

Every line opens a MuJoCo window. Run from `sim/` in `lerobot-mj4`:

```bash
conda activate lerobot-mj4 && cd sim
python view_scene.py --cams              # a truss scene + the two policy cameras; move the arm with the Control sliders
python check_truss_cut.py --windowed     # one stalk across the fingertips: close, cut, then shake it
python generate_truss.py --dry --watch --seed-start 200007 --episodes 1   # the scripted arm harvests one truss
python check_teacher.py solve --scenes 1 --seed-start 9100003 --watch     # the teacher does one scene
```

With the real leader arm plugged in, it drives the sim arm (nothing is recorded):

```bash
python view_scene.py --leader --cams     # from sim/
sim/scripts/teleop_sim.sh                # from the repo root: lerobot-teleoperate, MuJoCo window + rerun viewer
```

A trained policy in the sim, and everything else: [`sim/README.md`](sim/README.md#watch-it).

## Checks

```bash
~/miniforge3/envs/lerobot-mj4/bin/ruff check .   # lint, whole repo (ruff.toml); 0 findings expected
cd sim
python check_tomato_release.py   # fruit tethers hang, release above threshold, fall free
python check_scene.py            # unit mapping, truss layouts, full scene, cameras (needs follower_white.json)
python check_truss_cut.py        # the blades cut 3-5 mm stalks and the pads hold the truss while shaking
python check_plugin.py           # the plugin through lerobot: features, timing, an AV1 episode, rerun (--no-yolo skips the YOLO pass)
DRYRUN=1 scripts/train_sim.sh act-smoke   # any sim/scripts wrapper: print the lerobot command instead of running it
```

Each check prints PASS/FAIL per line and writes nothing to the repo. Known failures on MuJoCo 4.0.1 (2026-11-22):
`check_scene.py`'s gripper line (the calibration's open stop is 116.8°, the MJCF's 117.6°) and `check_truss_cut.py`'s
5 mm stalk under the violent shake (it pulls about 5 % harder on 4.0.1); `check_plugin.py`'s timing lines depend on
machine load and often fail on the truss scene. Details in [`sim/README.md`](sim/README.md#checks).
There is no pytest suite.

## The real-rig pipeline (planned, single-fruit era)

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

The gate decides **whether** to pick and the policy decides **how**. They are trained, tested and debugged
separately, so a policy mistake can never turn into picking a green fruit. In the sim the same split holds for
trusses: a truss YOLO (`sim/yolo_truss/`) picks the ready truss and its cut point on the first frame, and the
policy gets that box and cut point as numbers.

## Layout

| Folder | What's in it |
|---|---|
| [`config/`](config/) | Shared hardware facts: arm ports, calibration ids, camera indices, codec. Every script reads these. |
| [`ripeness/`](ripeness/) | **Ripeness gate.** Capture, labelling guide, own claw-camera data, split-by-session build, YOLO training, threshold eval, `pick_target()`. |
| [`harvest/`](harvest/) | **Real-rig picking pipeline.** Episode plan, record / train / rollout scripts, the Hub dataset manifest, gated harvest loop, eval logs. |
| [`sim/`](sim/) | **MuJoCo sim.** The SO-101 as a lerobot robot plugin, truss scenes, scripted expert and teacher, episode generator, DAgger, policy eval, training recipes, post-mortem diagnostics. |
| [`hardware/`](hardware/) | The pipe-rail test rig (current), the earlier printed and laser-cut greenhouses, the fin-ray end effector, camera mounts. |
| [`envs/`](envs/) | Conda env file for the ripeness model. |

## Where the data lives

| Data | Where | Why |
|---|---|---|
| Own claw-camera frames + YOLO labels + 6-stage sidecar (none captured yet) | **this repo**, `ripeness/data/` | A few hundred JPGs, so plain git copes |
| Frozen test split | **this repo**, `ripeness/data/splits.yaml` | Must never change once made |
| Released ripeness weights (a few MB) + results | **this repo**, `ripeness/models/` | Small, and they need to be versioned with the data they came from |
| Public pretraining sets (Laboro, Mendeley, tomatOD) | local only, gitignored, `ripeness/data/public/` | Two are non-commercial. The repo holds the download notes, not the images. |
| LeRobot episode datasets (video), sim and real | **Hugging Face Hub**, private, listed in [`harvest/datasets.md`](harvest/datasets.md); lerobot caches what it needs in `~/.cache/huggingface/lerobot/krish5831/` | Hundreds of MB of video. GitHub rejects files over 100 MB. |
| Policy checkpoints | `outputs/train/` (local, gitignored) or the Hub, same manifest | Same reason |
| Training runs, eval logs, notes (`runs/`, `outputs/`), sim attempt and state logs (`~/.cache/blush_sim/`) | local only, gitignored | Scratch |

⚠️ Git LFS is not installed on the laptop. If the claw-camera data outgrows plain git, install `git-lfs` and move
`ripeness/data/raw/**` to LFS then. Don't do it half-way.

## Branches

- **`dev`**: work in progress. Push here.
- **`main`**: final, working code only. It's protected: no direct pushes, even for admins. Changes land by PR from
  `dev`, and only the owner merges.

## Order of work

From the vault's BLUSH roadmap (`03 Projects/Vinea/08 Modules/BLUSH/`). Each step's "done when" gates the next one.

1. Picking end effector + test rig: [`hardware/end_effector/`](hardware/end_effector/), [`hardware/greenhouse/`](hardware/greenhouse/)
2. Ripeness gate: [`ripeness/`](ripeness/)
3. Record picking episodes: [`harvest/`](harvest/) on the real rig; teacher episodes in [`sim/`](sim/) (dataset v5 done)
4. ACT, trained on the laptop. In the sim: `act_teacher_v5` trained, post-mortem in progress before the next
   dataset; SmolVLA was tried and removed on 2026-11-08
5. The original plan's π0.5 fine-tune from `lerobot/pi05_base` (rented GPU or the hosted π API, since the
   laptop's 6 GB GPU can't run it). Not started
6. Gated end-to-end runs, success and false-pick rate measured

## License

Copyright (c) 2026 Krishaanth Ramaraj. Source-available under the [PolyForm Strict License 1.0.0](LICENSE): you may
read and run it for noncommercial purposes, but not change or redistribute it. Commercial use, modification or
redistribution needs a separate written license. Third-party files keep their own licenses (see
[`sim/NOTICE.md`](sim/NOTICE.md) and `hardware/end_effector/finray/stock/NOTICE.md`).
