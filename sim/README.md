# sim/: leader arm → MuJoCo follower

The real leader arm drives a simulated SO-101 with the fin-ray gripper in the mini greenhouse. The sim
cameras record the episodes as **co-training data for π0.5**, next to the real-arm episodes. Plan and
reasoning: vault `08 SOARM101 Training/(C) BLUSH Sim Teleop Plan.md`.

## Watch it

Run in the `lerobot` env (`conda activate lerobot`), from `sim/`:

```bash
python view_scene.py --cams              # greenhouse window + claw | birdseye images (magenta = true target box)
python eval_yolo_recall.py --show        # the recall score: per layout, YOLO's boxes vs the true target (key = next)
python view_scene.py --leader --cams     # the real leader arm drives the sim arm (read only, records nothing)
python check_tomato_release.py --windowed   # one truss: double-click a tomato, Ctrl + right-drag to pluck it
```

Without `--leader`, move the arm with the actuator sliders in the viewer's right panel (Control).
Ctrl + right-drag pulls a tomato by hand; plucks are printed in the terminal. Wayland prints
`libdecor` / `GLFWError: window position` warnings, which are harmless.

## lerobot commands against the sim (`--robot.type=so101_mujoco`)

The sim is a lerobot robot plugin (`plugin/`), so the normal lerobot tools run on it. The wrappers in
`scripts/` mirror `harvest/scripts/` flag for flag (same TASK, timings, codec, camera names):

| Sim | Real equivalent | What |
|---|---|---|
| `scripts/teleop_sim.sh` | `~/teleop.sh` | leader → sim arm, MuJoCo window + rerun with both cameras |
| `scripts/record_sim.sh [--resume]` | `harvest/scripts/record.sh` | teleop demos into `$SIM_DATASET` |
| `scripts/replay_sim.sh <ep> [ds]` | `lerobot-replay` | play an episode back on the sim arm (real datasets too) |
| `scripts/train_sim.sh act-smoke\|act-full\|merge\|pi05-smoke\|pi05-full` | `harvest/scripts/train_*.sh` | train on sim, or merge sim + real for co-training |
| `scripts/rollout_sim.sh eval\|run\|pi05 <policy>` | `harvest/scripts/rollout.sh` | run a policy on the sim arm |

The scripts pass `--robot.overlay=yolo`: on each scene's first birdseye frame the ripeness YOLO picks
the target box, which then stays put for the scene, like the real start-frame box. When YOLO misses
or picks another fruit, the sim's true box fills in, so no episode is lost. The terminal says which
(`scene N target box: YOLO box ...` / `true box (yolo missed)`), and `session.json` in the state log
keeps every box. `OVERLAY=none` gives raw frames.

`DRYRUN=1 scripts/<script>.sh ...` prints the lerobot command instead of running it. MuJoCo window:
**N** = new vine layout (press it in lerobot-record's reset period), Ctrl + right-drag = pull a tomato.
The sim reports `robot_type=so_follower` like the real arm, because `lerobot-edit-dataset merge`
refuses mixed robot types. Sim and real are told apart by the dataset name.
Full sim state per frame goes to `~/.cache/blush_sim/states/<session>/` for re-renders.

## Setup (once)

```bash
./fetch_so101.sh                  # SO-101 MJCF + meshes into assets/so101 (gitignored, ~16 MB)
pip install -e plugin             # registers --robot.type=so101_mujoco in the lerobot env
python build_finray_meshes.py     # only after changing the fin-ray design; needs numpy shapely trimesh manifold3d
```

`assets/finray/` is committed: the printed fin-ray parts as metre meshes in the MJCF body frames.

## Checks

```bash
python check_tomato_release.py    # step 1: the tether hangs, releases above threshold, falls free
python check_scene.py             # step 2: unit mapping, 200 layouts, cameras, grasp + pluck through contact
python check_plugin.py            # the plugin through lerobot: features, 30 fps, AV1 episode, rerun, YOLO overlay
python eval_yolo_recall.py        # step 4 score: how often YOLO picks the right fruit, by target size
```

All print PASS/FAIL (or a table) and write nothing.

## Files

| File | What |
|---|---|
| `tomato.py` | Trusses + tomatoes. Each fruit hangs on a soft weld (the tether). `Detacher` switches the weld off once its force stays above the fruit's break force (OrchardBench fruit logic) |
| `scene.py` | Greenhouse (laser-cut dimensions), SO-101 + fin-ray fingers + palm, vines, `birdseye` + `arm/claw` cameras. Exactly one ripe fruit within reach |
| `arm_map.py` | lerobot units (`use_degrees=false`) ↔ sim joint angles through `follower_white.json`, the way the real follower converts them |
| `lens.py` | The InnoMaker 130° lens on MuJoCo's pinhole cameras: render wider, remap through a fisheye model (OpenCV `cv2.fisheye` K, D). ⚠️ Guessed from the spec; a calibration in `cameras.json` replaces it |
| `cameras.py` | Renders `birdseye` + `claw` through the lens; true target box from the segmentation render |
| `target.py` | The start-frame box: YOLO selects (rule in `ripeness/gate/birdseye.py`), the true box checks it and fills in |
| `yolo_worker.py`, `yolo_client.py` | The ripeness YOLO in its own env (no ultralytics in the lerobot env), over pipes |
| `eval_yolo_recall.py` | The recall score |
| `view_scene.py` | The window; `--leader` reads the real leader |
| `check_*.py` | Checks |

## Traps found so far

- ⚠️ **Friction cones.** Grasping needs `cone="elliptic"` + `impratio`. With MuJoCo's default pyramidal
  cones, a squeezed 10 g fruit crept out of the fingers at 0.1 N.
- ⚠️ **Palm collider.** The fin-ray stubs are visual only. A box stands in for the gripper body
  between the fingers; without it, a squeezed fruit slides back through the palm.
- ⚠️ **Fingers in group 2, not 3.** They're visual + collision in one geom, and group 3 is hidden,
  which left them out of the claw image.
- ⚠️ **Rigid trusses pluck on touch.** A closing jaw drags a fruit ~8 mm, so the rachis hangs on a
  sprung ball joint. At the weakest break force (0.58 N), an off-centre close still peaks at 0.57 N:
  the most fragile fruit are on the edge of dropping when touched.
- ⚠️ **One renderer per model.** A `mujoco.Renderer` is bound to the model it was made with.
- ⚠️ **Segmentation renders blend ids at edges.** The offscreen buffer is multisampled, so two
  neighbouring fruits' ids average to a third geom's id. Stray "target" pixels gave a 23 px fruit a
  107 px true box. `cameras.true_box` keeps only the target's real blobs.
- ⚠️ **Join the viewer thread on close.** `viewer.close()` returns before the render loop has stopped.
  Opening the next scene's viewer, or exiting, while it still runs hung the process at exit.

## Known gaps (block collecting episodes, step 5)

- ⚠️ **The jaw closes to its −10° stop on every grasp.** The fruit sinks ~6 mm into the finger outline
  instead of stalling the servo, so the 50 % and 25 % torque grasp tests give identical results: the hold
  comes from that sink, not from the motor torque. Recorded gripper state while holding reads ~0 whatever
  the fruit size, while the real servo should stall higher. Measure first: grip a cherry tomato with the
  real follower and read its gripper value, then tune the finger contact until the sim stalls there.
- ⚠️ **The lens is a guess from the retail spec** (130° diagonal, 103° horizontal at 1080p), with
  640x480 assumed to be a 4:3 crop. That gives 72° x 52° at 640x480 and targets 10–35 px across
  (median 23), not the ~6 px the older notes estimated. If the camera squashes 16:9 instead of
  cropping, horizontal FOV is wider (set `lens.ASPECT = "squash"`). Calibrate before trusting the
  recall score.
- **Recall score on the flat-colour tier: 18 %** (`eval_yolo_recall.py`, 100 layouts, `pretrain_v4`,
  t 0.25). YOLO puts some box on the target 95 % of the time but mostly calls the flat red sphere
  `not_tomato`. That's expected (it learned from photos), and it's what the re-render tiers (step 6)
  should raise.

## Unverified (needs the real rig)

- **Joint signs and zero offsets** (`arm_map.SIGN`): check with `view_scene.py --leader`. That path has
  never run against the arm: the leader object builds and loads `leader_black.json`, but `connect()` is untested.
- **Camera intrinsics + lens distortion** (`lens.py`, guessed from the spec) and the **birdseye mast
  pose** (the generator's design pose; the mast isn't printed yet). The claw camera is on the stock
  SO-101 wrist mount, which the sim already uses.
- **Selection threshold and weights** for the birdseye (`target.T_RIPE_BIRDSEYE` 0.25, `pretrain_v4`):
  the real ones come from a measured birdseye session (Roadmap M2.5).
- **Arm yaw and target row** (`scene.ARM_YAW_DEG`, `TARGET_ROW`).
- **Contact and plant constants**: break force, truss stiffness, finger friction and softness, and
  the jaw angle while holding a fruit (sim: −10°).
- **Filament colours** (`PLA_RGBA`, `TPU_RGBA`).

## Status

1. ✅ Tomato release (`check_tomato_release.py`, all pass)
2. ✅ Scene (`check_scene.py`, all pass; the unverified list above still applies)
3. 🟨 Teleop recorder: `scripts/record_sim.sh` (plugin checked headless, never run with the leader). Full state + calibration hash go to the state log, not the dataset metadata
4. ✅ YOLO overlay in sim: lens model, start-frame YOLO box with true-box fallback, recall score 18 % on the flat tier (`eval_yolo_recall.py`, `check_plugin.py` all pass). Camera numbers are spec guesses
5. ⬜ Collect · 6. ⬜ Re-render tiers · 7. ⬜ Real episodes + training
