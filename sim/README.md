# sim/: the SO-101 harvesting cherry-tomato trusses in MuJoCo

A simulated SO-101 in the mini greenhouse, harvesting **whole trusses** (since 2026-11-03). Crowded vines with
6-8 fruit per truss. The arm has rigid fingers with a blade pair and spring pads at the tips: one close cuts the
truss's stalk and holds it, then the arm carries the whole truss to the crate. Episodes come from a scripted
expert (`generate_truss.py`), not from a person on the leader.

The single-fruit scene (fin-ray fingers plucking one fruit, 2026-10) and its tools were removed on 2026-11-08;
its datasets were deleted on 2026-11-03.

Plan and reasoning: vault `08 SOARM101 Training/(C) BLUSH Sim Teleop Plan.md`.

## Watch it

Run in the `lerobot` env (`conda activate lerobot`), from `sim/`. Every line opens a window.

```bash
python view_scene.py --cams              # a truss scene + birdseye | claw exactly as recorded (green box + cross = target)
python view_scene.py --cams --rig base   # the rig "base" cameras: base | claw, and the base camera's bracket
python view_scene.py --leader --cams     # the real leader arm drives the sim arm (read only, records nothing)
python check_truss_cut.py --windowed     # one truss stalk across the fingertips: close, cut, then shake it
python check_tomato_release.py --windowed   # one truss: double-click a tomato, Ctrl + right-drag to pluck it
python generate_truss.py --dry --watch --seed-start 200007 --episodes 1   # the scripted expert harvests one truss
```

Without `--leader`, move the arm with the actuator sliders in the viewer's right panel (Control).
Ctrl + right-drag pulls a tomato by hand; plucks and stalk cuts are printed in the terminal. Wayland prints
`libdecor` / `GLFWError: window position` warnings, which are harmless.

A trained policy (none right now: all were deleted on 2026-11-08; `scripts/train_sim.sh act-truss` writes the next
one to `../outputs/train/act_truss_v1/`):

```bash
python rollout_truss.py ../outputs/train/<run>/checkpoints/last/pretrained_model   # scene after scene
python eval_truss.py ../outputs/train/<run>/checkpoints/last/pretrained_model      # the same, graded per scene
python dagger_truss.py --dry --watch --episodes 1   # the policy drives, the expert takes over when it veers off
```

`rollout_truss.py` and `eval_truss.py` show the MuJoCo viewer plus one window per camera (exactly what the
policy gets). Keys in a camera window: N = next scene, Q = quit. They load any lerobot checkpoint (ACT now), take
the rig from the checkpoint's camera inputs and the dataset from its `train_config.json`. The YOLO worker runs in
the `ripeness` env; `BLUSH_RIPENESS_PYTHON` (below) puts it in its own memory scope.

## Truss harvesting (K7 2026-11-03)

**The scene** (`truss_scene.py`): the greenhouse, arm base, crate side and start pose from `scene.py`, with 4-5
stems of 2-3 trusses each (minus any touching the arm at its start pose, `truss_scene.START_CLEAR_M`), crossing
in front of each other. 1-2 trusses per scene are ready (tip fruit coloured) and cuttable; the rest are unripe
or "almost" decoys. The crate is big enough for a truss.

**The fingers** (`scene.add_cut_hold`, `truss.TrussCutter`): rigid PLA fingers with a blade pair on the
stem-side face of the tips and spring pads on the rest. When both blades press the stalk hard enough for long
enough, the stalk is cut and the pads keep holding it. `check_truss_cut.py` tests that per stalk size.

**The target** (`truss_target.py`): on the first frame the truss YOLO (`yolo_truss/`) picks the ready truss
and its cut point; if it's wrong or misses, the sim's true box is used. Box + cut point are held for the
episode, drawn into the overview camera (green box + cross) and recorded as numbers in
`observation.environment_state`.

**The cameras** (`--rig`, `cameras.RIG_CAMERAS`):
- `birdseye`: the mast behind row R0 + the claw camera. Used by `blush_sim_truss_v1` (deleted 2026-11-08).
- `base` (K7 2026-11-06): a camera low on a bracket left of the arm's base + the claw camera. The target is
  still picked on the birdseye's first frame (the truss YOLO learned birdseye views); its box and cut point
  go onto the base camera from the sim's truth. The claw camera stays clean. `generate_truss.py`'s default.

**The expert** (`truss_expert.py`): IK (`truss_ik.py`) finds the goal poses (cut pose, pre-cut, above the
crate, release), RRT-Connect (`rrt.py`, copied from S.H.E.A.R) finds every free move between them, steering
clear of the greenhouse, the other trusses and the target's own fruit. Both build on `expert.py`'s arm IK,
plan and min-jerk timing. A share of episodes first close short of the stalk and go in again (recovery), and
with `--hold-share` some scenes have nothing ready and the arm stays home.

**Recording** (`generate_truss.py`): each seed is planned, dry-run with physics and judged; only a pass is
rebuilt and recorded like `lerobot-record` (same features, units, AV1). Judged on the last frame: target truss
cut and whole in the crate, no other truss cut, no fruit off, neighbours barely moved, arm back home.
Defaults: rig `base`, 200 episodes into `krish5831/blush_sim_truss_v3` (the next dataset).

```bash
export BLUSH_RIPENESS_PYTHON=$PWD/scripts/capped_ripeness_python.sh   # the YOLO worker in its own 3 GB scope
python generate_truss.py --pilot --watch     # 5 episodes into <repo-id>_pilot, MuJoCo window at real time
python generate_truss.py --dry --episodes 20 # plan + dry run + judge only: keep rate, writes no dataset
# how blush_sim_truss_v2 was made (2026-11-06), under a memory cap:
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  python generate_truss.py --episodes 200 --repo-id krish5831/blush_sim_truss_v2 --rig base \
    --start-jitter-deg 3 --hold-share 0.10 --balance --seed-start 7600000
lerobot-dataset-viz --repo-id krish5831/blush_sim_truss_v2 --episode-index 0   # what got recorded
```

Writes the dataset under `~/.cache/huggingface/lerobot/<repo-id>`, the state log and
`~/.cache/blush_sim/truss/<name>_<stamp>.jsonl` (every attempt and why). Never pushes.

**DAgger** (`dagger_truss.py`, 2026-11-05): the policy drives a fresh scene; when it starts to veer off, the
episode is rewound a little and the scripted expert finishes it. The recorded episode holds the policy's own
actions up to the takeover. The sim runs on the checkpoint's rig. Defaults: `--policy` =
`../outputs/train/act_truss_v1/checkpoints/last/pretrained_model` (not trained yet), appending to
`krish5831/blush_sim_truss_v2`; it refuses a dataset whose cameras don't match the checkpoint's.

**The truss YOLO** (`yolo_truss/`, sim-only weights, classes `truss_ready` / `truss_not_ready` / `cut_point`):
`make_dataset.py` renders birdseye frames with labels from the sim's truth (lerobot env), `train.sh v1` trains
YOLO11n into `../runs/yolo_truss/truss_v1/` and `train.sh eval v1` scores the frozen test split (ripeness env).
`generate_truss.py`, `view_scene.py` and the `scripts/` wrappers use those weights when they exist.

**Datasets** (manifest: `../harvest/datasets.md`):

| Dataset | Rig | Episodes |
|---|---|---|
| `krish5831/blush_sim_truss_v2` | base | 200 scripted (seeds 7600000+), ~10 % stay-home, 3° start jitter, targets spread along the row |

`blush_sim_truss_v1` (birdseye, 200 scripted + 100 DAgger) and the pilots were deleted on 2026-11-08.

**Training** (`scripts/train_sim.sh`, run from the repo root): ⚠️ **ACT is the path** (K7 2026-11-08).
`train_sim.sh act-truss` trains ACT on `blush_sim_truss_v2` with the box + cut point as numbers and temporal
ensembling, 100k steps, a checkpoint every 10k into `outputs/train/act_truss_v1`, picked by graded rollouts
(`eval_truss.py`), not by loss. Not started yet. The other recipes: `act-smoke [ds]` (500 steps on v2: data
loads, loss falls), `act-full`, `merge <real_ds>`, `pi05-smoke`, `pi05-full`; the script header says what each
does. SmolVLA was tried on the truss episodes and removed with its recipes on 2026-11-08.

## Teacher v3 pipeline (the post-mortem ladder, November 2026)

The scripted planner's episodes held coin-flips the camera can't see, so the policy data now comes from a
deterministic closed-loop teacher (`truss_teacher.py`), half of the episodes with noise on the executed action and
the teacher's clean action as the label. Plan and results: vault note "BLUSH + SHEAR Policy Post-Mortem".

| Step | What | State |
|---|---|---|
| 0 | teacher + its three tests (`check_teacher.py`) | done: solves 73 of 200 scenes |
| 1 | one layout, 50 episodes, ACT: the pipeline check | passed at 40k steps (18 of 19 noisy rollouts) |
| 2 | dataset v3 | done: `krish5831/blush_sim_teacher_v3`, 610 episodes (525 cuts, 85 stay-home), local only |
| 3 | ACT on v3, a test at every 5k checkpoint | set up, not started |

```bash
python check_teacher.py solve --scenes 1 --seed-start 9100003 --watch     # watch the teacher do one scene
# step 2, how v3 was recorded: one process per 20-episode part (the recorder grows to ~6-7 GB while it records,
# and a dataset is only readable after its final write), then merged and replayed
python generate_truss.py --teacher --no-dr --no-yolo --balance --hold-share 0.10 --noise-share 0.5 \
    --seed-start 9700000 --episodes 20 --max-attempts 900 --repo-id krish5831/blush_sim_teacher_v3_w_p00
python merge_parts.py krish5831/blush_sim_teacher_v3 krish5831/blush_sim_teacher_v3_s krish5831/blush_sim_teacher_v3_w
python replay_check.py ~/.cache/blush_sim/truss/blush_sim_teacher_v3_w_p*.jsonl --n 10    # clean episodes must reproduce
# step 3 (from the repo root): cosine learning-rate decay, a checkpoint every 5k
RUN=act_teacher_v3 sim/scripts/train_sim.sh act-teacher krish5831/blush_sim_teacher_v3
# ... and in a second shell, a test of every checkpoint as it lands, on the CPU (the GPU stays with training)
python pick_checkpoint.py ../outputs/train/act_teacher_v3 --watch --until 100000 --cpu -- --headless --no-dr \
    --no-yolo --n-action-steps 100 --seeds-from heldout_v3.jsonl --scenes 20 --holds 5
python eval_truss.py ../outputs/train/act_teacher_v3/checkpoints/<step>/pretrained_model --no-dr --no-yolo \
    --n-action-steps 100 --seeds-from heldout_v3.jsonl --scenes 3            # watch a checkpoint in the windows
```

`heldout_v3.jsonl`: the step-3 test set, 20 scenes from seed 9900000 on (outside every recorded range) that a
clean teacher dry run solved (20 of 53 tried). A success rate on it is "of the scenes the teacher solves". ⚠️ As
scenes come, not balanced like the dataset: 13 of the 20 targets hang in the first quarter of the row.
`--holds 5` adds nothing-ready scenes, where the arm must stay home; they're reported apart.
⚠️ Play whole chunks (`--n-action-steps 100`): ensembling and re-planning every 25 steps both failed in step 1.
⚠️ Pick checkpoints by rollouts: step 1 went 74 % at 20k, 95 % at 40k and 5 % at 50k with the loss flat.

## lerobot commands against the sim (`--robot.type=so101_mujoco`)

The sim is a lerobot robot plugin (`plugin/`), so the normal lerobot tools run on it. The wrappers in
`scripts/` mirror `harvest/scripts/` flag for flag (same timings, codec, camera names) and run a truss scene,
rig `birdseye` unless `RIG=base`. The task string is `generate_truss.TASK`. Extra flags pass through to lerobot.

⚠️ In real-time mode (teleop, record, rollout) the truss scene runs about 8x slower than real time
(`check_plugin.py`: p95 ~290 ms per step against a 33 ms budget, 2026-11-08), so teleop and recording on it lag.
⚠️ `record_sim.sh` can't record `observation.environment_state` (the box + cut point as numbers) the way
`generate_truss.py` does.

| Sim | Real equivalent | What |
|---|---|---|
| `scripts/teleop_sim.sh` | `~/teleop.sh` | leader → sim arm, MuJoCo window + rerun with both cameras |
| `scripts/record_sim.sh [--resume]` | `harvest/scripts/record.sh` | teleop demos into `$SIM_DATASET` (a placeholder, never recorded) |
| `scripts/replay_sim.sh <ep> [ds]` | `lerobot-replay` | play an episode back on the sim arm (real datasets too) |
| `scripts/train_sim.sh <recipe> [ds]` | `harvest/scripts/train_act.sh` | training recipes; the header lists them all. `act-truss` is the next run |
| `scripts/rollout_sim.sh eval\|run\|pi05 <policy>` | `harvest/scripts/rollout.sh` | run a policy on the sim arm; policies trained on the box numbers need `eval_truss.py` instead |

**What the cameras record** (`OVERLAY=`): `yolo` by default when the truss YOLO is trained
(`../runs/yolo_truss/truss_v1/weights/best.pt`, or `YOLO_WEIGHTS=...`), else `true_box`; `OVERLAY=yolo` without
weights is refused. `OVERLAY=none` records raw frames.
- `yolo`, rig `birdseye`: every truss YOLO detection as a red box + class + confidence on both cameras, plus
  the green target box and cut-point cross on the birdseye. The target is picked among the ready trusses on the
  scene's first frame and then held; the true box fills in when YOLO misses.
- `true_box`: only the green box + cross, from the sim's truth.
- rig `base`: the box + cross go onto the base camera from the sim's truth; the claw camera stays clean.

The terminal says which (`scene <n> target: <truss> of <k> ready, YOLO picked it (...)` or the true box and
why), plucks and cuts are printed, and `✅ target truss in the crate` counts only the target.
Sim frames can also be re-rendered without boxes from the state log.

`DRYRUN=1 scripts/<script>.sh ...` prints the lerobot command instead of running it. MuJoCo window:
**N** = new vine layout (press it in lerobot-record's reset period), Ctrl + right-drag = pull a tomato.
The sim reports `robot_type=so_follower` like the real arm, because `lerobot-edit-dataset merge`
refuses mixed robot types. Sim and real are told apart by the dataset name.
Full sim state per frame goes to `~/.cache/blush_sim/states/<session>/` for re-renders, with every
frame's raw YOLO output, the target, and the weights' sha256 (new weights change the training images).

## Setup (once)

```bash
./fetch_so101.sh                  # SO-101 MJCF + meshes into assets/so101 (gitignored, ~16 MB)
pip install -e plugin             # registers --robot.type=so101_mujoco in the lerobot env
python build_finray_meshes.py     # only after changing the finger design; needs numpy shapely trimesh manifold3d
```

`assets/finray/` is committed: the printed fin-ray parts as metre meshes in the MJCF body frames, plus
`chain.json`, the fingers' plate geometry, which the truss fingers are built from too.

## Checks

```bash
python check_tomato_release.py    # the tethers hang, release above threshold, fall free (one truss)
python check_scene.py             # unit mapping, truss layouts, the full scene at the start pose, cameras (both rigs)
python check_truss_cut.py         # the blades cut 3-5 mm stalks and the pads hold the truss while shaking
python check_plugin.py            # the plugin through lerobot on a truss scene: features, timing, AV1 episode, rerun, rig base, YOLO view
```

All print PASS/FAIL and write nothing (`check_plugin.py` uses a temp dir it deletes). ⚠️ `check_plugin.py`'s
timing checks fail on the truss scene (Known gaps).

## Files

| File | What |
|---|---|
| `scene.py` | Greenhouse (laser-cut dimensions), SO-101 facing the vine row with the finger model (rigid cut-and-hold tips in every truss scene; the soft fin-ray version only for `expert.ArmIK`'s kinematics) + palm, crate, `birdseye` + `arm/claw` cameras and always the `base` camera (its bracket only in rig "base" scenes). `compose()` puts them together |
| `tomato.py` | One fruit and its tether (pedicel lean 15–45° + swing, fruit hanging off-vertical, visual calyx). Each fruit hangs on a soft weld; `Detacher` switches it off once its force stays above the fruit's break force (OrchardBench fruit logic). `truss.py` builds its fruit from these |
| `truss.py` | A truss on a cuttable stalk (one free body on a sprung knuckle weld); `TrussCutter` cuts it when both blades press it |
| `truss_scene.py` | Crowded truss scenes, 1-2 ready trusses with a cut pose, the crate |
| `truss_target.py` | The start frame: target truss, its box and cut point (YOLO or true box), YOLO labels |
| `target.py` | IoU, YOLO-to-truth matching and the reach mask, used by `truss_target.py` |
| `arm_map.py` | lerobot units (`use_degrees=false`) ↔ sim joint angles through `follower_white.json`, the way the real follower converts them |
| `lens.py` | The InnoMaker 130° lens on MuJoCo's pinhole cameras: render wider, remap through a fisheye model (OpenCV `cv2.fisheye` K, D). ⚠️ Guessed from the spec; a calibration in `cameras.json` replaces it |
| `cameras.py` | Renders `birdseye`, `base` and `claw` through the lens (`RIG_CAMERAS`: which two the policy gets); true boxes from the segmentation render |
| `../harvest/yolo_overlay.py` | Draws the YOLO view; shared with the real recordings and rollout |
| `yolo_worker.py`, `yolo_client.py` | The YOLO in the `ripeness` env (no ultralytics in the lerobot env); frames through `/dev/shm`. Needs weights |
| `expert.py` | Shared arm motion for the truss expert: `ArmIK`, `Plan`, min-jerk, timing constants |
| `truss_ik.py`, `rrt.py` | IK for the cut pose; RRT-Connect routes around what the arm mustn't touch (from S.H.E.A.R) |
| `truss_expert.py` | The scripted truss harvester (cut, carry, release, home; miss + recovery; takeovers for DAgger) |
| `randomize.py` | Domain randomisation per scene seed (own RNG stream, so the layout doesn't change) |
| `generate_truss.py` | Scripted truss episodes: dry run, judge, record |
| `dagger_truss.py` | DAgger: policy drives, expert takes over, recorded into the truss dataset |
| `eval_truss.py`, `rollout_truss.py` | A truss policy in the sim, graded / just watched (windows; `eval_truss.py --headless` for batches) |
| `view_scene.py` | The window on a truss scene; `--cams`, `--rig birdseye\|base`, `--overlay`, `--yolo-weights`, `--leader` reads the real leader |
| `check_*.py` | Checks |
| `yolo_truss/` | The truss YOLO: dataset from the sim's truth, training, frozen-test eval |
| `scripts/capped_ripeness_python.sh` | The ripeness env's python in its own 3 GB memory scope (the YOLO worker) |

## The layout

K7's call (2026-10-20): the arm base faces the vine row R1 (turned 90° left from facing down the
aisle, which reached only a third of the tomatoes) and the crate stands on its right, down the
aisle. Only trusses with a cut pose (`truss_ik.py`) and a clear approach are made ready. Each scene
starts with the arm tucked to its left (`scene.START_NORM`), clear of fruit, posts and crate, until the
first leader command snaps it to your pose. The MuJoCo window opens behind the robot, so its left and
right are yours.

## The fingers

The finger model comes from the printed fin-ray geometry (`chain.json`, from `finray_gripper.py`) as plates:
front flank, back flank, ribs and tip cap, joined by sprung hinges where the ribs meet the flanks, with the
ribs pinned to the back flank. As soft TPU fingers (`scene.FINRAY_K_MID_N_PER_MM`, 2 N/mm, a guess) they
gripped single fruit; on trusses the tip blades didn't cut (the finger curled round the stalk, twisting the
blades off it). Truss scenes use the same shape near-rigid (`scene.RIGID_FINGER_K_N_PER_MM`, PLA, K7
2026-11-03) with the blade + pad inserts on the tip caps.

## Traps found so far

- ⚠️ **Friction cones.** Grasping needs `cone="elliptic"` + `impratio`. With MuJoCo's default pyramidal
  cones, a squeezed 10 g fruit crept out of the fingers at 0.1 N.
- ⚠️ **Contact softness scales with the fruit's mass.** With MuJoCo's default `solref` the 10 g fruit
  sank up to 13 mm into rigid fingers and slid out. The fingers now set the contact (`priority`) and
  are stiff; the softness lives in the finger hinges.
- ⚠️ **Palm collider.** The fin-ray stubs are visual only. A box stands in for the gripper body
  between the fingers; without it, a squeezed fruit slides back through the palm.
- ⚠️ **Fingers in group 2, not 3.** They're visual + collision in one geom, and group 3 is hidden,
  which left them out of the claw image.
- ⚠️ **One renderer per model.** A `mujoco.Renderer` is bound to the model it was made with.
- ⚠️ **The start pose.** With the base facing the row, the joint zero (arm straight out) starts inside
  the vines, the old folded pose leans into row 0's floor rail, and a fully folded elbow presses
  into the shoulder so the servo can't hold it. `START_NORM` avoids all three.
- ⚠️ **Camera renders go through EGL.** A GLFW camera renderer starting while the MuJoCo window's thread
  starts GLFW tripped a GLFW assertion and aborted the process (1 run in 2).
- ⚠️ **Segmentation renders blend ids at edges.** The offscreen buffer is multisampled, so two
  neighbouring fruits' ids average to a third geom's id. Stray "target" pixels gave a 23 px fruit a
  107 px true box. `cameras.true_box` keeps only the target's real blobs.
- ⚠️ **Join the viewer thread on close.** `viewer.close()` returns before the render loop has stopped.
  Opening the next scene's viewer, or exiting, while it still runs hung the process at exit.
- ⚠️ **Frames through a pipe cost 4 ms.** Two 640x480 frames per call pushed a frame past 20 ms;
  they go through a `/dev/shm` file instead.

## Known gaps

- ⚠️ **The truss scene is too slow for real time** (about 8x, see the lerobot section). Teleop, `record_sim.sh`
  and `rollout_sim.sh` lag on it; scripted episodes (`generate_truss.py`) don't, since they step the sim
  exactly 1/30 s per frame.
- ⚠️ **The finger design is a guess** until K7 prints and mounts the real cut-and-hold fingers: blade and pad
  sizes and the rigid stiffness (`scene.py`, cut-and-hold section).
- ⚠️ **The lens is a guess from the retail spec** (130° diagonal, 103° horizontal at 1080p), with
  640x480 assumed to be a 4:3 crop. That gives 72° x 52° at 640x480, and ripe fruit 19–36 px across in the
  birdseye (median 26, measured in the old single-fruit scenes). If the camera squashes 16:9 instead
  of cropping, horizontal FOV is wider (set `lens.ASPECT = "squash"`). Calibrate before trusting the images.
- **The label text doesn't reach the policy.** At the policy's input resolution the boxes survive, the ~3 px
  text doesn't. Per-class box colours would carry the class if that matters.

## Unverified (needs the real rig)

- **Camera intrinsics + lens distortion** (`lens.py`, guessed from the spec), the **birdseye mast pose** (the
  generator's design pose; the mast isn't printed yet) and the **base camera's bracket pose** (design guess,
  nothing printed). The claw camera is on the stock SO-101 wrist mount, which the sim already uses.
- **The crate's exact spot and size** (`truss_scene.CRATE_CENTRE`, `CRATE_SIZE`): "on the robot's right" is K7's; the numbers are a guess.
- **Contact and plant constants**: break force, stalk size and stiffness, finger stiffness and friction.
- **Filament colours** (`PLA_RGBA`, `TPU_RGBA`).
- Checked by K7 with the leader (2026-10-20): joint directions and poses (`arm_map.SIGN`).
- Decided by K7 (2026-10-20): the base faces R1, the crate stands on its right.

## Status

- ✅ Tethers (`check_tomato_release.py`), truss scenes (`check_scene.py`), cut-and-hold fingers
  (`check_truss_cut.py`), truss YOLO (`truss_v1`, sim-only), scripted expert, DAgger.
- ✅ Leader → sim: K7 drove the sim with the leader (2026-10-20, on the old single-fruit scene): joints and
  poses match. Nobody has teleoperated a truss scene yet, and it lags in real time (Known gaps).
- ✅ `blush_sim_truss_v2` (rig base): 200 scripted, kept 200 of 534 attempts in 397 min, 25 % miss + recovery,
  20 stay-home (`run_v2_rig_base.log`, 2026-11-06). The only dataset now.
- ⬜ Policy: none. Earlier runs (single-fruit ACT, SmolVLA on truss v1) were deleted on 2026-11-08. Next:
  ACT on v2 (`train_sim.sh act-truss`).
- ⬜ Re-render tiers · ⬜ Real episodes + training
