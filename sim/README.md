# sim/: the SO-101 harvesting cherry tomatoes in MuJoCo

A simulated SO-101 in the mini greenhouse. Two kinds of scene live here, picked by the plugin's `task`:

- **`task="truss"`, whole trusses (since 2026-11-03, where the work is now).** Crowded vines with 6-8 fruit
  per truss. The arm has rigid fingers with a blade pair and spring pads at the tips: one close cuts the
  truss's stalk and holds it, then the arm carries the whole truss to the crate. Episodes come from a scripted
  expert (`generate_truss.py`), not from a person on the leader.
- **`task="fruit"`, single cherry tomatoes (2026-10, still the plugin default).** The fin-ray fingers pluck
  one ripe fruit. The leader arm can drive it (`scripts/`), and `generate_scripted.py` scripts it. The
  single-fruit datasets were deleted on 2026-11-03.

Plan and reasoning: vault `08 SOARM101 Training/(C) BLUSH Sim Teleop Plan.md`.

## Watch it

Run in the `lerobot` env (`conda activate lerobot`), from `sim/`. Every line opens a window.

Truss harvesting:

```bash
python check_truss_cut.py --windowed      # one truss stalk across the fingertips: close, cut, then shake it
python generate_truss.py --dry --watch --seed-start 200007 --episodes 1   # the scripted expert harvests one truss
python rollout_truss.py ../outputs/train/<run>/checkpoints/last/pretrained_model   # a trained policy, scene after scene
python eval_truss.py ../outputs/train/<run>/checkpoints/last/pretrained_model      # the same, graded per scene
python dagger_truss.py --dry --watch --episodes 1   # the policy drives, the expert takes over when it veers off
```

`rollout_truss.py` and `eval_truss.py` show the MuJoCo viewer plus one window per camera (exactly what the
policy gets). Keys in a camera window: N = next scene, Q = quit. They read the rig (which cameras) and the
dataset from the checkpoint's `train_config.json`. The YOLO worker runs in the `ripeness` env; `BLUSH_RIPENESS_PYTHON`
(below) puts it in its own memory scope.

Single fruit:

```bash
python view_scene.py --cams              # greenhouse window + claw | birdseye exactly as recorded (red = YOLO, green = target)
python check_grasp.py --windowed         # the fin-ray fingers closing on, plucking and shaking a tomato, slowed down
python eval_yolo_recall.py --show        # the recall score: per layout, the start frame's YOLO view (key = next)
python view_scene.py --leader --cams     # the real leader arm drives the sim arm (read only, records nothing)
python check_tomato_release.py --windowed   # one truss: double-click a tomato, Ctrl + right-drag to pluck it
```

Without `--leader`, move the arm with the actuator sliders in the viewer's right panel (Control).
Ctrl + right-drag pulls a tomato by hand; plucks are printed in the terminal. Wayland prints
`libdecor` / `GLFWError: window position` warnings, which are harmless.
`check_grasp.py --windowed --case down,tip,0,12.5` shows one case (pose, height, mm off-centre, radius mm).

## Truss harvesting (`task="truss"`, K7 2026-11-03)

**The scene** (`truss_scene.py`): the same greenhouse, arm base, crate side and start pose as the
single-fruit scene, with 4-5 stems of 2-3 trusses each (minus any touching the arm at its start pose,
`truss_scene.START_CLEAR_M`), crossing in front of each other. 1-2 trusses per
scene are ready (tip fruit coloured) and cuttable; the rest are unripe or "almost" decoys. The crate is bigger.

**The fingers** (`scene.add_cut_hold`, `truss.TrussCutter`): rigid PLA fingers with a blade pair on the
stem-side face of the tips and spring pads on the rest. When both blades press the stalk hard enough for long
enough, the stalk is cut and the pads keep holding it. `check_truss_cut.py` tests that per stalk size.

**The target** (`truss_target.py`): on the first frame the truss YOLO (`yolo_truss/`) picks the ready truss
and its cut point; if it's wrong or misses, the sim's true box is used. Box + cut point are held for the
episode, drawn into the overview camera (green box + cross) and recorded as numbers in
`observation.environment_state`.

**The cameras** (`--rig`, `cameras.RIG_CAMERAS`):
- `birdseye`: the mast behind row R0 + the claw camera. Every dataset up to `blush_sim_truss_v1`.
- `base` (K7 2026-11-06): a camera low on a bracket left of the arm's base + the claw camera. The target is
  still picked on the birdseye's first frame (the truss YOLO learned birdseye views); its box and cut point
  go onto the base camera from the sim's truth. The claw camera stays clean.

**The expert** (`truss_expert.py`): IK (`truss_ik.py`) finds the goal poses (cut pose, pre-cut, above the
crate, release), RRT-Connect (`rrt.py`, copied from S.H.E.A.R) finds every free move between them, steering
clear of the greenhouse, the other trusses and the target's own fruit. A share of episodes first close short
of the stalk and go in again (recovery), and with `--hold-share` some scenes have nothing ready and the arm
stays home.

**Recording** (`generate_truss.py`): each seed is planned, dry-run with physics and judged; only a pass is
rebuilt and recorded like `lerobot-record` (same features, units, AV1). Judged on the last frame: target truss
cut and whole in the crate, no other truss cut, no fruit off, neighbours barely moved, arm back home.

```bash
export BLUSH_RIPENESS_PYTHON=$PWD/scripts/capped_ripeness_python.sh   # the YOLO worker in its own 3 GB scope
python generate_truss.py --pilot --watch     # 5 episodes into <repo-id>_pilot, MuJoCo window at real time
python generate_truss.py --dry --episodes 20 # plan + dry run + judge only: keep rate, writes no dataset
# rig "base" (how blush_sim_truss_v2 was made, 2026-11-06), under a memory cap:
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
  python generate_truss.py --episodes 200 --repo-id krish5831/blush_sim_truss_v2 --rig base \
    --start-jitter-deg 3 --hold-share 0.10 --balance --seed-start 7600000
lerobot-dataset-viz --repo-id krish5831/blush_sim_truss_v2_pilot --episode-index 0   # what got recorded
```

Writes the dataset under `~/.cache/huggingface/lerobot/<repo-id>`, the state log and
`~/.cache/blush_sim/truss/<name>_<stamp>.jsonl` (every attempt and why). Never pushes.

**DAgger** (`dagger_truss.py`, 2026-11-05): the policy drives a fresh scene; when it starts to veer off, the
episode is rewound a little and the scripted expert finishes it. The recorded episode holds the policy's own
actions up to the takeover. Appends to `krish5831/blush_sim_truss_v1` by default.

**The truss YOLO** (`yolo_truss/`, sim-only weights): `make_dataset.py` renders birdseye frames with labels
from the sim's truth (lerobot env), `train.sh v1` trains YOLO11n into `../runs/yolo_truss/truss_v1/` and
`train.sh eval v1` scores the frozen test split (ripeness env). `generate_truss.py` uses those weights when
they exist.

**Datasets** (manifest: `../harvest/datasets.md`):

| Dataset | Rig | Episodes |
|---|---|---|
| `krish5831/blush_sim_truss_v1` | birdseye | 200 scripted (seeds 210000+) + 100 DAgger (seeds 7000000+) |
| `krish5831/blush_sim_truss_v2` | base | 200 scripted (seeds 7600000+), ~10 % stay-home, 3° start jitter, targets spread along the row |

**Training** (`scripts/train_sim.sh`, run from the repo root): ⚠️ **ACT is the path now** (K7 2026-11-08,
"we will not be using smolvla this time"): `train_sim.sh act-truss` trains ACT on `blush_sim_truss_v2` with the
box + cut point as numbers and temporal ensembling, 100k steps, a checkpoint every 10k, picked by graded
rollouts (`eval_truss.py`), not by loss. Tried before and not used now: SmolVLA + LoRA (`smolvla-truss`),
SmolVLA with the action expert trained (`smolvla-truss-expert`, `-expert-v2` on the 300 episodes with DAgger),
and SmolVLA with the vision encoder trained too (`smolvla-truss-v3`, and `colab_train_truss_v3.ipynb` for a
Colab T4).

## lerobot commands against the sim (`--robot.type=so101_mujoco`)

The sim is a lerobot robot plugin (`plugin/`), so the normal lerobot tools run on it. The wrappers in
`scripts/` mirror `harvest/scripts/` flag for flag (same TASK, timings, codec, camera names).
⚠️ They run the plugin's defaults, the single-fruit scene (`task="fruit"`, rig `birdseye`). Truss runs go
through the Python tools above. Extra flags pass through to lerobot, so a teleop look at a truss scene is
`OVERLAY=none scripts/teleop_sim.sh --robot.task=truss` (unverified: nobody has teleoperated a truss scene yet;
no overlay, because without `--robot.yolo_weights` the YOLO view would use the single-fruit weights).

| Sim | Real equivalent | What |
|---|---|---|
| `scripts/teleop_sim.sh` | `~/teleop.sh` | leader → sim arm, MuJoCo window + rerun with both cameras |
| `scripts/record_sim.sh [--resume]` | `harvest/scripts/record.sh` | teleop demos into `$SIM_DATASET` (a placeholder, never recorded) |
| `scripts/replay_sim.sh <ep> [ds]` | `lerobot-replay` | play an episode back on the sim arm (real datasets too) |
| `scripts/train_sim.sh <recipe> [ds]` | `harvest/scripts/train_*.sh` | training recipes; the header lists them all. `act-truss` is the current one |
| `scripts/rollout_sim.sh eval\|run\|pi05 <policy>` | `harvest/scripts/rollout.sh` | run a policy on the sim arm (single-fruit scene) |

**What the cameras record** (`--robot.overlay=yolo`, the scripts' default; K7 2026-10-20): the YOLO
view, drawn into both camera frames every frame by `harvest/yolo_overlay.py`:
- every detection: a red box + class + confidence, e.g. `ripe 82%`, on both cameras
- birdseye: a green box on **the target**. Every scene has 2–3 ripe fruits; on the scene's first frame
  the ripe box YOLO is most confident about (inside the reach mask) picks the target, and the green
  box then stays on it for the whole scene. **Pick the green one.** If YOLO picks nothing or a fruit
  that isn't ripe, the most visible ripe fruit is the target and its true box is drawn instead, with
  no label (YOLO never called it ripe).

The terminal says which (`scene 3 target: s0r1_t0 of 2 ripe, YOLO picked it (ripe 33%, ...)` or
`true box (yolo missed)`), plucks are printed, and `✅ target in the crate` counts only the green one.
`OVERLAY=none` records raw frames; sim frames can also be re-rendered without boxes from the state log.

`DRYRUN=1 scripts/<script>.sh ...` prints the lerobot command instead of running it. MuJoCo window:
**N** = new vine layout (press it in lerobot-record's reset period), Ctrl + right-drag = pull a tomato.
The sim reports `robot_type=so_follower` like the real arm, because `lerobot-edit-dataset merge`
refuses mixed robot types. Sim and real are told apart by the dataset name.
Full sim state per frame goes to `~/.cache/blush_sim/states/<session>/` for re-renders, with every
frame's raw YOLO output, the target, and the weights' sha256 (new weights change the training images).

## Scripted single-fruit episodes (`generate_scripted.py`, K7 2026-10-27): legacy

The single-fruit path, kept until the user decides about dropping single fruit. Truss harvesting replaced it.


A scripted picker (`expert.py`) instead of a person on the leader: IK to a grasp pose on the target,
an approach that the kinematic sweep keeps out of the scenery, close, pull (or twist then pull), lift,
crate, back to the start pose. Each seed is dry-run with physics first and only recorded if it passed;
the recording uses the same features, units, YOLO view and codec as `lerobot-record`. Its datasets are
deleted (`blush_sim_scripted_v0` on 2026-10-28, `_v1` on 2026-11-03, `_v1_box` gone from the local cache); its default
`--repo-id` still names v0.

```bash
export BLUSH_RIPENESS_PYTHON=$PWD/scripts/capped_ripeness_python.sh   # the YOLO worker in its own 3 GB scope
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 python generate_scripted.py --pilot --watch --max-attempts 40
python generate_scripted.py --dry --episodes 10 --max-attempts 30    # plan + judge only, writes no dataset
```

- Mix of the kept episodes: 35 % same-truss scenes, 30 % twist then pull, 10 % near miss + correction.
  2–3 ripe per scene. Domain randomisation (`randomize.py`): lights, colours inside each ripeness
  class, camera poses, finger friction + stiffness, truss stiffness, fruit mass.
- The close command is "fully closed" (`expert.CLOSE_NORM` 0–2): the jaw stalls on the fruit by
  itself, at about −9.5° for 20–25 mm fruit. ⚠️ Redo once the real `gripper.pos` holding a fruit is read.
- ⚠️ A twist doesn't weaken the tether in the sim (the `Detacher` only counts translational force),
  so a twisted fruit still comes off on the pull.
- ⚠️ Yield ~20–25 % of seeds. Most misses: a green neighbour 22–38 mm from the target (same truss) gets
  knocked off as the jaw closes or pulls. Every fruit uses the same 0.58–2.46 N break force, ripe or not.
- Writes the dataset under `~/.cache/huggingface/lerobot/<repo-id>` (refuses if it exists), the state
  log, and `~/.cache/blush_sim/scripted/<name>_<stamp>.jsonl` (every attempt and why). Never pushes.
- `eval_policy.py <checkpoint>` (windows) and `analyze_policy.py` (headless, writes
  `../outputs/analysis/<stamp>_<tag>/episodes.jsonl` + PNGs) test single-fruit ACT policies;
  `add_box_feature.py` added the box as numbers to `_v1`. All three need the deleted `blush_sim_scripted_v1`.

## Setup (once)

```bash
./fetch_so101.sh                  # SO-101 MJCF + meshes into assets/so101 (gitignored, ~16 MB)
pip install -e plugin             # registers --robot.type=so101_mujoco in the lerobot env
python build_finray_meshes.py     # only after changing the fin-ray design; needs numpy shapely trimesh manifold3d
```

`assets/finray/` is committed: the printed fin-ray parts as metre meshes in the MJCF body frames, plus
`chain.json`, the fingers' plate geometry for the bending model.

## Checks

```bash
python check_tomato_release.py    # step 1: the tether hangs, releases above threshold, falls free
python check_scene.py             # step 2: unit mapping, 200 layouts (2-3 ripe each), cameras
python check_grasp.py             # 36 teleop-like grasps: close, pluck, shake; --table for the numbers
python check_plugin.py            # the plugin through lerobot: features, 30 fps, AV1 episode, rerun, YOLO view per frame
python eval_yolo_recall.py        # step 4 score: how often YOLO picks a ripe fruit, by fruit size
python check_truss_cut.py         # truss: the blades cut 3-5 mm stalks and the pads hold the truss while shaking
```

All print PASS/FAIL (or a table) and write nothing.

## Files

| File | What |
|---|---|
| `tomato.py` | Trusses + tomatoes (pedicel lean 15–45° + swing, fruit hanging off-vertical, visual calyx). Each fruit hangs on a soft weld (the tether). `Detacher` switches the weld off once its force stays above the fruit's break force (OrchardBench fruit logic). `truss.py` reuses its fruit and `Detacher` |
| `scene.py` | Greenhouse (laser-cut dimensions), SO-101 facing the vine row + bending fin-ray fingers + palm, crate on its right, vines with 2–3 ripe fruits (each one IK-checked reachable), `birdseye` + `arm/claw` cameras and always the `base` camera (its bracket only in rig "base" scenes) |
| `arm_map.py` | lerobot units (`use_degrees=false`) ↔ sim joint angles through `follower_white.json`, the way the real follower converts them |
| `lens.py` | The InnoMaker 130° lens on MuJoCo's pinhole cameras: render wider, remap through a fisheye model (OpenCV `cv2.fisheye` K, D). ⚠️ Guessed from the spec; a calibration in `cameras.json` replaces it |
| `cameras.py` | Renders `birdseye`, `base` and `claw` through the lens (`RIG_CAMERAS`: which two the policy gets); true boxes from the segmentation render |
| `target.py` | The start frame: YOLO picks one of the ripe fruits (rule in `ripeness/gate/birdseye.py`), the sim's truth checks it and fills in. `truss_target.py` reuses its IoU, matching and reach mask |
| `../harvest/yolo_overlay.py` | Draws the YOLO view; shared with the real recordings and rollout |
| `yolo_worker.py`, `yolo_client.py` | The ripeness YOLO in its own env (no ultralytics in the lerobot env); frames through `/dev/shm` |
| `eval_yolo_recall.py` | The recall score |
| `expert.py` | The single-fruit scripted picker: arm IK, kinematic sweep (collisions, neighbour press), pick planner, min-jerk frames. `truss_expert.py` / `truss_ik.py` build on its `ArmIK`, `Plan`, min-jerk and timing constants |
| `generate_scripted.py` | Scripted episodes: dry run, judge, record (above) |
| `randomize.py` | Domain randomisation per scene seed (own RNG stream, so the layout doesn't change) |
| `view_scene.py` | The window; `--leader` reads the real leader |
| `check_*.py` | Checks |
| `eval_policy.py`, `analyze_policy.py`, `add_box_feature.py` | Legacy single-fruit ACT eval (windows), failure analysis, box-feature copy of a dataset. All need the deleted `blush_sim_scripted_v1` |
| `truss.py` | A truss on a cuttable stalk (one free body on a sprung knuckle weld); `TrussCutter` cuts it when both blades press it |
| `truss_scene.py` | Crowded truss scenes, 1-2 ready trusses with a cut pose, bigger crate |
| `truss_target.py` | The truss start frame: target truss, its box and cut point (YOLO or true box), YOLO labels |
| `truss_ik.py`, `rrt.py` | IK for the cut pose; RRT-Connect routes around what the arm mustn't touch (from S.H.E.A.R) |
| `truss_expert.py` | The scripted truss harvester (cut, carry, release, home; miss + recovery; takeovers for DAgger) |
| `generate_truss.py` | Scripted truss episodes: dry run, judge, record |
| `dagger_truss.py` | DAgger: policy drives, expert takes over, recorded into the truss dataset |
| `eval_truss.py`, `rollout_truss.py` | A truss policy in the sim, graded / just watched (windows; `eval_truss.py --headless` for batches) |
| `yolo_truss/` | The truss YOLO: dataset from the sim's truth, training, frozen-test eval |
| `colab_train_truss_v3.ipynb` | SmolVLA truss v3 fine-tune on a Colab T4 (not the current path: ACT is) |
| `scripts/capped_ripeness_python.sh` | The ripeness env's python in its own 3 GB memory scope (the YOLO worker) |

## The layout (single fruit; the truss scene keeps the base, start pose and crate side)

K7's call (2026-10-20): the arm base faces the vine row R1 (turned 90° left from facing down the
aisle, which reached only a third of the tomatoes) and the crate stands on its right, down the
aisle, 18 cm from the base. Every ripe fruit is one IK can reach: the grasp point between the
fingers gets to its centre within 3 mm (`scene.ik_gap`). Each scene starts with the arm tucked to
its left (`scene.START_NORM`), clear of fruit, posts and crate, until the first leader command snaps
it to your pose. The MuJoCo window opens behind the robot, so its left and right are yours.

## The fin-ray fingers

Each TPU finger is built from the printed geometry (`chain.json`, from `finray_gripper.py`) as plates:
front flank, back flank, ribs and tip cap, joined by sprung hinges where the ribs meet the flanks,
with the ribs pinned to the back flank. Pressed by a fruit, the flank dents around it, so the fruit
sits in a pocket against several plates. The springs are scaled so the finger is
`scene.FINRAY_K_MID_N_PER_MM` stiff when pushed at mid-length (2 N/mm, ⚠️ a guess until measured).
The jaw now stalls at an angle that depends on fruit size: about −9.5° for 20–25 mm fruit at
mid-finger, up to +4° for a 30 mm fruit deep against the palm.

`check_grasp.py`, rigid fingers (before) → fin-ray fingers (now):

| | rigid | fin-ray |
|---|---|---|
| plucks through the tether when pulled | 23 / 36 | 36 / 36 |
| fruit slides in the grip on the pull | up to 143 mm (out) | ≤ 1.2 mm |
| held through 6 s of teleop-speed shaking | 5 / 36 within 6 mm | 36 / 36 within 8 mm (≤ 4.4 mm for 20–25 mm fruit; 30 mm fruit up to 6.4 mm, gripped against the palm) |
| fruit sinks into the finger | up to 12.8 mm | ≤ 1.2 mm |

Not reproduced: the textbook fin-ray tip curling toward the object. With this rib angle and plate
thickness the model's tip moves slightly away when the mid flank is pushed; whether the printed
finger curls in is for the real finger to show.

## Traps found so far

- ⚠️ **Friction cones.** Grasping needs `cone="elliptic"` + `impratio`. With MuJoCo's default pyramidal
  cones, a squeezed 10 g fruit crept out of the fingers at 0.1 N.
- ⚠️ **Contact softness scales with the fruit's mass.** With MuJoCo's default `solref` the 10 g fruit
  sank up to 13 mm into rigid fingers and slid out. The fingers now set the contact (`priority`) and
  are stiff; the softness lives in the finger hinges.
- ⚠️ **A plain hinge chain makes the fruit creep out.** Hinges along the front flank alone bend away
  from the fruit and open the gap toward the tip: the fruit walked 6–7.5 mm out while shaking.
  The ribs + back flank fix that.
- ⚠️ **Palm collider.** The fin-ray stubs are visual only. A box stands in for the gripper body
  between the fingers; without it, a squeezed fruit slides back through the palm.
- ⚠️ **Fingers in group 2, not 3.** They're visual + collision in one geom, and group 3 is hidden,
  which left them out of the claw image.
- ⚠️ **Rigid trusses pluck on touch.** A closing jaw drags a fruit ~8 mm, so the rachis hangs on a
  sprung ball joint. The most fragile fruit (0.58 N) now come off as the jaw closes on them, and stay
  in the grip (`check_grasp.py` checks that).
- ⚠️ **One renderer per model.** A `mujoco.Renderer` is bound to the model it was made with.
- ⚠️ **The start pose.** With the base facing the row, the joint zero (arm straight out) starts inside
  the vines, the old folded pose leans into row 0's floor rail, and a fully folded elbow presses
  into the shoulder so the servo can't hold it. `START_NORM` avoids all three.
- ⚠️ **Which way the grasp test's truss runs matters.** The stand-in truss must pivot beyond the
  fingertips, as the scene's trusses do for an arm facing the row. Pivoting across the jaw, a 20 mm
  fruit gripped at the palm slid out on the pull, and some fruit came off while the jaw closed.
- ⚠️ **Camera renders go through EGL.** A GLFW camera renderer starting while the MuJoCo window's thread
  starts GLFW tripped a GLFW assertion and aborted the process (1 run in 2).
- ⚠️ **Segmentation renders blend ids at edges.** The offscreen buffer is multisampled, so two
  neighbouring fruits' ids average to a third geom's id. Stray "target" pixels gave a 23 px fruit a
  107 px true box. `cameras.true_box` keeps only the target's real blobs.
- ⚠️ **Join the viewer thread on close.** `viewer.close()` returns before the render loop has stopped.
  Opening the next scene's viewer, or exiting, while it still runs hung the process at exit.
- ⚠️ **Frames through a pipe cost 4 ms.** Two 640x480 frames per call pushed a frame past 20 ms;
  they go through a `/dev/shm` file instead (render + YOLO on both cameras + drawing: p95 ~16 ms).

## Known gaps (single-fruit scene; block collecting teleop episodes, step 5)

- ⚠️ **Finger stiffness and the jaw angle while holding are guesses.** Push a printed finger at
  mid-length with the scale (force at 5 mm) and read the real follower's `gripper.pos` while it
  holds a cherry tomato; then `FINRAY_K_MID_N_PER_MM` gets set so the sim jaw stalls there too.
- ⚠️ **The lens is a guess from the retail spec** (130° diagonal, 103° horizontal at 1080p), with
  640x480 assumed to be a 4:3 crop. That gives 72° x 52° at 640x480 and ripe fruit 19–36 px across
  (median 26). If the spec guess holds, that's ~4x the ~6 px the older notes estimated. If the
  camera squashes 16:9 instead of cropping, horizontal FOV is wider (set `lens.ASPECT = "squash"`).
  Calibrate before trusting the recall score.
- **Recall score on the flat-colour tier: 46 %** (`eval_yolo_recall.py`, 100 layouts with 2–3 ripe
  fruits, `pretrain_v4`, t 0.25; 18 % before the calyx and the varied hang, 2026-10-27). YOLO puts some box on a ripe fruit every time but mostly calls the
  flat red sphere `not_tomato`, so most recorded labels will read `not_tomato` until step 6. That's
  expected (it learned from photos), and it's what the re-render tiers should raise.
- **The label text doesn't reach the policy.** π0.5 sees 224x168; the boxes survive that, the ~3 px
  text doesn't. Per-class box colours would carry the class if that matters.
- **Recording is close to the 30 fps budget with the YOLO view on.** A full record-loop iteration
  (action, observation with YOLO on both cameras, dataset frame, rerun) took 27.5 ms median, 30 ms
  p95, 1 in 600 over 33 ms (2026-10-20, offscreen). If `lerobot-record` warns that the loop runs
  slower than its fps, the episode gets time-compressed: record with `OVERLAY=none` and draw the
  view afterwards instead.
- **Fingers crashed into the scenery fold too far.** Driven into the table by the arm's servos, a
  finger hinge folds up to 132° (a printed rib cell can't) and springs back straight once clear
  (`check_grasp.py` checks the spring-back and that nothing goes unstable).

## Unverified (needs the real rig)

- **Camera intrinsics + lens distortion** (`lens.py`, guessed from the spec) and the **birdseye mast
  pose** (the generator's design pose; the mast isn't printed yet). The claw camera is on the stock
  SO-101 wrist mount, which the sim already uses.
- **Selection threshold and weights** for the birdseye (`target.T_RIPE_BIRDSEYE` 0.25, `pretrain_v4`):
  the real ones come from a measured birdseye session (Roadmap M2.5).
- **The crate's exact spot and size** (`scene.CRATE_CENTRE`, `CRATE_SIZE`): "on the robot's right" is K7's; the numbers are a guess.
- **Contact and plant constants**: break force, truss stiffness, finger stiffness and friction (1.2).
- **Filament colours** (`PLA_RGBA`, `TPU_RGBA`).
- Checked by K7 with the leader (2026-10-20): joint directions and poses (`arm_map.SIGN`).
- Decided by K7 (2026-10-20): the base faces R1, the crate stands on its right.

## Status

1. ✅ Tomato release (`check_tomato_release.py`, all pass)
2. ✅ Scene (`check_scene.py`, `check_grasp.py`, all pass; the unverified list above still applies)
3. ✅ Teleop recorder: K7 drove the sim with the leader (2026-10-20): joints and poses match. `lerobot-record` itself hasn't run a full session yet. Full state + calibration hash go to the state log, not the dataset metadata
4. ✅ YOLO view in sim: lens model, 2–3 ripe fruits, live YOLO view in both recorded cameras, green start-frame target with true-box fallback, recall score 46 % on the flat tier (18 % before the calyx, 2026-10-27). Camera numbers are spec guesses
5. 🔶 Collect: single-fruit scripted episodes made and deleted (2026-10-27 to 11-03); teleop sessions not yet · 6. ⬜ Re-render tiers · 7. ⬜ Real episodes + training

Truss harvesting (since 2026-11-03):
- ✅ Cut-and-hold fingers (`check_truss_cut.py`), truss scenes, truss YOLO (`truss_v1`, sim-only), expert.
- ✅ `blush_sim_truss_v1` (birdseye): 200 scripted, kept 200 of 513 attempts in 661 min (`~/.cache/blush_sim/truss/run_v1.log`), then 100 DAgger episodes (2026-11-05/06).
- ✅ `blush_sim_truss_v2` (rig base): 200 scripted, kept 200 of 534 attempts in 397 min, 25 % miss + recovery, 20 stay-home (`run_v2_rig_base.log`, 2026-11-06).
- 🔶 Policy: SmolVLA runs tried (LoRA, action expert v1/v2, v3); now ACT on v2 (`train_sim.sh act-truss`, 2026-11-08).
