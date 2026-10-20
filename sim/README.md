# sim/: leader arm → MuJoCo follower

The real leader arm drives a simulated SO-101 with the fin-ray gripper in the mini greenhouse. The sim
cameras record the episodes as **co-training data for π0.5**, next to the real-arm episodes. Plan and
reasoning: vault `08 SOARM101 Training/(C) BLUSH Sim Teleop Plan.md`.

## Watch it

Run in the `lerobot` env (`conda activate lerobot`), from `sim/`:

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
```

All print PASS/FAIL (or a table) and write nothing.

## Files

| File | What |
|---|---|
| `tomato.py` | Trusses + tomatoes. Each fruit hangs on a soft weld (the tether). `Detacher` switches the weld off once its force stays above the fruit's break force (OrchardBench fruit logic) |
| `scene.py` | Greenhouse (laser-cut dimensions), SO-101 + bending fin-ray fingers + palm, vines with 2–3 ripe fruits, `birdseye` + `arm/claw` cameras |
| `arm_map.py` | lerobot units (`use_degrees=false`) ↔ sim joint angles through `follower_white.json`, the way the real follower converts them |
| `lens.py` | The InnoMaker 130° lens on MuJoCo's pinhole cameras: render wider, remap through a fisheye model (OpenCV `cv2.fisheye` K, D). ⚠️ Guessed from the spec; a calibration in `cameras.json` replaces it |
| `cameras.py` | Renders `birdseye` + `claw` through the lens; true fruit boxes from the segmentation render |
| `target.py` | The start frame: YOLO picks one of the ripe fruits (rule in `ripeness/gate/birdseye.py`), the sim's truth checks it and fills in |
| `../harvest/yolo_overlay.py` | Draws the YOLO view; shared with the real recordings and rollout |
| `yolo_worker.py`, `yolo_client.py` | The ripeness YOLO in its own env (no ultralytics in the lerobot env); frames through `/dev/shm` |
| `eval_yolo_recall.py` | The recall score |
| `view_scene.py` | The window; `--leader` reads the real leader |
| `check_*.py` | Checks |

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
- ⚠️ **Camera renders go through EGL.** A GLFW camera renderer starting while the MuJoCo window's thread
  starts GLFW tripped a GLFW assertion and aborted the process (1 run in 2).
- ⚠️ **Segmentation renders blend ids at edges.** The offscreen buffer is multisampled, so two
  neighbouring fruits' ids average to a third geom's id. Stray "target" pixels gave a 23 px fruit a
  107 px true box. `cameras.true_box` keeps only the target's real blobs.
- ⚠️ **Join the viewer thread on close.** `viewer.close()` returns before the render loop has stopped.
  Opening the next scene's viewer, or exiting, while it still runs hung the process at exit.
- ⚠️ **Frames through a pipe cost 4 ms.** Two 640x480 frames per call pushed a frame past 20 ms;
  they go through a `/dev/shm` file instead (render + YOLO on both cameras + drawing: p95 ~16 ms).

## Known gaps (block collecting episodes, step 5)

- ⚠️ **Finger stiffness and the jaw angle while holding are guesses.** Push a printed finger at
  mid-length with the scale (force at 5 mm) and read the real follower's `gripper.pos` while it
  holds a cherry tomato; then `FINRAY_K_MID_N_PER_MM` gets set so the sim jaw stalls there too.
- ⚠️ **The lens is a guess from the retail spec** (130° diagonal, 103° horizontal at 1080p), with
  640x480 assumed to be a 4:3 crop. That gives 72° x 52° at 640x480 and ripe fruit 19–36 px across
  (median 26). If the spec guess holds, that's ~4x the ~6 px the older notes estimated. If the
  camera squashes 16:9 instead of cropping, horizontal FOV is wider (set `lens.ASPECT = "squash"`).
  Calibrate before trusting the recall score.
- **Recall score on the flat-colour tier: 21 %** (`eval_yolo_recall.py`, 100 layouts with 2–3 ripe
  fruits, `pretrain_v4`, t 0.25). YOLO puts some box on a ripe fruit every time but mostly calls the
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
- **Arm yaw and target row** (`scene.ARM_YAW_DEG`, `TARGET_ROW`).
- **Contact and plant constants**: break force, truss stiffness, finger stiffness and friction (1.2).
- **Filament colours** (`PLA_RGBA`, `TPU_RGBA`).
- Checked by K7 with the leader (2026-10-20): joint directions and poses (`arm_map.SIGN`).

## Status

1. ✅ Tomato release (`check_tomato_release.py`, all pass)
2. ✅ Scene (`check_scene.py`, `check_grasp.py`, all pass; the unverified list above still applies)
3. ✅ Teleop recorder: K7 drove the sim with the leader (2026-10-20): joints and poses match. `lerobot-record` itself hasn't run a full session yet. Full state + calibration hash go to the state log, not the dataset metadata
4. ✅ YOLO view in sim: lens model, 2–3 ripe fruits, live YOLO view in both recorded cameras, green start-frame target with true-box fallback, recall score 21 % on the flat tier. Camera numbers are spec guesses
5. ⬜ Collect · 6. ⬜ Re-render tiers · 7. ⬜ Real episodes + training
