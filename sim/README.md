# sim/: leader arm → MuJoCo follower

The real leader arm drives a simulated SO-101 with the fin-ray gripper in the mini greenhouse. The sim
cameras record the episodes as **co-training data for π0.5**, next to the real-arm episodes. Plan and
reasoning: vault `08 SOARM101 Training/(C) BLUSH Sim Teleop Plan.md`.

## Watch it

Run in the `lerobot` env (`conda activate lerobot`), from `sim/`:

```bash
python view_scene.py --cams              # greenhouse window + claw | birdseye images (green = true target box)
python view_scene.py --leader --cams     # the real leader arm drives the sim arm (read only, records nothing)
python check_tomato_release.py --windowed   # one truss: double-click a tomato, Ctrl + right-drag to pluck it
```

Without `--leader`, move the arm with the actuator sliders in the viewer's right panel (Control).
Ctrl + right-drag pulls a tomato by hand; plucks are printed in the terminal. Wayland prints
`libdecor` / `GLFWError: window position` warnings, which are harmless.

## Setup (once)

```bash
./fetch_so101.sh                  # SO-101 MJCF + meshes into assets/so101 (gitignored, ~16 MB)
python build_finray_meshes.py     # only after changing the fin-ray design; needs numpy shapely trimesh manifold3d
```

`assets/finray/` is committed: the printed fin-ray parts as metre meshes in the MJCF body frames.

## Checks

```bash
python check_tomato_release.py    # step 1: the tether hangs, releases above threshold, falls free
python check_scene.py             # step 2: unit mapping, 200 layouts, cameras, grasp + pluck through contact
```

Both print PASS/FAIL per check and write nothing.

## Files

| File | What |
|---|---|
| `tomato.py` | Trusses + tomatoes. Each fruit hangs on a soft weld (the tether). `Detacher` switches the weld off once its force stays above the fruit's break force (OrchardBench fruit logic) |
| `scene.py` | Greenhouse (laser-cut dimensions), SO-101 + fin-ray fingers + palm, vines, `birdseye` + `arm/claw` cameras. Exactly one ripe fruit within reach |
| `arm_map.py` | lerobot units (`use_degrees=false`) ↔ sim joint angles through `follower_white.json`, the way the real follower converts them |
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

## Unverified (needs the real rig)

- **Joint signs and zero offsets** (`arm_map.SIGN`): check with `view_scene.py --leader`.
- **Camera intrinsics + lens distortion** (fovy 60 is a design guess) and the **claw camera mount**
  (the SO-ARM100 wrist camera mount is a placeholder).
- **Arm yaw and target row** (`scene.ARM_YAW_DEG`, `TARGET_ROW`).
- **Contact and plant constants**: break force, truss stiffness, finger friction and softness, and
  the jaw angle while holding a fruit (sim: −10°).
- **Filament colours** (`PLA_RGBA`, `TPU_RGBA`).

## Status

1. ✅ Tomato release (`check_tomato_release.py`, all pass)
2. ✅ Scene (`check_scene.py`, all pass; the unverified list above still applies)
3. ⬜ Teleop recorder: log full state, export LeRobot `so101_follower` with the calibration hash in the metadata
4. ⬜ YOLO overlay in sim (YOLO runs in the `ripeness` env, so it needs a separate process)
5. ⬜ Collect · 6. ⬜ Re-render tiers · 7. ⬜ Real episodes + training
