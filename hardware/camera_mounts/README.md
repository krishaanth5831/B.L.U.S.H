# Camera mounts

- **Birdseye:** the camera mast for the InnoMaker U20CAM-1080P is in
  [`../greenhouse/`](../greenhouse/) (`gh_cam_*` STLs, `birdseye_mast.fs`). It screws
  to the table, **not** the greenhouse frame, because the frame gets bumped. Tape-mark
  its position. ⚠️ If it moves, the recorded dataset no longer matches.
- **Claw (wrist):** put the printed mount for the gripper camera here, plus the small
  LED that keeps fruit lighting constant at every arm pose. It should look along the jaw
  so the target is centred when the jaw is aimed right. Check full range of motion with
  the cable attached before recording.
- **Cart post (dataset v6, rig `trio`):** two fixed cameras on one upright that travels with the arm, 350 mm to its
  left and 100 mm toward the row: `high` with its lens 420 mm above the deck top and `low` at 50 mm, plus the claw
  camera (`sim/scene.py` `CART_CAMS`; how they were picked: `own_sim_measurements.md` section 24). How the post is
  carried on the plank rig's cart, with a drawing and every clearance:
  [`../greenhouse/pipe_rail_rig/README.md`](../greenhouse/pipe_rail_rig/README.md). ⚠️ Proposed, not built, no
  printable clamp drawn yet.
- **Base (sim only so far):** the sim's rig "base" (K7 2026-11-06, `sim/cameras.py`) puts the overview
  camera low on a bracket left of the arm's base instead of the birdseye mast, and every sim truss dataset
  since `blush_sim_scripted_v2` (formerly `blush_sim_truss_v2`) uses it. No printable bracket is drawn yet; its
  pose in `sim/scene.py` is the design guess.

What each camera sees, in the sim (lerobot-mj4 env, a window with the birdseye | claw frames as recorded,
or base | claw with `--rig base`; dataset v6's three with the last line):

```bash
cd sim && python view_scene.py --cams
cd sim && python view_scene.py --cams --rig base
cd sim && python view_scene.py --cams --rig trio --world crate23 --layout-rig base
```
