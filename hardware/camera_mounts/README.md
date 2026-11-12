# Camera mounts

- **Birdseye:** the camera mast for the InnoMaker U20CAM-1080P is in
  [`../greenhouse/`](../greenhouse/) (`gh_cam_*` STLs, `birdseye_mast.fs`). It screws
  to the table, **not** the greenhouse frame, because the frame gets bumped. Tape-mark
  its position. ⚠️ If it moves, the recorded dataset no longer matches.
- **Claw (wrist):** put the printed mount for the gripper camera here, plus the small
  LED that keeps fruit lighting constant at every arm pose. It should look along the jaw
  so the target is centred when the jaw is aimed right. Check full range of motion with
  the cable attached before recording.
- **Base (sim only so far):** the sim's rig "base" (K7 2026-11-06, `sim/cameras.py`) puts the overview
  camera low on a bracket left of the arm's base instead of the birdseye mast, and the truss dataset
  `blush_sim_truss_v2` uses it. No bracket is drawn yet; its pose in `sim/scene.py` is the design guess.

What each camera sees, in the sim (lerobot env, a window with the birdseye | claw frames as recorded,
or base | claw with `--rig base`):

```bash
cd sim && python view_scene.py --cams
cd sim && python view_scene.py --cams --rig base
```
