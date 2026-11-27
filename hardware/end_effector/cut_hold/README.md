# Cut-and-hold jaw v6 (K7, 2026-11-24)

The truss harvest's jaw for dataset v6. One STS3215 servo, the stock fixed and moving jaws, each reprinted as
**one rigid PLA part**: the stock servo end with a new finger grown out of it. Along a 36 mm zone of each finger:

- **blade**: a 9 mm snap-off cutter blade (0.38–0.40 mm), 3 mm proud of the inner face, held by 2 × M2 pinch screws.
  The two blades sit in neighbouring planes and cross by 2 mm at the closed stop (bypass shears).
- **clamp pad**: a TPU 95A face strip on 2 compression springs, 3 mm proud of the blade edge, so **both pads clamp the
  stalk before either blade touches it** (v5's pads were 1 mm proud: 46 of 51 trusses the policy lost at the cut had
  one pad pressing, post-mortem 4.6).
- **flared tips**: the last 12 mm of each finger bends 6 mm outward, a V mouth that steers a stalk that is a little to
  the side into the zone.

The inner faces are parallel and 4 mm apart at the closed stop, so the jaw still closes at the `follower_white`
calibration's −10° and the calibration stays valid.

| | v5 jaw (every dataset so far) | v6 jaw |
|---|---|---|
| Cutting + holding zone | 16 mm at the fingertips | 36 mm, 47–80 mm from the hinge |
| Pads | 1 mm past the blade, 1 N/mm | 3 mm past the blade, 0.5 N/mm, 1.5 N preload, 5.5 mm travel |
| Fingers | fin-ray ladder, stiffened | solid PLA |
| Teacher's aim (sideways) | 1 mm off the fixed pad | 8 mm off the fixed pad (`truss_ik.V6_AIM_FROM_PAD_M`, set by the catch map) |

## Commands

```bash
# a throwaway venv: numpy shapely trimesh manifold3d matplotlib networkx
python cut_hold_gripper.py check     # build + checks + force budget + preview.png
python cut_hold_gripper.py build     # also out/*.stl (print) and ../../../sim/assets/cut_hold/ (the simulator's jaw)
python view_cut_hold.py              # a window with a jaw-angle slider
# in sim/, lerobot-mj4 env:
python check_truss_cut.py --jaw v6 --windowed    # watch it cut and hold a truss in the MuJoCo window
```

One script makes the printed parts and the simulator's jaw (`scene.add_jaw_v6`, `jaw="v6"` in the plugin config),
so they cannot drift apart. The v5 jaw stays the default everywhere until v6 is approved.

## Force budget (from `check`)

At lerobot's 50 % gripper torque the moving finger pushes 35 N at the top of the zone and 21 N at its tip end. Cutting
a 5 mm stalk needs 10.7 N (2.1 N/mm, SHEAR's placeholder, ⚠️ unverified for tomato peduncles) plus about 4 N of pad
springs: 14.7 N. If the servo drops to its 25 % overload limit (stalled too long) the tip end gives 10.5 N, which is
not enough for a 5 mm stalk; the zone's upper half still is.

## Print fits (K7 2026-11-24: no zero-gap fits; starting values, tune after a test print)

| Fit | Part | Hole / pocket | Clearance per side |
|---|---|---|---|
| Blade in its slot (snug, the pinch screws close it) | 0.40 mm | 0.70 mm | 0.15 mm (`SLOT_CLR`) |
| M2 pinch screw | 2.0 mm | 2.5 mm hole | 0.25 mm (`M2_D`) |
| Spring in its pocket (must move freely) | 4.0 mm OD | 4.8 mm | 0.4 mm (`SPRING_D`) |
| TPU pad face in its pocket (slides) | 12.9 × 36 mm | 13.5 × 38 mm | 0.3 / 1.0 mm (`FIT_SLIDE`) |

The servo end of each jaw is the stock SO-101 part, unchanged, so it fits the servo and horn as the stock print does.

## Onshape

`B.L.U.S.H v6 cut-and-hold jaw (2026-11-24)` in K7's Onshape: the parts imported as mesh bodies, assembled with the
moving jaw open 25 deg ("Assembly 1"). The Python script stays the source: after a change, re-import.

## Bill of materials (per jaw pair) ⚠️ unverified, nothing printed yet

- 2 × PLA/PETG jaws (`out/jaw_fixed_cut_hold_PLA.stl`, `out/jaw_moving_cut_hold_PLA.stl`), 4 walls, 40 %+ infill
- 2 × 9 mm snap-off blade, snapped to about 36 mm
- 2 × TPU 95A pad face (`out/pad_face_TPU_x2.stl`, print 2)
- 4 × compression spring, about 4 mm OD × 10 mm, about 0.25 N/mm
- 4 × M2 × 20 screw + nut (blade pinch)
- 1 × fit-test print first: print one jaw's zone section and try the blade, a spring and the pad before printing both jaws

## Results

| Date | What | Result |
|---|---|---|
| 2026-11-24 | `check_truss_cut.py --jaw v6` (sim) | 9/9 PASS: 3, 4 and 5 mm stalks cut 0.44–0.46 s into the close, held through the shake (0.9–10.9 mm drift), no fruit off. v5 on the same check: 8/9 (5 mm stalk: a fruit off in the shake) |
