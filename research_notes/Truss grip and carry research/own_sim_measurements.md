# Own measurements in the project's simulation (2026-11-25)

Source: the project's MuJoCo 4.0.1 simulation of the 5-joint arm with the "v6" cut-and-hold jaw (two flat
spring-loaded pads, 13 mm along the stalk x 36 mm along the finger, about 4 N each). Measured by the project itself,
not from the literature. Scripts: `sim/whatif/pad_hold_torque.py`, `sim/base_grid.py`, `sim/scripts/grid_ab.py`.
Label for the report: "measured in the project's simulation".

## 1. Scripted teacher success before any change (180 fixed test scenes, up to 6 retries per scene)

- Today's teacher: 120/180 = 66.7 %.
- Variant that refuses release poses where the truss would have turned more than 20 degrees in the pads' plane, and
  skips retries repeating a failed release: 126/180 = 70.0 % (won 12 scenes, lost 6; sign test p about 0.24, so not
  proven).
- Best of three planner variants per scene: 132/180 = 73 %; 48 scenes fail in all three. Tuning release poses is spent.
- The 54 failures of the best variant: no plan 24 (all at the carried-truss check, none at the approach), fruit knocked
  off or neighbour pushed on the way to the crate 14, truss not fully in the crate 6, neighbour bumped at the cut 5,
  damage on the way home 3, other 2. 95 % allows 9 failures.

## 2. Idealised-hold diagnostic (NOT a candidate fix; the owner ruled it out because it cannot be built)

- With the cut truss fixed rigidly to the gripper from the cut to the release (108 scenes finished): 77.8 %, against
  62 % for today's teacher on the same scenes. So turning in the grip explains about half of the gap; what still
  failed with a perfect hold: fruit off on the way to the crate 6, neighbour bumped at the cut 4 / on the way 3 /
  retreating 2, no plan 5.

## 3. What loads the pads and what they can resist (4 scenes where the truss turned 70-87 degrees)

- Contacts: 4 (2 per pad), at the two ends of the 13 mm stalk-pad overlap; total normal force 8.1 N.
- Truss in these scenes: 53 g, centre of mass 115 mm from the grip along the stalk; at the cut the stalk points almost
  straight down (gravity nearly along the stalk), so the truss already hangs.
- Gravity torque about the pads' normal (the pivot axis) during the carry: starts at 0.005-0.02 N.m, peaks at
  0.05-0.065 N.m when the planned path turns the gripper 90-120 degrees against gravity; the contacts carry up to
  0.07 N.m at the peaks (dynamic).
- Friction capacity about that axis computed from the contact forces and levers (mu 1.5, two end-point contacts per
  pad): 0.09-0.10 N.m. Spin capacity about the stalk's own axis: about 0.024 N.m.
- The stalk turned steadily (about 13 degrees/s) while the load was only 5-50 % of that capacity.

## 4. Creep versus real slip

- Same 4 scenes with MuJoCo's noslip pass on (5 iterations), nothing else changed: for most of the carry the stalk
  stayed within about 1-6 degrees in three scenes (it had drifted to 40-70 degrees), about 12 degrees in the fourth;
  but it still slipped 20-80 degrees later in every scene, when the carried torque reached about 0.07 N.m (roughly 70 %
  of the computed capacity).
- Reading: the slow drift is the solver's soft-friction creep (an artefact that exaggerates turning); the late jumps
  are a real friction release. With a realistic rubber-on-stalk friction of 0.8-1.0 instead of the model's 1.5, and
  an evenly loaded line contact instead of two end points, a real flat pad would hold about 0.02-0.04 N.m, so the real
  truss would also turn on these paths.

## 5. Gravity-aligned carry, first test (new opt-in switch `V6_ALIGN`, off by default)

- Idea: gravity as seen from the gripper depends only on the fingers' pitch and the jaw's tilt; the pan joint turns
  about the vertical. Release poses are built at the cut pose's own pitch and tilt (the existing IK holds both at any
  reachable point), and the lowering keeps them, so the truss keeps hanging as it hung at the cut. The carried-truss
  model "as at the cut" is then the correct one.
- Same 4 scenes, sim otherwise unchanged (creep still present):
  - 9730003: stalk angle stayed 10-14 degrees (was 87); success. Planned gripper-vs-gravity change 1.2 degrees.
  - 9730018: stayed 19-21 degrees (was 85); success. Planned change 0.1 degrees.
  - 9730005: the straight move to the crate was blocked, the detour used fixed via-poses that do not keep the tilt
    (planned change 116 degrees): the stalk swung to 55 degrees and back; success anyway.
  - 9730015: retreat variant with free orientation (planned change 16 degrees): fruit came off on the way; failure.
- Next step being built: via-poses and the retreat projected onto the same pitch and tilt, then the 180-scene paired
  comparison. Not yet tested: placing a vertically hanging truss in a 5 cm high crate (release above the floor and
  let it tip, or lay it down).
