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
- Built after this: the retreat, the lowering and a "crane" move (pull straight back and up, turn, reach out over
  the crate) all on the cut pose's pitch and tilt. Scored in section 6.

## 6. Gravity-aligned carry scored (60 scenes, 9730000-059, up to 6 retries; the sim unchanged)

- Aligned carry, no fallback to the old detours: 27/60 = 45 %. Today's teacher on the same scenes: 39/60 = 65 %; the
  release-pose-cap variant: 41/60 = 68 %. Won 5 scenes today's teacher loses, lost 17.
- Why it lost, by count: no plan in 21 scenes against 4 (the crane move is the only route and the row blocks its
  pull-out or its turn); retries recovered 6 scenes against 14 (the failing piece, the pull-out, is the same in every
  retry); and of 74 failed aligned attempts 60 were damage on the way to the crate on paths the planner had called
  clear.
- Predicted from these runs (deterministic): aligned as the first try, then today's teacher: 40/60; then the cap
  variant: 42-43/60. One or two scenes, inside the noise. Not worth a 180-scene run as built.
- What the aligned carry did do: the stalk's angle to the pads' line stays put (section 5), i.e. the tilt-driven
  pivot about the pads' normal is gone. That was not what decided success.

## 7. Where the cut truss really is (traces: `sim/whatif/carry_bump_trace.py`)

- In the failed aligned carries it is the cut truss itself that touches the neighbour (never the arm), 2-6 cm from
  where the planner's model has it at that moment.
- The truss's fruit are 35-110 mm off BOTH planner models (as cut, and hanging under the stalk) already when the
  retreat starts, and 60-250 mm off later in the carry. Same with the solver's creep removed (noslip 5), so this is
  not the creep artefact.
- Control: today's teacher on the same four scenes is just as far off its own model (45-110 mm at the retreat,
  up to 180-430 mm near the crate) and passes three of the four on the first plan. So model accuracy is not what
  separates the two; today's routes take the truss away from the row at once, the crane route keeps it beside the
  row for its pull-out and turn.
- Split by axis (gripper frame): the main error is a turn about the stalk's OWN axis, 11-111 degrees, plus 10-30
  degrees about the pads' normal. (Already noted 2026-11-24 in `truss_expert.py`, CARRY_AXIS.)

## 8. When and why the truss rolls about its stalk

- When: nothing moves until the jaw closes. The roll happens in the last 0.2 s of the close, after the blade is
  through (jaw +1.7 degrees) while the jaw snaps on to its shut position (-9 degrees): 0.6 -> 58 -> 89 degrees in
  9730034, -> 60 in 9730005, -> 11 in 9730031 (the sim as it is). It then keeps turning in the retreat and wanders
  over 100 degrees and more during the carry.
- It does not follow the stalk's thickness (radius 1.64 / 1.91 / 1.56 mm for 89 / 60 / 11 degrees).
- Probe 1, ease the jaw back open after the cut (to -4.9, -2.4 or +0.1 degrees instead of -8.7): the stalk rolls back
  at a steady 5-8 degrees per degree of jaw in all three scenes. So moving the jaw while it grips does roll the round
  stalk between the two flat pads, like a pencil between two palms (the pads slide past each other because the jaw
  pivots). But what the snap-shut leaves behind varies (11-89 degrees), so a fixed ease-back over-corrects one scene
  and under-corrects another: 9730031 went from +11 to -64 degrees. Pad force drops from 7.6 N to 3.9-6 N in total.
  Not a fix. Code removed.
- Probe 2, a what-if sawtooth face on the fixed pad (ridges parallel to the stalk, 3 mm pitch, 1.5 mm high, 90 degree
  valleys; everything else unchanged): the roll still came after the cut (71 / 38 / -6 degrees at the end of the
  squeeze), and the cut needed 34-47 N instead of 16-17 N (the stalk sat on a ridge tip). Not a fix as drawn. Code
  removed. Jaw design stays the owner's decision.
- Both probes: three scenes each, one plan each; enough to drop them, not to tune them.

## 9. The neighbour-swing rule

- The judge fails a run when any fruit of another truss is more than 15 mm from where it hung, at any frame
  (`generate_truss.NEIGHBOUR_MOVE_M`, marked a guess on 2026-11-03).
- Of today's teacher's 60 failed scenes (180), 20 end on that rule with no fruit lost: swings of 15-42 mm, one of
  87 mm. Of the cap variant's 54 failures, 12 (16-55 mm).
- That label only names the last retry's fault; an earlier retry may have failed on something else (9730004: six
  retries ended "truss not in the crate" with 9-11 mm swings, the seventh "moved a neighbour 17 mm").
- `base_grid.py` now also logs, per attempt, the largest swing and the verdict with that one rule left out;
  `scripts/grid_ab.py --swing` prints what each limit would have scored. The verdict itself is unchanged.
