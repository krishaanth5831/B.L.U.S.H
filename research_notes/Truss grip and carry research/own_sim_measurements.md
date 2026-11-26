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

## 10. Old jaw against new jaw, same teacher, same 180 scenes (2026-11-26)

The owner's direction since 2026-11-25: keep the old teacher and only adjust it for the new jaw. Three runs on the
180 tuning scenes (9730000-179), up to 6 retries, same base (0.34/0.14), same crate (30 x 28), same start pose:

| Run | Solved | On the first plan |
|---|---|---|
| Old jaw (v5), old teacher (`--jaw v5`) | 141/180 = 78.3 % | 106 |
| New jaw (v6), the old teacher's logic (`BLUSH_V6='{"V6_TEACHER": false, "V6_CARRY": ["rigid", 0]}'`) | 122/180 = 67.8 % | 75 |
| New jaw (v6), today's teacher (the `V6_*` changes on) | 120/180 = 66.7 % | 73 |

- The gap is the jaw: 10.5 points with the same logic (won 13 scenes, lost 32). The `V6_*` teacher changes (slower
  carry, settle above the crate, lift home, hanging fit) together are worth nothing measurable (120 against 122).
- Where the new jaw loses with the same logic (172 scenes all three had finished, scenes that end failed): at the cut
  11 against 2 (neighbour pushed or its fruit knocked off while the jaw closes, never cut), on the carry to the crate
  28 against 19, no plan 11 against 12, at the crate 3 against 4.
- First plans: the truss misses the crate or loses its own fruit while lowering in 30 against 20 scenes; retries
  recover most of them.
- With the neighbour-swing rule left out (second reading only): old jaw 84 %, new jaw with the old logic 79 %, new jaw
  today 73 %. The rule is worth 5-11 points to every run; it does not close the gap.
- ⚠️ The old jaw's 78 % is on today's layout. Its 87/100 and 163/200 were measured on the old base (0.37/0.20), the
  old start pose and the old crate. Old jaw on the old base and start pose, same 180 scenes: section 14 (86 %).
- ⚠️ "Same 180 scenes" is not "same truss" everywhere: in 31 of the 180 the two jaws get a different target truss
  (section 12). On the 149 scenes with the same target: 118 (79 %) / 104 (70 %) / 100 (67 %), old jaw against the new
  jaw with the old logic won 23, lost 9. The reading above stands.

## 11. What the closing jaw does to the stalk (`whatif/carry_bump_trace.py --legs slide_in,settle_at_stalk,close_cut,squeeze,retreat`)

Eight scenes where the new jaw damaged a neighbour at the cut (9730009/012/039/053/065/071/087/117), first plan,
both jaws (`--jaw v5` for the old one). Medians over the eight:

| | Stalk tilt in the pads' plane, peak during the close | Roll about the stalk, end of the close | Roll, end of the squeeze | Fruit off the planner's "as cut" model, end of the close |
|---|---|---|---|---|
| Old jaw (stalk aimed 1 mm off the fixed pad) | 6 deg | 2 deg | 9 deg | 22 mm |
| New jaw, stalk aimed 8 mm off the fixed pad (today) | 12 deg | 22 deg | 40 deg | 37 mm (peak 46) |
| New jaw, aimed 4 mm off | 9 deg | 14 deg | 28 deg | 32 mm |
| New jaw, aimed 2 mm off | 8 deg | 15 deg | 21 deg | 37 mm |
| New jaw, slide in 8 mm off, move sideways to 2 mm, close | 8 deg | 18 deg | 27 deg | 36 mm |

- With the stalk 8 mm from the fixed pad, the moving side reaches it half-way through the close (one pad contact) and
  pushes it across to the fixed pad and on into the pad's travel: about 11 mm sideways before the cut. The truss is
  still on the plant, so it swings like a pendulum: its fruit move 15-30 mm and touch the neighbours (0.1-0.3 N), and
  the neighbour swings past the judge's 15 mm. The old jaw's pads meet the stalk only in the last frame of the close.
- Old catch map (`outputs/v6/catch_map_jaw_v6_*.jsonl`, 9 scenes, 2026-11-24) says the same from another side: the
  truss stays put after the cut (drift under 15 mm) in 9 of 9 scenes with the stalk 2-5 mm from the fixed pad, in 4-5
  of 9 with it 8-11 mm away. The 8 mm aim was chosen as the middle of the zone, for room on the fixed finger's side.
- Aiming closer takes away about a third of the tilt and half of the roll. What is left (roll 15-27 deg against 2-9)
  is the jaw: flat pads, 3 mm proud of the blades, 5.5 mm of travel.
- `V6_MEET_FROM_M` (off by default): the slide runs with the stalk 8 mm from the fixed pad, the last way-points of
  `slide_in` move the jaw sideways until it is `V6_AIM_FROM_PAD_M` away. No new leg name.

## 12. The same seed is not the same truss across jaws, aim points and bases (found 2026-11-26)

`truss_scene.build` draws the vines and trusses from the seed, then makes ripe only trusses the arm can cut from where
it stands (`cut_ik` at three points of the cut zone and `truss_expert.approach_ok`). Both depend on the jaw, the aim
point in the jaw, the base and the start pose. So which truss is ripe, and so the target, differs between runs.
⚠️ The layout can differ too: `build` draws the whole layout again (up to 4 draws) while no truss is reachable, and
`_build_once` removes trusses that touch the arm at the start pose. So the same name ("s0t1") in two runs is not
proof of the same truss. Every "same target" count in sections 10, 13, 14 and 16 was matched by name and is being
checked against a build-only census (`base_grid.py --build-only`: every truss's position, which are reachable, the
target's position). Targets by name:

| Compared runs | Scenes of 180 with a different target truss |
|---|---|
| Old jaw against new jaw (same base, same start pose) | 31 |
| New jaw, aim 2 mm against aim 8 mm | 14 |
| Old base and start pose against today's (old jaw) | 132 |
| New jaw, old logic against today's logic | 0 |

- Every paired count across those settings is a count per scene, not per truss. Each comparison below also gives the
  scenes with the same target.
- The traces: release trace (section 15) same truss in 5 of 6 scenes (9730075 is another truss for the old jaw); cut
  trace (section 11) the 2 mm aim has another truss in 2 of 8 (9730012, 9730053), the old jaw in 2 of 8 (9730012,
  9730117).

## 13. The closer aim, scored (r19: `BLUSH_V6='{"V6_AIM_FROM_PAD_M": 0.002}'`, today's teacher otherwise)

| Scenes | Aim 8 mm (today) | Aim 2 mm | Won | Lost |
|---|---|---|---|---|
| 9730000-089, the half the aim was first scored on | 54/90 | 64/90 | 12 | 2 |
| 9730090-179, held out | 66/90 | 67/90 | 6 | 5 |
| All 180 | 120 (66.7 %) | 131 (72.8 %) | 18 | 7 |
| Same target only: first half / held out / all | 50/81, 63/85, 113/166 | 58/81, 62/85, 120/166 | 9, 4, 13 | 1, 5, 6 |

- The mechanism holds: scenes that end on a neighbour pushed while the jaw slides in, closes, squeezes or retreats go
  from 7 to 0. Six of those 7 sit in the first half, which is why the held-out half shows no gain. As a score gain the
  closer aim is not confirmed on the held-out half.
- What it costs: plans that end with the truss not (all) in the crate go from 104 to 123 (scenes ending on it 6 ->
  11); plans that lose own fruit go from 158 to 125. The 7 lost scenes end on own fruit off in the carry or the
  return (3), the crate missed (2), a neighbour pushed in the return (1) and no plan (1, slide hits). With the 8 mm
  aim all 7 passed on their first or second plan.
- No stale copy of the aim: every user reads it at call time through `truss_ik.cut_point_in_gripper` (IK, carry
  check, hanging model, release fit, the scene builder).
- **Default since 2026-11-26: 2 mm** (the owner: "turn on the closer aim"; commit edd5fdb,
  `truss_ik.V6_AIM_FROM_PAD_M`). ⚠️ "Today's teacher" in sections 10 to 16 means the 8 mm aim of that morning. The
  reference for the current default is r19: 131/180 = 72.8 %. `BLUSH_V6='{"V6_AIM_FROM_PAD_M": 0.008}'` brings
  the old aim back.
- "Slide in 8 mm off, move sideways to 2 mm, then close" (r20: `{"V6_AIM_FROM_PAD_M": 0.002, "V6_MEET_FROM_M":
  0.008}`, today's teacher otherwise): 129/180 = 71.7 %. Against the 8 mm aim: first half 54 -> 62, held-out half
  66 -> 67 (won 18, lost 9). Against the plain 2 mm aim: won 2, lost 4, and two scenes end on a joint stalled at its
  force limit, which no other run has. No gain over the plain closer aim; `V6_MEET_FROM_M` stays off.

## 14. Old jaw on the old base and the old start pose (r22: `--jaw v5 --base 0.37,0.20 --old-start`, crate 30 x 28)

| Old jaw, old teacher | Solved | On the first plan | No plan | Damage or miss |
|---|---|---|---|---|
| Old base 0.37/0.20 and old start pose | 155/180 = 86.1 % | 139 | 23 | 2 |
| Base 0.34/0.14 and its searched start pose (r17) | 141/180 = 78.3 % | 106 | 13 | 26 |

- This is where the remembered "high 80 %" comes from: old base and old start pose, even with the new crate. It
  solves 155 of the 157 scenes it finds a plan for.
- The kind of failure changes: at the old base almost nothing goes wrong once a plan exists (plans that lose own
  fruit: 25 against 131; plans that miss the crate: 9 against 38). What fails is planning: slide hits 10, no IK along
  the slide 8, retreat -> crate blocked 4, 1 other.
- ⚠️ Not truss for truss: 132 of the 180 scenes have a different target (section 12; the old base gets truss t0 in 81
  scenes and t1 in 88, base 0.34/0.14 gets t0 in 18 and t1 in 146). All 180: won 37, lost 23. The 48 scenes with the
  same target: 42 against 36 solved (won 12, lost 6), damage or miss 0 against 5.
- ⚠️ Base and start pose changed together; this run cannot say which of the two it is.
- Without the neighbour-swing rule (second reading only): 87 % against 84 %.
- Base 0.34/0.14 was picked on the plan rate (plan-only census 91 %) and the claw camera's angle to the row (11 deg),
  with scores that could not be told apart at 30 scenes (21-23 of 30). Going back is the owner's decision.
- New jaw on the old base and start pose: section 16.

## 15. Where the truss is when the jaw opens over the crate (`whatif/release_trace.py`)

Six scenes that keep missing the crate with the new jaw (9730047/051/054/058/075/080), three plans each, today's
teacher logic:

| | Plans ok | Own fruit's room to the crate's inner walls when the jaw opens |
|---|---|---|
| New jaw, aim 8 mm | 0 of 18 | outside in 16 of 18 plans (-4 to -214 mm); +48 and +86 mm in two plans of 9730051 |
| New jaw, aim 2 mm | 0 of 18 | outside in 18 of 18 plans (-3 to -210 mm) |
| Old jaw | 4 of 9 | +22 to +72 mm in the 4 plans that pass; -61 to -80 mm in 4 that fail, +32 mm in 1 |

- With the new jaw the fruit are already outside the walls when the jaw opens (34 of 36 plans), often with the truss
  hanging straight down and touching the table outside the wall, while the planner's fit expected it inside.
- ⚠️ This reads today's teacher logic (the `V6_*` changes), not the jaw alone: section 10's first plans miss the crate
  18 times with the old logic and 33 times with today's. The run that separates the two is r21 (new jaw, old logic,
  aim 2 mm).

## 16. New jaw with the old teacher's logic, both bases, both aims (r18, r21, r23, r24; 2026-11-26)

All on the 180 tuning scenes, up to 6 retries, crate 30 x 28. "Old logic" = `{"V6_TEACHER": false, "V6_CARRY":
["rigid", 0]}`; aim = `V6_AIM_FROM_PAD_M`; old base = `--base 0.37,0.20 --old-start`.

| Jaw | Teacher logic | Stalk aimed | Old base and old start pose | Base 0.34/0.14 and its start pose |
|---|---|---|---|---|
| Old | old | 1 mm off the fixed pad | 155 = 86.1 % (first plan 139) | 141 = 78.3 % (106) |
| New | old | 8 mm | 142 = 78.9 % (110) | 122 = 67.8 % (75) |
| New | old | 2 mm | 145 = 80.6 % (114) | 128 = 71.1 % (77) |
| New | today's (`V6_*` on) | 8 mm | not run | 120 = 66.7 % (73) |
| New | today's (`V6_*` on) | 2 mm | not run | 131 = 72.8 % (73) |

- **Where the arm stands is the largest effect, for either jaw:** old base against 0.34/0.14 is +14 scenes with the
  old jaw, +20 with the new jaw at 8 mm (won 48, lost 28) and +17 at 2 mm (won 44, lost 27). On the scenes with the
  same target truss (section 12): 37 against 33 of 49, and 43 against 36 of 50.
- **The jaw's gap on the old base:** 13 scenes at 8 mm (won 2, lost 15), 10 scenes at 2 mm (won 3, lost 13). Same
  target truss only: 132 against 141 of 160 (won 0, lost 9) and 138 against 143 of 163 (won 1, lost 6), so 85 %
  against 88 % with the closer aim.
- **What the new jaw still loses on the old base (2 mm):** own fruit knocked off with the truss in hand in 9 scenes
  (to the crate 6, lowering 2, 1 other) against 1 with the old jaw; plans that lose own fruit 82 against 25. No plan
  is the same for both jaws (24 against 23: slide hits 10, no IK along the slide 9, retreat -> crate blocked 4).
- **The closer aim, three paired readings:** old base +3 (won 9, lost 6; same target won 4, lost 3); base 0.34/0.14
  with the old logic +6 (won 17, lost 11; first half 54 -> 62, held-out half 68 -> 66; same target won 10, lost 10);
  with today's logic +11 (section 13). Never worse on all 180, never confirmed on a held-out half or on the same
  trusses. It does end the cut-stage failures in every run (old base: 1 -> 0; 0.34/0.14 old logic: 6 -> 0).
- **The `V6_*` teacher changes as a package:** nothing measurable at 0.34/0.14 (aim 8 mm: won 12, lost 14; aim 2 mm:
  won 13, lost 10). Not run on the old base.
- Without the neighbour-swing rule (second reading only), old base: old jaw 87 %, new jaw 83 % at either aim.
