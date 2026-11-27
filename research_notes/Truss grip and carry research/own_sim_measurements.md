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

## 17. What each setup is given: the build-only census (`base_grid.py --build-only`, 2026-11-26)

`outputs/v6/base_grid_cen_*`: per seed every truss (name, knuckle position, fruit), which are reachable, the target.
⚠️ Load with `_[0-9]*.jsonl` (tag `cen_v6a8_nb` is a prefix of `cen_v6a8_nb_oldstart`).

| Setup | Trusses per scene | Fruit per scene | Reachable per scene |
|---|---|---|---|
| Base 0.34/0.14, searched start pose (either jaw) | 3.9 | 26 | 2.0 |
| Old base 0.37/0.20, old start pose (either jaw) | 2.2 | 14 | 1.3 |
| Old base, searched start pose (from the run's log, 76 scenes) | 3.3 | 23 | |
| Base 0.34/0.14, old start pose (from the run's log, 43 scenes) | 4.5 | 30 | |

- **The old setup's scenes are about half as full.** The builder removes every truss that touches the arm at its start
  pose, and at the old base the old start pose stands in the plants. Between the two setups only 9 to 11 of the 180
  layouts are identical (114 are the same draw with trusses removed, 55 a different draw). On those identical layouts
  both setups solve the same number (7 against 7 with the old jaw, n = 9; 7 against 7 with the new jaw, n = 11).
- **Base apart from start pose (new jaw, old logic, aim 8 mm; both runs stopped early when the owner changed
  direction):** old base with the searched start pose 44 of 75 against 45 for base 0.34/0.14 on the same seeds; base
  0.34/0.14 with the old start pose 21 of 41 against 27, with no plan in 10 against 3. So the old base is no better
  in itself; the old setup's higher score is its emptier scenes.
- **The jaw's gap on the identical layout and the same target truss** (replaces the name-matched "same target" counts
  of sections 10, 13, 14 and 16): base 0.34/0.14, 139 scenes: old jaw 113 (81 %), new jaw with the old logic and the
  8 mm aim 96 (69 %), won 6, lost 23. Old base, 157 scenes: 138 (88 %) against 130 (83 %), won 0, lost 8.
- Left unfinished at the direction change: r26 (75 of 180), r27 (41), r29 (`V6_FIT` rigid, 0), r30 (`V6_CARRY`
  rigid, plan-only, partial), the census of the 2 mm arms.

## 18. Back to the old setup (owner, 2026-11-26: "we will not be changing the gripper or the teacher")

Old jaw (v5), old teacher, old base 0.37/0.20, old start pose, crate 30 x 28: these are the code's defaults.

| Scenes | Solved | On the first plan | Has a plan | No plan | No ripe truss | Damage |
|---|---|---|---|---|---|---|
| Tuning, 9730000-179 (r22) | 155 = 86.1 % | 139 | 157 | 22 | 1 | 2 |
| Fresh, 9730180-359 (r31) | 143 = 79.4 % | 121 | 148 | 29 | 3 | 5 |
| Both | 298 of 360 = 82.8 % | 260 | 305 | 51 | 4 | 7 |

- Once it has a plan it hardly ever fails: 298 of 305. Getting above 90 % means finding plans.
- What blocks the 22 no-plan tuning scenes (`blocked_by`): the way in from the start pose 9 (in 4 the plants already
  touch the arm after they settle: "start pose not clear"), the way in and the slide-in 4, the slide-in 5 (neighbour
  fruit or stalks, the target's own fruit, the arm itself), the carry to the crate 4. In about 6 the blocker is the
  target's own fruit.
- By the target's knuckle height: 130-180 mm 43 of 49, 180-230 mm 35 of 37, 230-270 mm 24 of 28, 270-300 mm 22 of
  31 (9 with no plan), 300-330 mm 31 of 34.
- The teacher's margins at the start are 4 mm (neighbours) and 2 mm (own fruit), under the builder's 10 mm: the 4
  "start pose not clear" scenes come from the builder checking before the plants settle and the teacher after.

## 19. A bigger crate at the old setup (what-if, `base_grid.py --crate 0.40,0.34,0.6325,0.20`)

40 x 34 cm in place of 30 x 28 (10 cm past the end frame in +x, 6 cm toward the row in +y), the plants pinned to the
standard crate (`truss_scene.KEEP_OUT_CRATE`): the same 180 scenes, the same target in all 180. ⚠️ 30 x 28 is the
most that fits the greenhouse; this only asks whether size matters.

| Crate | Solved | On the first plan | No plan | Plans that miss the crate | Plans that lose own fruit |
|---|---|---|---|---|---|
| 30 x 28 | 155 | 139 | 22 | 9 | 25 |
| 40 x 34 | 156 | 120 | 22 | 3 | 62 |

Won 1 (9730176), lost 0. The crate is not what holds the old setup back. With more room the release fit takes poses
that lose fruit more often on the first plan (28 first plans lost, 25 of them on fruit off; 9 gained).

## 20. More fixed via-poses for the scenes with no plan (changes "3 and 4", search seeds 9740000-299)

Old setup, `outputs/v6/via_old/`. Teacher side: `truss_teacher.VIA_NORM_V5B`, used only in a last pass after phase C
found nothing more, so a scene that had a plan keeps it (30 tuning scenes reproduce record for record with the list
empty, r33). **Result: +4 of 180 tuning scenes, +2 of 180 fresh ones, none lost. Left OFF** (owner, 2026-11-26, before
the result: "if this test doesnt seem to give good results then we ... just stick with this teacher"); the three
poses stay in the code as `VIA_NORM_V5B_FOUND`.

- Pass 1 (`whatif/via_search_crate.py collect --base 0.37,0.20 --old-start --jaw v5`): 294 scenes with a target, 33
  with no plan (11 %). A blocked way in is among the reasons in 30 of them, a blocked carry in 3.
- **The way in (change 3,** `VIA_EXTRA=1 whatif/via_search.py collect`): 329 blocked candidates in 29 scenes, an RRT
  route (8000 iterations, 2 tries) for 116 of them in 18 scenes, none in 11 scenes. Most routes need 3 to 5
  way-points; the teacher allows one or two. 464 way-points, 247 candidate poses (12 norm units apart); one candidate,
  alone or chained with one existing via, clears the way in for 6 of the 29 scenes. Kept: 3 poses (+3, +2, +1).
  With them the teacher plans 5 of those 6 search scenes, each through a `v5b:` way in (`fires_planonly.jsonl`).

| Scenes (the unsolved ones re-run, r34 / r35) | Old teacher | With the 3 poses | Newly a plan | Lost |
|---|---|---|---|---|
| Tuning, 9730000-179 | 155 = 86.1 % | 159 = 88.3 % | 5 (9730006, 017, 091, 175 pass; 144 moves a neighbour 27 mm) | 0 |
| Fresh, 9730180-359 | 143 = 79.4 % | 145 = 80.6 % | 2 (9730284, 293 pass) | 0 |
| Both | 298 of 360 = 82.8 % | 304 = 84.4 % | 7 | 0 |

  Only the unsolved scenes were re-run: a solved scene stops at its successful plan, which comes before the last pass.
  Cost: a scene with no plan takes about twice as long to give up.
- **The carry (change 4): no pose found.** 18 blocked carries in the 3 scenes, no RRT route for any. Is that the
  search's own fault? ⚠️ `rrt.plan` gives up at once when the start touches something, while the teacher's moves
  never check a path's first way-point (`Checker.path_ok`); the tool now tries a first hop off such a start by the
  teacher's rule. Control (`--control 4`, scenes 9740000-029): the RRT finds a route for 28 of 28 carries the teacher
  can do (4 of them needed a via-pose). Re-check of the blocked ones (`crate_diag_blocked.jsonl`), first scene
  (9740086): in 4 of 6 the retreat ends with the carried truss within 1 cm of a neighbour's fruit and no first hop
  off it exists in 300 tries (no via-pose can help; it is the retreat's end, not the route), in 2 start and goal are
  free and the RRT finds nothing. The other two scenes (9740204, 9740281, finished 21:25): 12 of 12 the same first
  kind, the carried truss on a neighbour's fruit (`s2t2_t6_g`, `s2t1_t5_g`) with no first hop off it. All three
  scenes: 16 of 18 blocked at the retreat's end, 2 with free ends and no route, 0 with a route. A via-pose cannot
  help a carry that starts in contact: what would is the retreat ending further from the neighbour, a teacher
  change, and the teacher is closed (section 18).
- What is left of the no-plan scenes would need the two options not taken: the builder re-checking the start pose
  after the plants settle (4 of the 22 tuning no-plans), and only ripening trusses the teacher has a plan for.

## 21. The policy side of v6: what was decided, and the two worlds (owner, 2026-11-26 evening)

- Owner: the teacher, the gripper and the base stay as they are (section 18). What changes: "more varied episodes. 3
  cameras instead of 2 but also placed in a more calculated location and position", and the post-mortem's fixes.
  Then: the new episodes go "to the same dataset as before", "another 600 episodes on top of the previous 800",
  training again at batch 64 and learning rate 3e-5. Nothing is recorded, re-made or trained without the owner's go.
- **A seed is a scene only together with its world.** A seed draws its plants around the crate, and only trusses the
  arm can reach from where it stands become ripe, so the crate, the brackets standing in the scene, the jaw, the base
  and the start pose all decide which scene a seed gives (section 12). Dataset v5 and `heldout_v5*.jsonl` were made
  with the 23 x 28 cm crate; the code default has been 30 x 28 since 2026-11-24. `truss_scene.WORLDS` names both
  after their crate: `crate23` (dataset v5, and so dataset v6 and every test of a policy trained on it) and
  `crate30` (the code default, no dataset yet). ⚠️ The names are not dataset versions: `--world crate23` is right for
  v6. `rebuild_v5_check.py`: 14 of 14 logged v5 scenes rebuild to the logged model, target and cue in `crate23`.
- The same seed, both worlds (9730000): `crate23` gives 311 position values, `crate30` 135. In `crate23` the scenes
  are fuller (3.3 trusses, 22 fruit per scene on v5's own scenes; 2.2 and 14 on the tuning seeds in `crate30`) and
  the cut points reach x 0.47 m (0.40 in `crate30`).
- **The scene generator of dataset v6 = the one of v5**: world `crate23`, scenes decided with the old camera bracket
  standing in them (`layout_rig="base"`), shown and simulated with the new brackets (`rig="trio"`). So the 600 new
  episodes, the 800 old ones and the held-out sets come from one generator; only the cameras differ. Options, all
  off by default: `generate_truss.py --world --rig --layout-rig`, `eval_truss.py --world --layout-rig`,
  `base_grid.py --world --rig --layout-rig`.

## 22. Can the old 800 episodes get new pictures? Yes: the state logs hold every frame (`remake_pictures.py`)

- The recorder logged the simulator's full state at every recorded frame (`~/.cache/blush_sim/states`, 800 files).
  `remake_pictures.py align` (read-only, all 800 episodes, 383,877 frames): every episode has one logged state more
  than it has frames, the logged joint reading of frame k equals the dataset's `observation.state[k]` (largest
  difference 3.8e-6 norm units = float32 rounding), and the logged cue equals the dataset's in all 800.
- `remake_pictures.py check --episodes 1,2` (read-only; fetched 336 MB of the old videos, deleted after): the old rig
  re-rendered from the logs against the stored v5 frames, every 10th frame: mean difference 2.2-2.6 grey levels of
  255, worst frame 4.6, through the cut and the carry. v5's AV1 storage alone costs about 1.9 (picture_store.py), a
  truss or the arm in another place costs far more, so the logs give back the recorded pictures.
- So a re-make is a render job: per frame the logged state is set, the new rig's cameras rendered, the joint
  readings and commands copied from the dataset bit for bit. What differs from a recording: the pictures show the
  state at the end of the simulator step (the recorder rendered one 2 ms physics step earlier), MuJoCo 4.0.1 renders
  them (v5: 3.8.1), and the brackets in view are the new rig's.
- ⚠️ The state logs are the only source of the old episodes' pictures in any new form, and they sit in a cache
  folder (one copy). `remake_pictures.py part` (writes a dataset part) has NOT been run.

## 23. Rig "trio", picture storage, the guards and the notebook: what has run and what has not

| Piece | Check | Result |
|---|---|---|
| Rig `trio` (cameras high, low, claw since the pick of section 24; `layout_rig`) | rig `base` scenes before and after the change | identical (geoms, state, ripe truss); only the model's camera count 3 -> 5 |
| | `remake_pictures.py dry --start-only` on 2 old episodes, provisional cameras | both rebuilt as recorded, a box and a cut pixel on both fixed cameras (all 800 with the picked cameras: section 26) |
| Picture storage (`picture_store.py`) | `selfcheck`, `gatecheck` | H.264, every frame a key frame, constant qp: a stored frame equals its picture sent through alone; the gate passes that, refuses v5's storage, survives a merge |
| Notebook's own gate (cell 4 of `colab_train_act_v6.ipynb`) | `scripts/check_colab_gate.py` | passes a three-camera dataset stored by the rule, refuses one stored as v5 and one with a changed fingerprint |
| Start guard, command clip, "cut with the first closing and still held" | `eval_truss.py` on the old checkpoint (`act_teacher_v5b64` 70k), 2 held-out scenes + 2 nothing-ripe, `--world crate23 --start-guard --clip-actions` | ran; the two held-out seeds give the targets `heldout_v5.jsonl` lists; guard fired in 2 of 2 nothing-ripe scenes (policy not run); 538 commands clipped (444 of them in the scene the policy failed); first-closing measure 1 of 2 |
| Training recipe (`--policy.dropout=0.0`, both learning rates 3e-5, no scheduler) | LeRobot's config parsed with the notebook's flags | AdamW lr 3e-5, weight decay 1e-4, scheduler None |

Not run: the recorder with the gate (`generate_truss.py`, any rig; its `--dry`, which writes no dataset, has: section
26), `remake_pictures.py part`, `check_plugin.py`'s trio block (it writes a 3 s throwaway dataset), the notebook
itself, a rollout of a trio policy.

## 24. Where the two fixed cameras go (rig "trio"), 2026-11-26 night

Owner: "3 cameras instead of 2 but also placed in a more calculated location and position", and full autonomy for
the placement. Tools: `sim/whatif/camera_score.py` (collect, rank, render) and `sim/whatif/camera_sweep_logs.py`.

**How it was scored.** On dataset v5's own cut scenes (734 of them rebuilt, world `crate23`, old start), every
candidate lens position on a grid over the cart (deck 238 mm wide centred on the arm base, plus an outrigger past
its left end; the row side of the carry zone and the crate left out), each with a small grid of aims. Per scene and
camera, from the simulator's geometry with the real lens model (72 x 52 deg, 640 x 480):
- **start**: the cut point is seen at the start frame (not hidden, at least 1 px per mm);
- **before the cut / at the cut**: the stalk and a blade are both seen at those two arm poses;
- **in the picture**: the cut point falls inside the frame at the start (else the cue box is empty);
- for a pair: at least one sees stalk and blade; the angle their lines of sight cross at; how far the cut point's 3D
  position moves when each picture is 5 px off.
A position is dropped when the lens or its post comes within 4 cm of the arm or of a plant at those poses. The
claw camera for reference: it sees the cut point at the start in 0 % of the scenes and just before the cut in 99 %.

**Then the recorded motion.** The poses above miss the way in between and the carry, so each surviving position's
post was swept against the 800 recorded episodes' state logs (arm: all 800, every 4th frame; plants and the carried
truss: 75 episodes spread over every target slot, every 3rd frame):

| Spot | Arm: nearest | Arm: episodes under 4 cm | Plants / carried truss: nearest |
|---|---|---|---|
| Today's base camera bracket | 2.2 cm | 291 of 800 | 0.6 cm (under 4 cm in 8 of 75) |
| Best score before the sweep: on the deck by the crate (`g_0.47_0.10_0.34`) | through it (-2.0 cm) | 561 of 800 | |
| Outrigger, 30 cm left of the arm, row side, lens 34 cm up (`g_0.07_0.30_0.34`) | 0.1 cm | 174 of 800 | |
| Outrigger, 35 cm left, 0-10 cm toward the row, above 10 cm (`g_0.02_0.10..0.20`) | 4.6 cm | 0 | 1.8 cm (one carry, seed 10000046) |
| **Chosen upright: 35 cm left, 10 cm toward the row (`g_0.02_0.30`), lens at 5 cm** | 10.9 cm | 0 | 9.1 cm |
| **the same upright, lens at 34-50 cm** | 4.5 cm (seed 10005040) | 0 | 7.1 cm (seed 10009015) |

⚠️ These two sweeps were coarse (the arm every 4th frame, the plants in 75 episodes). The fine ones over every
episode are in section 26: the arm's nearest stays 4.4 cm, but a carried truss came as near as 1.7 cm to the chosen
upright's line, not 7.1 cm; with the real bracket in the model nothing touches it.

After both sweeps 19 grid positions are left with the lens at or below 34 cm (21 up to 50 cm); all but five of them
are on the outrigger 35 cm to the arm's left. Nothing between the arm and the row survives: the arm folds out to
its left on the way to a truss and the carried truss swings through the middle.

**What each camera sees** (share of the 734 scenes; aim = the best of the aim grid):

| Camera | Lens (x, y, z) m | Start | Before the cut | At the cut | In the picture | px per mm (median, least) |
|---|---|---|---|---|---|---|
| Today's base camera | 0.09, 0.26, 0.10 | 79 % | 57 % | 50 % | 94 % | 1.8, 1.2 |
| Birdseye (not recorded since rig `base`) | 0.37, -0.15, 0.34 | 27 % | 7 % | 7 % | 100 % | 1.0, 0.8 |
| **low** | 0.02, 0.30, 0.05 | 85 % | 56 % | 54 % | 100 % | 1.5, 1.1 |
| high at 0.34 | 0.02, 0.30, 0.34 | 97 % | 90 % | 74 % | 99.7 % | 1.6, 1.1 |
| **high (0.42)** | 0.02, 0.30, 0.42 | 98 % | 98 % | 76 % | 100 % | 1.4, 1.0 |
| high at 0.50 | 0.02, 0.30, 0.50 | 97 % | 99 % | 79 % | 100 % | 1.3, 1.0 |

| Pair on the one upright | Both see the cut point at the start | One sees stalk + blade before the cut | at the cut | Crossing angle | 3D point moves (median, 90 %) |
|---|---|---|---|---|---|
| low + high at 0.34 | 82 % | 95 % | 91 % | 49 deg | 5, 10 mm |
| **low + high at 0.42** | 83 % | 99 % | 95 % | 61 deg | 4, 8 mm |
| low + high at 0.50 | 83 % | 100 % | 95 % | 71 deg | 4, 7 mm |

**The pick** (`scene.CART_CAMS`, rig order high, low, claw): one upright 35 cm to the left of the arm base and
10 cm toward the row; **high** with its lens 42 cm above the deck top, aimed at (0.30, 0.37, 0.22); **low** at 5 cm,
aimed at (0.34, 0.37, 0.26). High is first in the rig, so the six cue numbers are its.
- Against today's base camera: the cut point is seen at the start in 98 % instead of 79 %, stalk and blade just
  before the cut in 98-99 % instead of 57 %, and the post is out of the arm's way (today's bracket had the arm
  within 4 cm in 291 of the 800 recorded episodes).
- 0.42 or 0.34 for the high lens: the owner, shown both (`outputs/v6/cams/cameras_today_vs_new.png`), 2026-11-26:
  "stick with 42". ⚠️ 0.42 is above the truss band (cut points 0.13-0.31 m), where a real crop has leaves; the
  plank rig has none. The gain over 0.34 is small (the pair's rows above). Both are the same post, so changing it
  later is a re-render from the state logs, not a re-recording: the new episodes must be recorded with state
  logging on.
- ⚠️ Real cart: an upright on an outrigger batten reaching 21 cm past the deck's left end. On the plank rig as
  designed the end frames stand about 225 mm from the arm, so this upright is 12.5 cm outside the end frame, and
  arm B needs the same post turned half a turn at the other end (first written here as "the mirror image", which
  is wrong: section 29). Nothing is built.
- ⚠️ In the simulated house the post and the housings' backs reach 3-6 mm into the end frame's ply (x <= 0), behind
  both lenses; and high's post runs through low's housing (one post).
- Files: `outputs/v6/cams/v5_*.npz` (the scores), `rank_v5_sweep.txt` / `rank_v5_sweep_z50.txt` (lens up to 0.34 /
  0.50 m), `sweep_arm_v5.*`, `sweep_plants_v5.*`, `sheet_candidates.png` (pictures).

## 25. Start position: old (folded) against the searched one, what the cameras see and what the teacher does

World `crate30` (these were collected before the two worlds were named), old base, v5 jaw, 356 / 357 scenes
(`outputs/v6/cams/rank_v6world_old.json`, `rank_v6world_new.json`); teacher runs r22 and r36 on tuning seeds
9730000-9730044.

| | Old start (-90, -40, 90, -40, 0, 50) | Searched start (-25, -75, -15, 75, 0, 50) |
|---|---|---|
| Claw camera sees the cut point at the start | 0 % | 34 % |
| Claw camera sees it just before the cut | 98 % | 89 % |
| Base camera sees the cut point at the start | 81 % | 63 % |
| Base camera: stalk and blade before the cut / at the cut | 71 % / 68 % | 46 % / 50 % |
| Base camera: cut point inside the picture | 92 % | 84 % |
| Teacher, 45 scenes | 38 of 44 (r22) | 39 of 45 (r36, the first 45 of 90) |

- The searched start points the claw camera at the row, so it sees the target from the start in a third of the
  scenes, but the unfolded arm then stands in the base camera's view, which loses more than the claw gains.
- ⚠️ The two teacher runs are not the same scenes: the start pose is part of what decides which truss is ripe
  (section 21), and only 14 of the 45 seeds give the same target. The two rates say "no difference seen", no more.
- With rig `trio` the high camera sees the cut point at the start in 98 % of the scenes from the old start
  (section 24), which was the searched start's one gain. The 800 old episodes all begin at the old start. So
  dataset v6 keeps the old start; the default is the owner's to switch.

## 26. Checks of the camera pick (2026-11-26 night; nothing recorded, no dataset written)

**The post against every frame of the old episodes** (`camera_sweep_logs.py arm --every 1`, all 800 episodes,
384,677 frames; distances to the post's centre line and the lens point, so the post's 8 mm radius and the housing
come off):

| Post | Arm: nearest | Episodes under 4 cm | under 6 cm |
|---|---|---|---|
| Chosen upright, lens at 34-50 cm | 4.4 cm (seed 10005040, frame 50) | 0 | 5 |
| Chosen upright, lens at 5 cm | 10.9 cm | 0 | 0 |
| Today's base camera bracket | 2.2 cm | 314 | 487 |

No old episode has to be left out of a re-make because of the arm.

**The same for the plants and the carried truss** (`camera_sweep_logs.py plants --n 0 --every 3`: the 735 cut
episodes that rebuild, every 3rd frame; from a fruit's or stalk's surface to the line under the lens):

| Post | Nearest plant part or carried truss | Episodes under 4 cm | under 2 cm |
|---|---|---|---|
| Chosen upright, lens at 34-50 cm | 1.7 cm (seed 10023057, frame 207: a fruit of the truss being carried) | 10 | 2 |
| Chosen upright, lens at 5 cm | 4.7 cm | 0 | 0 |
| Today's base camera bracket | through the line (-0.1 cm) | 85 | 23 |

⚠️ The coarse sweep of section 24 (75 episodes) had said 7.1 cm for the chosen upright: the full one finds a carried
truss passing 1.7 cm from the line under the lens. The post itself stands about 1.8 cm behind the lens.

**Does anything touch the real bracket?** `camera_sweep_logs.py touch`: the 32 old episodes that came within 6 cm
of either chosen post's line (arm or plants) replayed frame by frame with the rig's real post cylinders and camera
housings in the model, and the simulator's own collision check asked at every frame: **0 of 32 have the arm or a
plant part inside a bracket.** The check sees both kinds: with a post put where the arm is known to pass (the
deck spot of section 24, seed 10017026) it reports the arm inside it in 79 frames, 29 mm deep; with a post put
where the carried truss of seed 10023057 passed, it reports that truss's fruit inside it from frame 207. So no old
episode has to be left out because of the new upright. (`outputs/v6/cams/fine_arm_*`, `fine_plants_*`,
`touch_trio.*`.)
- ⚠️ The other way round (`touch --rig base --cams base`: the old bracket the episodes were recorded with, which
  the re-made pictures no longer show): the arm never touched it (0 of the 15 episodes that came within 2.5 cm),
  a fruit did in **3 of the 85** episodes that came within 4 cm: seed 10000046 (the carried truss, 3 frames,
  0.9 mm deep), 10021000 (the target truss, 6 frames, 0.7 mm) and 10020046 (a neighbour's fruit resting against
  the housing through 446 frames, 0.1 mm). In those 3 re-made episodes a fruit is held off by something the
  pictures do not show; they stay in (grazes of under 1 mm). (`outputs/v6/cams/touch_base_*`.)
- ⚠️ Queue trap met here: `sim_queue.sh` was edited while a queue ran it and bash read on from the changed file
  (`unexpected EOF`, exit 2). The jobs are their own units and finished; only the queue script stopped.

**The start frame of all 800 old episodes through the three-camera path** (`remake_pictures.py dry --start-only
--shard i/4`): 799 rebuild as recorded (the model's size, the target, the first frame's joint readings). In all 735
cut episodes among them both fixed cameras have the target's box and its cut point inside the picture (high 735 of
735, low 735 of 735) and none has an empty cue; the 64 nothing-ripe episodes have the empty cue they should. So
the rig's order (high first, its box gives the cue numbers) stands.
- ⚠️ **One old episode cannot be re-made: v5 episode 353 (seed 10014003, part 14).** MuJoCo 4.0.1 builds a scene of
  4 trusses from its seed (399 position values), the log holds one of 5 (498). MuJoCo 3.8.1 (env `lerobot`, which
  recorded v5) builds the recorded one from the same code, so the simulator version flips which draw of this
  seed's layout has a truss in reach: 1 seed in 800. `remake_pictures.py` leaves it out by name (`LEFT_OUT`) and
  stops on any other mismatch, so the re-made set is 799 episodes (735 with a cut, 64 without).
- ⚠️ The planner's count of the old episodes per cell (section 27) comes from the camera scorer's 734 scenes: it
  lacks two episodes the scorer found no cut pose for (seeds 10007069, 10029014; both re-make) and has 10014003 as
  the scene 4.0.1 builds. At most one episode off in three cells.

**The first three-camera pictures** (`remake_pictures.py dry --episodes 0,23,1,13,799,15,147,233 --shots`, rig trio,
scenes decided with the old bracket; the episodes are a nothing-ripe one, a far stem, near ones, a high truss, the
last one, and the three closest calls of the sweeps): 8 of 8 rebuilt as recorded (the logged joint readings equal
the dataset's on every frame), every 5th frame built as a dataset frame and passed the dataset's own frame check,
the target's box and cut point inside the picture on both fixed cameras in all 7 cut episodes, an all-zero cue only
in the nothing-ripe one. 4,063 frames in 186 s with three other jobs running = 0.046 s per frame, so the 383,877
frames of the old 800 are about 5 hours of rendering in one process (the video writing is not in that figure).
- Pictures: `outputs/v6/remake_dry/shots/remake_trio_e*.png` (columns high, low, claw; rows start, 15 frames
  before the cut, the cut, mid-carry).
- Neither bracket shows in the other fixed camera's picture. The claw camera sees the post when the arm swings past
  it (episode 147, frames 40-60), as it would on the real cart.
- ⚠️ For a truss on the nearest stem the high camera has the cut point well inside the picture but the fruit below
  it run out of the bottom edge, so the box is cut off there; and the low camera, looking along the row, can have
  a nearer truss in front of a far target, so its box then sits on that nearer truss (it marks where the target
  is, hidden or not, as on the base camera in dataset v5).
- ⚠️ The first run of this failed on the check itself: `validate_frame` against the raw feature dict refuses the cue
  feature's list shape `[6]`; a dataset turns shapes into tuples when it is created, which is what the dry mode now
  checks against.

**The recorder's seed-plan path** (`generate_truss.py --dry` on a 7-episode toy plan made from a 33-scene sample
census, `--rig trio --layout-rig base --world crate23 --start-jitter-deg 3 --seeds`; plans and judges, writes no
dataset): 7 of 7 kept after 9 scenes built (the teacher refused 2), one per cell as the quota says, the seeds of
full cells skipped without a scene, and the recorder's own cell equal to the census's in all 9 (the census is made
with rig `base` and no start change, the recording with rig trio and the start change). 4.3 min.
- ⚠️ One of the two refusals was "start pose not clear": with the 3 deg start change. r39 below counts how often.
- Run again on 2026-11-27 with the clear-aware start change (below) and the same plan: again 7 of 7 kept from 9
  scenes, the recorder's cell equal to the plan's in 8 of 8, 4.6 min. The refusal that names the start pose (seed
  10100005, strict mode only) is not the start change's: a neighbour's fruit hangs 0.9 mm from the wrist at the
  exact start pose in that scene, no draw in 20 clears it, so the scene starts at the exact pose, and the teacher's
  strict mode refuses that with or without a start change (dataset v5's recording, which had no start change,
  has the same line in 18 of its 1,736 scene records).
- The recorder's log now carries what each scene drew (`start_drawn_deg`, `start_draws`; all zeros = exact start).

**The teacher with the new post standing in the scene** (same 90 tuning scenes 9730000-9730089, world `crate23`,
old start, no start change):

| Run | Brackets in the scene | Teacher |
|---|---|---|
| r37 | today's base camera bracket | 65 of 89 scenes with a ripe truss = 73 % (16 no plan, 4 moved a neighbour, 2 fruit came off, 1 never cut, 1 not all in the crate) |
| r38 | the new upright with both cameras (scene decided with the old bracket) | 65 of 89 = 73 % |

- The same truss is the target in 90 of 90 scenes and every scene ends the same way (65 work in both, 24 fail in
  both). The attempt records are the same in 89 of 90; in the last one (9730013) a fruit an early failed attempt
  knocked off a neighbour rolled 392 mm instead of 249 mm. One no-plan scene (9730061) counts 81 blocked crate
  routes instead of 88. So the post costs the teacher nothing on these scenes.
- `sim/scripts/grid_ab.py "r37=outputs/v6/base_grid_r37_CRATE23_BASE_[0-9]*.jsonl"
  "r38=outputs/v6/base_grid_r38_TRIO_[0-9]*.jsonl"` pairs the runs by seed: won 0, lost 0, the same failure list.

**The start change (`--start-jitter-deg 3`): a plain draw does not work, a clear-aware one does.**
- r39 (new post + each arm joint drawn within 3 deg of the start pose, first 15 tuning scenes): 6 of 15 work, where
  11 of the same 15 work without the change. All 5 lost scenes are refused before any motion: "start pose not
  clear". Cause (seeds 9730004/5/7 rebuilt): the folded start has the lower arm 2-3 mm off the shoulder, and a
  draw that folds the elbow about 1.5 deg further brings them under the teacher's 2 mm self-clearance
  (`self: shoulder / lower_arm`). Nothing else is near: the table is 16 mm from the arm at the start.
- Fix (`so101_mujoco.py` `_start_clear`, 2026-11-27): a drawn start pose is kept only when the arm stays 3 mm from
  itself and 5 mm from everything else that collides in the physics (1 mm over what the teacher asks), else it is
  drawn again from the same stream, up to 20 times, then the exact pose. The first draw is the old one, so a seed
  whose first draw was clear starts where it did. 9 scenes rebuilt with it (the three above and six more): 1-4
  draws each, and the teacher's own start check has no hit in any.
- ⚠️ So the elbow's change is one-sided: about -3 to +1 deg. The other joints keep the full range.
- r40 (the fixed change) against r38 on the same 45 tuning scenes (9730000-9730044, world `crate23`, new post):

| Run | Start of each scene | Teacher |
|---|---|---|
| r38 | the exact start pose | 35 of 44 scenes with a ripe truss = 80 % (5 no plan, 2 moved a neighbour, 1 never cut, 1 fruit came off) |
| r39 (first 15 scenes, stopped) | each joint drawn within 3 deg, plain | 6 of 15, where r38 has 11: 5 refused at the start pose |
| r40 | each joint drawn within 3 deg until the arm is clear | 35 of 44 = 80 %, the same 9 failures |

- Won 0, lost 0, no refusal names the start pose, and every one of the 45 scenes ends the same way in both runs.
  The rule set before the run (no start-pose refusal, at most about 2 of 45 lost) is met, so the documented
  recording command keeps `--start-jitter-deg 3`.
- One scene (9730028) gets another target with the start change (s0t0 instead of s1t1): the target is the ripe
  truss with the most of it in view at the start frame, and the moved arm hides a little more or less of each.
  It works in both runs.
- Draws needed, last 15 scenes (logged since a logging-only commit made between the second and the third chunk
  of r40; the start change's code is the same in all three): one draw in 11 scenes, 2, 5, 7 and 8 draws in one
  each, none back at the exact pose. r40's 45 scenes took 72 minutes of job time against r38's 66.

## 27. The 600 new episodes: what the teacher keeps, time, disk, and the order after the go (2026-11-27 night)

Nothing here was recorded: the numbers come from dataset v5's own recording logs
(`~/.cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl`, 32 parts) and from build-only scenes.

**What the teacher keeps, by where the truss hangs.** v5's recording built 1,736 scenes: 625 were turned away
by its row-part balance before the teacher ran, 64 were nothing-ripe episodes (all 64 kept), 3 had no target, and
of the 1,044 cut scenes the teacher tried it kept 736 = 70 %. Per cell (4 parts along the row x 3 heights,
`generate_truss.cell_of`; a kept scene's cell from its cut point, a refused scene's from its logged row position
and cue pixel against the kept ones, which gets 96 % of the kept ones right; 734 of the 736 kept scenes are in
the table, the camera scorer's files lack two):

| Height \ row part | x0 (0.20-0.27 m) | x1 (0.27-0.34) | x2 (0.34-0.41) | x3 (0.41-0.48) |
|---|---|---|---|---|
| z0, below 0.20 m | 83 % (87 kept, 18 refused) | 75 % (56, 19) | 66 % (67, 35) | 61 % (45, 29) |
| z1, 0.20-0.26 m | 80 % (61, 15) | 83 % (48, 10) | 65 % (51, 28) | 68 % (68, 32) |
| z2, above 0.26 m | 68 % (82, 38) | 70 % (70, 30) | 60 % (35, 23) | 67 % (64, 31) |

- So no cell is below 60 %. The seed planner's floor is 1.8 census seeds per planned episode (56 % kept;
  `MIN_SUPPLY`, was 1.6 = 63 %, too tight for x2z2 and x3z0), and it lists more than that where the census has them.
- ⚠️ A part's seeds are its own (the parts record side by side), so a cell with 1-3 episodes in a part can end one
  short by bad luck even at 3 seeds per episode. `v6_seed_plan.py` now prints how many episodes its lists should
  give at these keep rates, and `--topup` plans the missing ones as one more part from seeds not yet listed (it
  writes `plan_topup_pNN.json`, never over the plan; tried on the toy plan: 7 kept of 14 planned -> 7 more planned).

**The census.** Where a fresh scene's target hangs is known only once the scene is built, so 4,500 fresh seeds
(10100000-10104499, used by nothing else) were built without running the teacher, as 15 queue jobs of 300
(`sim/scripts/sim_queue.sh`, seven side by side, about 8 s a scene):

    python -u base_grid.py --base 0.37,0.20 --jaw v5 --old-start --world crate23 --rig base --build-only \
        --seed-start <S> --scenes 300 --out ../outputs/v6/census_v6/cen_<S>.jsonl

Every seed is there once (counted by the records, not by the jobs' done markers). 4,487 scenes have a ripe truss,
13 have none. Where the target's cut point hangs when nothing steers it:

| Height \ row part | x0 (0.20-0.27 m) | x1 (0.27-0.34) | x2 (0.34-0.41) | x3 (0.41-0.48) |
|---|---|---|---|---|
| z0, below 0.20 m | 718 = 16.0 % | 296 = 6.6 % | 265 = 5.9 % | 227 = 5.1 % |
| z1, 0.20-0.26 m | 523 = 11.7 % | 236 = 5.3 % | 237 = 5.3 % | 367 = 8.2 % |
| z2, above 0.26 m | 775 = 17.3 % | 360 = 8.0 % | 221 = 4.9 % | 262 = 5.8 % |

- 45 % of the fresh scenes have their target in row part x0 and 16 to 20 % in each of the other three; the three
  heights are even (34 / 30 / 36 %). Likely reason, not measured: the target is the ripe truss that shows most at
  the start picture of the old base camera, and x0 is the row part nearest that camera.
- v5's balance step evened the row parts only partly (31 / 24 / 21 / 24 % of its 734 cut episodes) and did
  nothing about height within a row part: its cells hold from 35 (x2z2) to 87 (x0z0) episodes.

**The plan** (`sim/seed_plan_v6`, in the repo together with the census records it was made from; from `sim/`:
`python scripts/v6_seed_plan.py --census "../outputs/v6/census_v6/cen_*.jsonl" --old "../outputs/v6/cams/v5_*.npz"
--out seed_plan_v6`). 600 new episodes = 540 cut + 60 nothing-ripe (v5's share), in 24 parts of 25. The cut
episodes go to the cells with the fewest old ones, so that old and new together are level:

| Cell | Old 800 | New | Together | Fresh scenes in the census | Listed | Left for a top-up | Expected short |
|---|---|---|---|---|---|---|---|
| x0z0 | 87 | 20 | 107 | 718 | 60 | 634 | 0.0 |
| x1z0 | 56 | 50 | 106 | 296 | 150 | 146 | 0.0 |
| x2z0 | 67 | 39 | 106 | 265 | 117 | 148 | 0.4 |
| x3z0 | 45 | 61 | 106 | 227 | 183 | 44 | 0.8 |
| x0z1 | 61 | 46 | 107 | 523 | 138 | 385 | 0.1 |
| x1z1 | 48 | 58 | 106 | 236 | 174 | 62 | 0.0 |
| x2z1 | 51 | 55 | 106 | 237 | 165 | 72 | 0.5 |
| x3z1 | 68 | 38 | 106 | 367 | 114 | 253 | 0.4 |
| x0z2 | 82 | 24 | 106 | 775 | 72 | 634 | 0.4 |
| x1z2 | 70 | 36 | 106 | 360 | 108 | 252 | 0.3 |
| x2z2 | 35 | 71 | 106 | 221 | 213 | 8 | 0.8 |
| x3z2 | 64 | 42 | 106 | 262 | 126 | 136 | 0.4 |
| all | 734 | 540 | 1,274 | 4,487 | 1,620 | | 4.1 |

- Three scenes are listed per wanted episode (`OVERSUPPLY`), a part's list being tried in order until each of its
  cells is full. At the keep rates above the lists should give about 536 of the 540 cut episodes. Why 3: at 2.5
  they give 529 and at 4 they give 538, but 4 lists every fresh scene of x3z0 and x2z2, so a part that ends short
  there could not be topped up; at 3 every cell keeps more scenes than twice its expected shortfall needs (the
  tightest is x2z2 with 8 left for 0.8 expected short).
- The 60 nothing-ripe episodes: each part lists two scenes more than it needs (106 in all): the census's 13
  scenes without a ripe truss first, then 93 from the two fullest cells (x0z2 69, x0z0 24), which the recorder
  builds with nothing ripe (`hold`), as v5 did (it kept 64 of 64 of them). The parts hold 0 to 5 of them (part 15
  none, part 0 four).
- A part's list has 67 to 75 scenes for its 25 episodes; no scene is in two lists (1,726 listed in all).

**Is the census's scene the scene the recorder builds?** The census (above) builds each fresh seed with rig
`base` and no start change; the recorder uses rig `trio` on the old layout and the start change. The first 300
recording seeds built the recorder's way (`base_grid.py --build-only --rig trio --layout-rig base
--start-jitter-deg 3`, `outputs/v6/census_v6_jit`): the same trusses at the same places in 300 of 300, the same
target and the same cell in 299. In the one that differs (10100228) two trusses are ripe and the moved arm
changes which of them shows more at the start frame; the recorder counts an episode in the cell it really has, so
this costs nothing but one seed's place in the plan.
- Draws the start change needs: the first is clear in 56 % of the scenes, three in 92 %, eight in 99 %; 2 of 300
  (10100005, 10100237) are never clear in 20 draws and start at the exact pose. The elbow's change runs from -3.0
  to +0.8 deg (mean -1.0), the other four joints use the full 3 deg both ways.

**The rehearsal on the real lists** (2026-11-27, 02:21-02:47). The recorder's own command with `--dry` (the
teacher plans and plays every scene, no dataset is written) on the first two lists, as two queue jobs side by
side with a third teacher job: `python generate_truss.py --teacher --no-dr --no-yolo --rig trio --layout-rig base
--world crate23 --start-jitter-deg 3 --noise-share 0.5 --retries 6 --seeds seed_plan_v6/p00.jsonl --seed-start
10100000 --episodes 25 --dry --repo-id krish5831/v6_dry` (and `p01.jsonl`; logs `outputs/v6/rehearsal_p0*.log`,
`~/.cache/blush_sim/truss/v6_dry*_dry_20261127_0221*.jsonl`):

| Part | Kept | Cells at their quota | Scenes built | Refused | Rows of the list reached | Time | Peak memory |
|---|---|---|---|---|---|---|---|
| p00 | 25 of 25 (4 nothing-ripe) | all 10 | 31 | 6: 4 no plan, 2 with every dry run failed | 49 of 69 | 25 min | 1.9 GB |
| p01 | 25 of 25 (3 nothing-ripe) | all 11 | 32 | 7: 5 no plan, 2 with the noisy run failed | 47 of 71 | 20 min | 2.1 GB |

- The teacher kept 43 of the 56 cut scenes it tried = 77 % (v5's recording: 70 %), so both parts filled every
  cell with a third of their lists unused. 18 and 15 rows were passed over without a scene being built (their
  cell was full by then), which is the step that replaces v5's balance.
- A kept episode's cell is the list's in 42 of 43; the one that differs (10100032: listed as x3z2, is x2z2) was
  counted where it really hangs.
- Start change: the first draw was clear in 39 of the 63 scenes, the others took up to 12; one scene started at
  the exact pose (10100005 again, refused: "start pose not clear", as without the change).
- About half of the kept cut episodes are noisy ones (10 of 21 and 10 of 22; `--noise-share 0.5`). ⚠️ The log
  record of a nothing-ripe episode can carry the `noise` key too (it is drawn per seed before the scene is built);
  such an episode is never played with noise, so count noisy episodes among the cut ones only.
- ⚠️ What this does not show: a dry part plans and plays only (0.8-1.0 minutes per kept episode here). The
  recording pass (the kept plan played once more with the three cameras rendered, the pictures stored and the
  gate's fingerprints taken) comes on top and has still never run; the trial part is its first run. Peak memory
  with a dataset open was 4.4 GB a part for v5.

**The policy-test scenes with the new post** (2026-11-27, 02:25-03:25; 8 queue jobs of 25 scenes,
`outputs/v6/heldout_trio/`): `python -u base_grid.py --base 0.37,0.20 --jaw v5 --old-start --world crate23 --rig
trio --layout-rig base --retries 6 --seeds <25 seeds of the list>`, no start change, as in a policy test.
- All 200 scenes of `heldout_v5.jsonl` and `heldout_v5b.jsonl` rebuild with the truss their list names (200 of
  200), and the teacher completes all 200 (its first plan in 83 and 79 of the 100, the others on a retry), the
  25 scenes a checkpoint test uses included.
- ⚠️ No teacher rate: the lists hold scenes a teacher dry run with `--retries 6` solved when they were made (on
  MuJoCo 3.8.1, old bracket). What it says: the test sets stand as listed on MuJoCo 4.0.1 with the new post in the
  scene, so a policy trained on dataset v6 can be tested on the same 100 scenes as the v5 policies.

**Time.** v5's 800 episodes took 1,548 worker-minutes by the parts' own logs = 1.93 minutes per kept episode
(parts of 25: median 45 minutes, 17 to 81; `record_parts.sh`'s header says 7.4 h by the clock). 600 episodes at
that rate are about 19 worker-hours, so 6 to 7 hours on three workers; the third camera adds rendering and
writing, the missing balance step saves scenes (v5 built 625 scenes only to turn them away; the seed plan skips a
full cell's seeds without building them). The re-make of the
old 799: 1.4 to 4.9 hours of rendering in one process (0.013 s a frame alone, 0.046 s with three other jobs
running: section 26), before the video writing, which has not been timed. One part of each, run first, gives the
real figures.

**Disk.** The picture rule (H.264, every frame a key frame, qp 23: `picture_store.py`) costs about 18 kB a frame
on start pictures, more with the arm in view. About 1,400 episodes x 3 cameras x 480 frames = 2.0 M frames, so
the merged dataset is about 36 GB (v5: 6.6 GB for two cameras), 29 GB at qp 26. The parts and the merged copy
exist side by side until the parts are deleted: 70 to 95 GB at the peak. Free on the laptop tonight: 136 GB.
⚠️ The qp must be chosen before the first part is written (one dataset, one setting).
⚠️ This estimate was low: the trial (section 28) measured 25.8 kB a picture on real frames, so about 52 GB at
qp 23 and 104 GB at the peak.

**Order after the go** (the commands are in `sim/README.md`, "Dataset v6"). Neither `remake_pictures.py part`, the
real recorder with three cameras, nor a merge of the two kinds has ever run, so:
1. one re-made part on one worker (`remake_parts.sh ... 0 1 1`): look at its peak memory, seconds per episode with
   the video writing, its size on disk, and at a few of its frames;
2. one recorded part (`record_parts.sh ... 0 1 1` with the seed plan): the same, and kept against quota;
3. merge the two into a trial dataset (`merge_parts.py`, which runs the picture gate on the result);
4. only then the other parts on three workers each, the top-up if parts ended short, the real merge, the gate, and
   the upload (the owner's step: nothing goes to the Hub by itself).

⚠️ Owed before the owner's five-video review of the new dataset: `scripts/review_videos.py` is built for v5 (two
cameras side by side, merged index = 25 x part + local index). Dataset v6 has three cameras, the re-made part 14
has 24 episodes and a new part can end short, so its episode numbers must come from the merged dataset, not from
25 x part. Not changed tonight. Done 2026-11-27: `scripts/v6_check.py videos` cuts the five videos with the three
cameras side by side and takes the episode numbers from the parts' logs and sizes (section 28).

**Decisions that are the owner's** (asked in the morning summary of 2026-11-27; nothing below is switched by me).
Owner, 2026-11-27 morning: "yea go for the trail and the data set name whatever you gave is good and go with the
36gb version". So of the list below: 1 is given for the trial only (the rest of the run waits for the trial's
report), 2 is `krish5831/blush_sim_teacher_v6`, 3 is qp 23 (which measures 52 GB, not 36: section 28). 4 to 10 are
open; 5 is in the trial's recorded part because it is in the documented command.
1. The go: first the trial of one part of each (steps 1-3), then the rest. The re-make and the recording run one
   after the other, not side by side (three workers each fill the memory rule).
2. The dataset's name. "The same dataset as before" cannot be the old repo itself: the pictures of the old 800
   are new files under other camera names (`high`, `low`, `claw` for `base`, `claw`), so pushing there would
   replace v5. Proposed: a new private repo (`krish5831/blush_sim_teacher_v6`); v5 stays as it is.
3. The stored pictures' quality before the first part is written: qp 23 (about 36 GB) or qp 26 (about 29 GB).
4. The start position: dataset v6 keeps the old folded start (section 25); the code default is the owner's.
5. The start change of up to 3 deg per joint for the new episodes (section 26): in the documented command, off by
   default in the code. Policy tests keep starting at the exact pose.
6. The cue's form on the pictures: the box is drawn over whatever hides the truss and is cut off at the picture's
   edge (section 26, low camera and near trusses), as in v5.
7. Whether the two guards (`--start-guard --clip-actions`) become the standard for policy tests.
8. A second copy of `~/.cache/blush_sim/states` (3.0 GB, the only source of the old episodes' pictures in any new
   form); an upload anywhere needs the owner's word.
9. `CLAUDE.md` says "video codec libsvtav1": true for the real rig's recordings, while the simulator's datasets now
   store H.264 by the picture rule. One line there would say so.
10. The real cart: the camera upright stands 12.5 cm outside the plank rig's end frame, and arm B needs the same
    post turned half a turn, not a mirror image (section 24). Nothing is built or drawn.

Owner, 2026-11-27 midday, shown 7 to 10 as "four smaller things": "finish the four smaller things also". What
was done for each: section 29. 4 to 6 stay as they are in the run (old start, 3 deg start change, the cue's form).

## 28. The trial (one re-made part, one recorded part, one merge), then the full run (2026-11-27, after the owner's go)

The go was for the trial only (section 27, decisions). Names: parts `krish5831/blush_sim_teacher_v6_old_p00` and
`..._new_p00`, merge `krish5831/blush_sim_teacher_v6_trial`, all local (`~/.cache/huggingface/lerobot`); nothing
went to the Hub. Commands as in `sim/README.md`, one after the other, one worker each:

| | Re-made part (v5 episodes 0-24) | Recorded part (seed plan `p00`) | Merge of the two |
|---|---|---|---|
| Ran | 08:26-08:38 | 08:39-09:13 | 09:14 |
| Episodes | 25 of 25 (2 nothing-ripe) | 25 of 25 (4 nothing-ripe) | 50 |
| Frames | 12,254 | 11,413 | 23,667 |
| By the clock | 11 min 17 s | 34 min 31 s | 9 s |
| Processor time | 22 min 4 s | 47 min 6 s | 15 s |
| Peak memory (the job's cgroup, from the journal) | 3.3 GB | 3.6 GB | 2.6 GB |
| Video on disk | 953 MB = 38.1 MB an episode | 881 MB = 35.3 MB an episode | 1.83 GB |
| Picture gate (stored frames read back) | 150 of 150 the same | 150 of 150 | 300 of 300 |

**Checked with `scripts/v6_check.py`** (new; "the worker ended with exit 0" says nothing, both worker scripts
exit 0 whatever the part did, and a short part still finalizes):
- Both parts and the merge: cameras `high`, `low`, `claw`; stored as H.264, every frame a key frame, qp 23, no crf
  (`picture_store.encoder_of(features) == encoder()`, what `eval_truss.py` will ask of the training dataset); the
  gate file is there and every fingerprinted frame reads back the same (2 frames x 3 cameras an episode).
- Re-made part: the joint readings and commands of all 12,254 frames equal v5's bit for bit; no cut episode has an
  all-zero box in its cue numbers.
- Recorded part: every zone at the plan's quota; the cue numbers in the dataset equal the log's. The recorder
  built the same 31 scenes in the same order as the rehearsal of the night (section 27) and kept the same 25. It
  refused 6: 4 with no plan (one is 10100005, whose start pose is never clear) and 2 whose seven dry runs all
  failed. ⚠️ The two refusals only a recording pass can make ("recording differs from the dry run", "recording
  picked another truss") did not occur: 0 of 25. Noisy: 10 of the 21 cut episodes. Start pose: first draw clear in
  25 of 31 scenes, second in 5, one at the exact pose.
- Merge: episodes and frames are the parts' sums, and every part's joint readings, commands and cue numbers stand
  at their place in the merge bit for bit.
- Pictures read back through `LeRobotDataset` (`v6_check.py sheet`, `outputs/v6/trial/`: a re-made cut, a
  recorded cut, a recorded noisy cut, a nothing-ripe scene of each kind; start frame and mid-episode, three
  cameras): the cue box and cross are on `high` and `low`, none on a nothing-ripe scene, the claw camera shows the
  stalk between the jaws mid-episode. Five review videos (`v6_check.py videos`,
  `outputs/videos/review_blush_sim_teacher_v6_trial/`, one look only); two videos and one sheet went to the owner.
  ⚠️ The two videos sent had one frame too many: the next episode's start frame as their last frame (cut by
  rounded timestamps). A slip of the cutting, not of the dataset; `videos` now stops by frame count, and the five
  were cut again.

**Size: the estimate of section 27 was low by a factor 1.43.** It came from six start pictures (18.1 kB a
picture at qp 23). Real frames, with the row in view of the two post cameras and the arm moving, take 25.8 kB:

| MB for a part's frames | high | low | claw | per picture, mean |
|---|---|---|---|---|
| Re-made part (12,254 frames) | 422 | 359 | 172 | 25.9 kB |
| Recorded part (11,413 frames) | 393 | 336 | 152 | 25.7 kB |

At these sizes the whole dataset is 30.4 GB (799 old) + 21.2 GB (600 new) = **51.6 GB of video at qp 23**, not
36. Other settings, measured by decoding the re-made part's own videos and encoding them again (the same method
at qp 23 gives 0.954 of the stored size, so the ratios are taken against that), and the loss on the six raw start
pictures (mean difference to the raw picture in grey levels of 255; about 1.6 of it is the colour format, which
every setting shares; `picture_store.py selfcheck`'s method):

| qp | Size against qp 23 | Whole dataset | Loss |
|---|---|---|---|
| 18 | not measured on real frames | | 1.61 |
| 23 (the owner's choice) | 1 | 52 GB | 1.70 |
| 26 | 0.83 | 43 GB | 1.78 |
| 28 | 0.68 | 35 GB | 1.87 |
| 30 | 0.58 | 30 GB | 1.96 |
| v5's storage (AV1, 2 cameras) | | 6.6 GB | 1.94 |

Every constant qp keeps the picture rule (a stored frame = its picture sent through alone). Peak on disk at
qp 23: the parts and the merged copy side by side, 104 GB, against 134 GB free. A free Hugging Face account has
100 GB of private storage (huggingface.co/docs/hub/storage-limits, read 2026-11-27).

**Time for the rest, from these two parts.** Re-make: 31 parts x 11.3 min = 5.8 hours of one worker, about 2 hours
on three. Recording: 23 parts x 34.5 min = 13.2 hours of one worker, about 4.5 hours on three (a part's time
varies with its scenes: the rehearsal's dry passes took 25 and 20 minutes), then the top-up part and the merge.
Together about 7 to 8 hours, against the 10 to 12 guessed in section 27. Memory: the peaks above are one part of
each kind. v5's recorded parts peaked at about 4.4 GB (`record_parts.sh`), and a part's peak depends on its
scenes. Three workers at 3.6 GB leave 10 GB of the 21 GB available free, three at 4.4 GB leave just under 8 (the
rule asks 8). So: three workers, one kind after the other as planned, and a look at `free -g` once the first
three parts are well under way. ⚠️ Not measured: three workers side by side (the trial ran one at a time; v5's
recording ran three without slowing down), the top-up, the 52 GB merge.

**What is now in the parts.** If nothing changes, the two `p00` parts are part 0 of the real run: qp 23 and the
start change of up to 3 deg (`--start-jitter-deg 3`) are then in the dataset. Changing either means deleting both
part folders and redoing them (46 minutes). The `_trial` merge is a throwaway and does not get in the way of the
real merge (`krish5831/blush_sim_teacher_v6`).

**The full run: done 2026-11-27, 09:52-21:51** (owner, after the trial's report: "go" and "lets do 52gb" = the
rest at qp 23, the start change kept, the trial's two parts as part 0). The dataset is
`krish5831/blush_sim_teacher_v6`, on this laptop only (`~/.cache/huggingface/lerobot/krish5831/blush_sim_teacher_v6`):
nothing went to the Hub.

| | Re-made (v5's episodes, new cameras) | Recorded (new) | Merged |
|---|---|---|---|
| Episodes | 799 (64 nothing-ripe) | 600 (60 nothing-ripe) | 1,399 |
| Frames | 383,289 | 282,025 | 665,314 |
| Video on disk | 29.61 GB | 21.75 GB | 51.36 GB |
| Parts | 32 | 25 (24 and the top-up) | 57 |
| Ran | 09:52-12:15 | 12:04-21:23, top-up 21:27-21:45 | 21:47-21:50 |
| One part by the clock | 11-15 min | 23-111 min, median 44 | 2 min 54 s |
| Peak memory of one part (journal) | 2.8-3.4 GB | 3.3-3.9 GB | 7 GB = the cap |

It ran as the queue unit `blush-queue-v6run` (`MAXJ=3 sim/scripts/sim_queue.sh v6run outputs/v6/jobs_v6run.txt`), three
workers at a time: the three re-make workers, then, as each ended, a recording worker. At the five looks during the
recording, 15 to 17 GB of memory were available with three workers running (the rule asks 8). The journal gives the
merge a peak of 7 GB, which is its cap; it ended with exit 0. Probably the files it copies count as its memory until
the cap makes the kernel drop them (a guess, not measured; the trial's merge of two parts peaked at 2.6 GB).

**Checked** (`scripts/v6_check.py parts` on all 57 parts, `merged` on the merge; `outputs/v6/check_parts_final.log`,
`check_merged.log`; every line "ok"):
- Every part and the merge: cameras `high`, `low`, `claw`; H.264, every frame a key frame, qp 23; the gate file is
  there and every fingerprinted frame reads back the same (8,394 of 8,394 on the merge = 1,399 episodes x 2 frames
  x 3 cameras).
- Re-made: the joint readings and commands of all 383,289 frames equal v5's bit for bit; no cut episode has an
  all-zero box in its cue numbers. Part 14 has 24 episodes (v5 episode 353 does not rebuild), the other 31 have 25.
- Recorded: 849 scenes built, 600 kept (71 %). Refused 249: no plan 144, every dry run failed 62, the noisy run of
  the chosen plan failed 35, the scene's cell was already full 8. The two refusals only a recording pass can make
  ("recording differs from the dry run", "recording picked another truss"): 0 in all 25 parts. Noisy: 244 of the 540
  cut episodes. Start pose: the first draw was clear in 502 of the 849 scenes; 10 scenes fell back to the exact
  pose. The cue numbers in the dataset equal the log's in every part.
- Five parts ended one episode short (06, 11, 13, 14, 16: a cell's listed scenes ran out before its quota).
  `v6_seed_plan.py --topup` planned exactly those five (cells x1z1, x2z1, x1z2, x2z2, x3z2; 30 scenes not listed
  before: `sim/seed_plan_v6/p24.jsonl`), and part 24 kept 5 of the 9 scenes it built, in 18 minutes. With it, every
  cell holds exactly the number of new episodes the plan asked for, and 60 are nothing-ripe. (The plan had
  expected 536 of the 540 cut episodes from its lists; they gave 535.)
- Where the cut point hangs, old + new together (4 parts along the row x 3 heights): 106 or 107 episodes in each
  of the 12 cells, against 35 to 87 in v5 alone. That is by the plan's count, which has 734 of v5's cut points; the
  re-made set has 735 cut episodes (799 less 64 nothing-ripe), so one cell holds one more than counted (which one
  was not looked up). The new 540 cut episodes: 20, 50, 39, 61 (low), 46, 58, 55, 38 (middle), 24, 36, 71, 42
  (high), each row from the arm's left to its right.
- Merge: 1,399 episodes and 665,314 frames, the parts' sums; every part's joint readings, commands and cue numbers
  stand at their place in the merge bit for bit.
- Five review videos (`v6_check.py videos`, `outputs/videos/review_blush_sim_teacher_v6/`, one look only: a re-made
  cut, a short and the longest new clean cut, a new noisy cut, a new nothing-ripe scene). Four frames of each
  looked at: the box and the cross are on `high` and `low`, the claw camera shows the stalk between the jaws, the
  cut truss ends in the crate, the nothing-ripe scene has no box and the arm stays home. In the noisy clip (seed
  10102775) three red fruit stay in the box after the cut: they are the neighbour s3t0, an "almost ready" truss
  (3 ripe, 2 turning, 1 green) that the harvest rule leaves; the target s2t1 (6 ripe) is the one in the crate.

**Time: the estimate was too low.** 7 to 8 hours were estimated; it took 12 (09:52-21:51).
- The re-make went as estimated (2 h 23 min). The recording took 20.0 hours of worker time against 13.2
  estimated from the trial's one part: a part took 23 to 111 minutes, with how many of its scenes the teacher
  refuses and how long the refused ones' dry runs last (part 19 built 33 scenes in 23 minutes, part 20 built 42 in
  111).
- ⚠️ The three workers each had a fixed list (every third part). One was done at 16:45, the second at 18:14, the
  third at 21:23: the last four and a half hours ran on two workers, then one. A helper could not take over a
  waiting part safely, because a worker deletes a part folder that has no parquet footer yet. A next recording
  should let a free worker take the next part nobody has started (not changed in `record_parts.sh` now).

**Left as it is, the owner's call:** nothing is uploaded and nothing trained. On disk: the merged dataset (51.4 GB),
its 57 parts (51.4 GB again), the trial merge (1.8 GB) and the two review-video folders; 34 GB are free. The
second copy of the simulator states was brought up to date after the run (3,169 files, 3.6 GB, every file's
SHA-256 the original's). Deleting the parts, the trial merge or the review videos needs the owner's word, as does
the upload (`sim/colab_train_act_v6.ipynb` reads the dataset from the Hub).


## 29. The four smaller decisions, done (owner, 2026-11-27 midday: "finish the four smaller things also")

Decisions 7 to 10 of section 27. Done while the full run was going, so nothing the run's workers load was touched
(`scene.py`, `cameras.py`, `lens.py`, `picture_store.py`, the `truss_*` modules, the worker scripts), and no fourth
simulator job was started.

**7. The two guards are the standard for policy tests.** `colab_tests.sh`, `test_all_checkpoints.sh` and
`train_and_test.sh` pass `--start-guard --clip-actions` unless `GUARDS=""` is set. `eval_truss.py`'s own defaults
stay off, so an old checkpoint re-run by hand still compares with the earlier tables. `checkpoint_table.py` has a
"Guards" column, and a step tested with and without them gets a row for each.
- Checked: the three scripts against a stand-in launcher that only prints (the flags appear, and vanish with
  `GUARDS=""`); the table script on `act_teacher_v5b64`'s existing logs (53 lines, the same apart from the new
  column, which reads "none") and on made-up logs with and without guards.
- ⚠️ Not run: a real checkpoint through the scripts with the guards. The one real run of the guards is section 23's
  by hand. With the start guard on, the "holds left alone" column is the guard's result in the scenes it fired in,
  not the policy's: the column says how many.

**8. A second copy of the saved states**: `~/datasets/blush_sim_states`, 3.0 GB, 2,570 files, made 2026-11-27 11:41
with `rsync -a` from `~/.cache/blush_sim/states`; every file's SHA-256 equals the original's (`MANIFEST.sha256` in
the folder: `cd ~/datasets/blush_sim_states && sha256sum -c --quiet MANIFEST.sha256`).
- ⚠️ Same disk: it covers a wiped cache folder, not a dead drive. Nothing was uploaded; a copy off the laptop needs
  the owner's word.
- Brought up to date after the run (2026-11-27 21:53, the same `rsync -a`): 3,169 files, 3.6 GB, with the states
  of all 600 new episodes; the manifest is made from the original and every file of the copy checks against it.

**9. `CLAUDE.md`**: one line under the codec rule: `libsvtav1` is for the real arm's recordings; simulator datasets
since v6 store H.264 through `sim/picture_store.py`.

**10. The camera post on the real cart**: worked out and drawn for the plank rig, not built
(`hardware/greenhouse/pipe_rail_rig/cart_camera_post.py`, its README section, `renders/cart_camera_post.png`).
- The lens points are the dataset's: 350 mm to the arm's left, 100 mm toward the row, 420 and 50 mm above the deck
  top (by ruler from the pan axis: 350 mm left, 61 mm toward the row). The rod stands at rig x -143, y 296 for arm 1.
- **Arm 2 is not a mirror image.** `armYaw = [90, -90]` in `pipe_rail_rig.fs`: arm 2 is arm 1 turned half a turn, so
  its post is arm 1's turned half a turn about the middle of the house (x 1043, y 104). A mirrored post would stand
  on arm 2's right and flip its pictures. Sections 24 and 27 said "mirror image"; corrected there. ⚠️ `scene.py`'s
  comment above `CART_CAMS` still says it: to be corrected once the run's workers have ended.
- Carried by a 350 mm boom of the 44 x 44 batten under the deck's end, turned 14 deg to pass between the wheel
  brackets, and a 500 mm M12 threaded rod through its outer end. M12 is thinner than the simulator's 16 mm post, so
  the sweeps of section 26 still hold. Everything else stays at or below the deck top. 15 clearances computed from
  the `.fs` dimensions; the tightest are 2 mm (boom end to the cleat under the arm), 9 mm (boom over the rail tubes)
  and 18 mm (rod past the printed fruit when the cart rolls).
- **⚠️ New finding: the rig's end frame is in the pictures.** In the simulator the end frame is 370 mm from the arm,
  behind the cameras. On the plank rig it is 225 mm away, so the post looks in through it: the end frame's row post
  covers 25 % of the high camera's picture and 24 % of the low one's (the left side; rig boxes projected through the
  simulator's lens model). The band where the trusses hang is clear of it (columns 260-369 and 241-354 of 640,
  against 0-235 and 0-221 for the frame).
- Three ways, the owner's call: build as drawn and see; a house 300 mm longer (end frames 375 mm from each arm,
  "Inside length" 1200: at least one more batten and a new cut plan); or put the rig's end frame into the simulated
  scene and re-make the pictures from the saved states.
- Sway of the cameras: a beam-formula estimate, not a measurement: about 22 Hz for the M12 rod on this boom, 25 Hz
  for M16 (the boom's twist sets the limit, so a thicker rod hardly helps).
- ⚠️ Weak point: only 75 mm of the boom lies under the deck, because the cleat under the arm is in the way. Its three
  screws sit close together and a knock on the rod levers on them. Notching the boom over the cleat gives more hold.
- **⚠️ Noticed on the way: the crates.** No crate is drawn on the cart. The dataset's crate (world `crate23`) lies
  133 to 363 mm to the arm's right, which on this cart is between the two arms. Placed like that, arm 1's crate is
  at rig x 358-588, y 30-310 and arm 2's at x 312-542, y 90-370: they overlap by 185 mm along the cart, and each
  reaches 51 mm past the deck's edge (deck y 81-319). Not solved here: it is a layout question for the rig.
- ⚠️ Not done: the Onshape model and the cut plan (no CAD was changed), the camera clamps, a sweep of the recorded
  episodes against the boom (it is below the deck top, where the simulator has its table).
- **⚠️ Run 2026-11-27 21:27-22:15, after the dataset run: the recorded motions do NOT stay clear of this rig's END
  POSTS**, which stand 145 mm nearer the arm than the simulator's end frame. `whatif/camera_sweep_logs.py arm
  --every 1` over the 800 old logged episodes (384,677 frames, 28 min) and, with the new `--sessions v6new`, over
  the 600 new ones (282,625 frames, 20 min; that is the dataset's 282,025 plus one per episode). 16 upright lines
  from the deck to 0.52 m: 2 for a control (the dataset's own camera post in both spellings), 9 on the row-B end
  post's two faces that look at the arm (x 0.101-0.145, y 0.378-0.422), 5 on the row-A post (x 0.101-0.145,
  y -0.022 to 0.022). The cart stands as drawn: centred, each arm 225 mm from its end frame.

  | | old 800 | new 600 |
  |---|---|---|
  | control: nearest to the dataset's camera post | 4.4 cm (seed 10005040, frame 50), as in section 26 | 4.7 cm (seed 10100680, frame 416) |
  | row B, nine lines together: a line through one of the arm's boxes | at least 50 episodes | at least 53 |
  | row B: within 1 / 2 / 4 / 6 cm | 123 / 169 / 380 / 492 | 115 / 155 / 264 / 332 |
  | row B, the corner line 0.145:0.378 alone: within 4 cm | 380 episodes, 8,259 frames | 264 episodes, 6,408 frames |
  | row A: nearest | 7.9 cm (seed 10007066, frame 18) | 8.2 cm (seed 10100680, frame 530) |

  - Where things are in simulator coordinates: the arm base at (0.37, 0.20), the row it faces at y 0.38-0.40, its
    left is -x. The simulator's end frame is three 6 mm plates at x -0.006 to 0 (`ef0_post0..2`, y 0.4, 0 and -0.4
    +/- 0.03, with collision), 370 mm left of the base; this rig's posts' inside faces are at x 0.145.
  - Cause, from four of the nearest old episodes frame by frame (seeds 10012041, 10024034, 10019045, 10002053): the
    start pose has the fixed finger's tip at (0.140, 0.196, 0.102): the hand points left, 230 mm from the base, and
    is already 5 mm past the plane of the rig's end frame, in the opening between the two row posts (per
    `pipe_rail_rig.fs` the end frame has only the posts between the floor tie and the string rails). The swing out
    to the row and the swing home from the crate both pass the corner (0.145, 0.378): the tip goes through e.g.
    (0.133, 0.337), (0.172, 0.398) on the way out and (0.151, 0.389), (0.154, 0.422) on the way home. The parts
    nearest the line are the finray plates and the fixed pad, all box geoms.
  - "At least": lines 11 mm apart, sampled every 10 mm of height, against plates a few mm thick; the -0.2 cm the
    sweep prints is half a plate's thickness (`box_dist` is negative inside a box), not a depth into the post.
  - So the longer house is needed for the motions, not only for the pictures: built as drawn the hand meets the row
    post, and re-made pictures keep the same motions. Arm 2 is the half-turn image, so the same holds at the open
    end. Written into the rig README ("Checked 2026-11-27"); the owner was told.
  - Over all 1,400 episodes (`whatif/end_post_sketch.py`, which also draws
    `hardware/greenhouse/pipe_rail_rig/renders/end_post_check.png`): the fixed finger's tip at the first frame is at
    x 0.133-0.146, y 0.183-0.209, z 0.081-0.133 (the start pose with its 3 degree change); it is past the frame's
    inside face at some frame in every episode, the 124 nothing-ripe ones included, and reaches x 0.066 at most
    (79 mm past the face, at y 0.248, z 0.262: through the opening). When, from six of the nearest episodes (old
    10012041, 10024034, 10019045, 10002053; new 10102678, 10100955; one-off, the corner line, every frame): within
    4 cm in frames 17-70 (the swing out) and 349-406 (the swing home), in 10012041 also 106-121 (at the cut: its
    target hangs on the leftmost stem). The finray box geoms are 1.2 to 4 mm thick.
  - **The plants and the cut truss** (`plants --n 0 --every 3`, 16 lines, old in two halves 21:27-22:25, new with
    `--sessions v6new` 22:06-22:43; no scene skipped). Control: 1.7 cm for the old (seed 10023057, frame 207), as in
    section 26. Row B, nine lines together: within 4 cm in 207 of 735 old and 158 of 540 new cut episodes, within
    2 cm in 67 / 58, a ball on a line in 34 / 37, nearest -1.3 / -1.4 cm (a fruit's ball has a 10-15 mm radius).
    Row A: within 4 cm in 8 / 11, on a line in 4 / 2.
  - ⚠️ What that is: NOT mostly trusses hanging there. One-off lookup of the nearest part at the named frame in
    eight episodes (row B: 10022070, 10004032, 10102048; row A: 10102852, 10021000, 10000046; the dataset's camera
    post: 10100475, 10100422): each time a fruit of the TARGET truss, which at the first frame was 19 to 41 cm from
    that line.
    Frame by frame in 10100475, 10102852, 10021000: after the cut the hand does not go right. It pulls the truss to
    the arm's left (tip to x 0.08, 29 cm left of the base, z 0.28-0.30), then lifts to z 0.45-0.49 and passes over
    the arm's own base to the crate. The truss dangles below and outward on the way (a fruit at x 0.02-0.07).
  - The string rails of the rig (underside 0.52 m above the arm base, along both row lines): one-off look over all
    1,400 episodes with the arm's parts as bounding boxes: the highest top under the row-B rail is z 0.478 (seed
    10023024, frame 447), under the row-A rail 0.451: at least 42 and 69 mm of air. The highest anywhere is 0.527,
    over the arm's own base (seed 10100172), where the rig has nothing.
  - **How much of that is a truss hanging there before anything moves:** `plants --n 0 --every 100000` (the first
    frame only, the same 16 lines; old 22:27-23:05, new 23:05-23:29). Within 4 cm of a row-B line at the first
    frame in 94 old / 65 new episodes (nearest 2.3 / 2.5 cm, no ball on a line); none at row A (nearest 12.0 /
    13.3 cm) and none at the dataset's camera post (8.9 / 8.2 cm). So a part gets within 4 cm only during the
    episode in 113 old / 93 new, and every ball on a line (34 / 37) comes from motion.
  - The old `plants` runs measure 735 scenes, not 736: seed 10014003 (v5 episode 353) does not rebuild as recorded
    and is skipped, the same episode v6 lacks (section 26). In the arm sweep it is 4.2 cm from the corner line, so
    the arm's 50 / 123 / 169 / 380 also hold for v6's 799 re-made episodes (492 within 6 cm becomes 491).
  - **The longer house** (way 2, inside face of the row post 375 mm from the arm): `arm --every 1
    --cams=g_0.02_0.30_0.42,-0.005:0.378:0.52,...` with nine lines at x -0.005 to -0.049, y 0.378-0.422 (old
    22:42-23:02, new 23:02-23:16). Nearest 10.0 cm old (seed 10005040, frame 59) and 9.4 cm new (seed 10100680,
    frame 407); no episode within 6 cm of any line; the control gives 4.4 / 4.7 cm again. That post's inside face
    lies 5 mm behind the front of the simulator's end-frame plate `ef0_post0` (`scene.greenhouse_xml`: x -0.006 to
    0, y 0.37-0.43, to z 0.59, a solid in every recording), which is 60 mm wide against the post's 44.
- **⚠️ New, from the control line of the same runs: in the NEW episodes the cut truss comes nearer the dataset's own
  camera post than in the old ones.** Section 26 checked the 800 old episodes only. Arm: never nearer than 4.7 cm.
  Plants: within 4 cm of the line under the lenses in 15 of the 540 new cut episodes (old: 10 of 735), within 2 cm
  in 4, and in one a fruit's ball lies on that line (seed 10100475, frame 315: fruit s2t1_t4 at (0.025, 0.299,
  0.122), carried by the left).
  - `camera_sweep_logs.py touch --sessions v6new --from new_arm_eps.json new_plants_eps_0.json
    new_plants_eps_1.json --cams g_0.02_0.30_0.42` (23:00-23:06): the 36 new episodes within 6 cm of that line,
    replayed in the recorder's own model. 35 touch nothing. **One does: seed 10100475**: the fruit s2t1_t4 and
    s2t1_t5 are in contact with the high camera's post in 4 frames (313, 314, 315, 317), 6.0 mm deep at most.
  - ⚠️ **Corrected 2026-11-27 23:47: that is a real knock, not a pass through something that was not there.** This
    bullet (and both READMEs) first said the post was drawn in the new episodes but was not a solid when they
    were recorded. Wrong. With `--rig trio --layout-rig base` the scene is DECIDED with the old bracket (which
    plants, which truss is ripe); then `truss_scene._build_once` composes the model again with rig trio's
    brackets, as ordinary solids, and without the old one. Checked on that seed's model built as the recorder
    builds it (`ts.build(seed, mounts=RIG_FIXED["base"], show_mounts=RIG_FIXED["trio"], jaw="v5")`, the call in
    `so101_mujoco._build`): `high_cam_post`, `low_cam_post` and both housings have contype 1 and conaffinity 1,
    and no `base_cam_*` geom is in it. The logged motion says the same: the two fruit come at the post at
    1.1 m/s (frames 310-311), are down to 0.1-0.3 m/s at its surface in frame 313 (6.0 mm into it in that one
    frame, which is what a contact with solref 0.02 gives at that speed), stay against it to frame 317 and slip
    past. The cut was at frame 181; this is the carry, 4.4 s later. (One-off script, not in the repo.)
  - So: the 799 re-made episodes were recorded with the old bracket and have the new ones drawn afterwards
    (nothing touches the new ones in the 32 nearest, and 3 graze the old one that is no longer drawn: section
    26). The 600 new ones were recorded with the new ones, solid, and without the old one.
  - That is part `v6_new_p06` episode 4 = **episode 953 of the merged dataset** (found by its joint readings, equal
    bit for bit; 505 frames). Its pictures around frames 300-331, looked at once: the truss swings past the low
    camera at close range and leaves its view, and the claw camera sees the post beside the truss.
  - One rough carry in 1,399 episodes, and its pictures and physics agree. Left in the dataset; taking it out is
    the owner's call. Written into `sim/README.md` ("Dataset v6", state) and the rig README's camera-post section.
