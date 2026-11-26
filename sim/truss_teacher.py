"""Teacher v5: the scripted truss picker as a closed-loop teacher (post-mortem §6.1, K7 approved 2026-11-09).

v5 (K7 2026-11-18: "a better teacher which has a success rate of at least 90 %"; v4 kept 80 of the 154 held-out
scenes of the 23 x 28 crate world): v4's search unchanged (phases A and B: every scene v4 plans keeps its first plan,
action for action, except that 8-fruit trusses are now carried at CARRY_SLOW_BIG of the pace), then phase C when they
find nothing: finer cut points (CUT_U_V5), the stalk tilted in the jaw
(TILT_DEG; the square cut pose can fold the arm into itself), position-only retreats (V5_RETREATS), more release poses
(RELEASE_ALTERNATES_V5, RELAXED_*), longer standoffs (STANDOFF_M_V5), three more fixed via-poses (VIA_NORM_V5, found
from offline RRT routes on scenes kept apart; RRT still never drives), the "push" neighbour mode, within PHASE_C_MAX.
next_plan() goes on with the same fixed order after a plan fails its dry run (generate_truss --retries): the same cut
with the next release pose, lowest first, after a failure from the carry on; else another approach. Heavy trusses
(BIG_TRUSS_N) are carried slower. Planning reads a frozen copy of the start frame. The cue marks the plan's cut point
(meta cut_u; the plugin's new_scene(cut_u=)).

v4 (K7 2026-11-17: "do the teacher v4"): v3 + the release fallback (RELEASE_FALLBACK_N): when no plan is found, a
second pass retries the candidates that failed at the release or the carry with the next-ranked release poses. Every
scene v3 plans keeps its plan, action for action. Datasets named *_teacher_v3* were recorded by v3 with the old crate
(truss_scene.CRATE_*, changed the same day): v4 data is a new dataset version; v4 was never recorded, v5 is.

    teacher = TrussTeacher(robot.m, robot.d, robot.info, k).plan()      # after the start frame; PlanError if none
    while not teacher.done:
        robot.send_action(teacher.act(state_norm))                      # the measured state, every frame

What changed from truss_expert.TrussPlanner (the v2 expert, which recorded blush_sim_truss_v2):
- Same scene, same motion. No random speed, holds, durations, cut point, approach order or retreat length, and no
  deliberate misses. Every choice comes from a fixed priority list checked against the scene (first clear one
  wins: CUT_U, APPROACH_BLEND_DEG, SLIDE_M, STANDOFF_M, retreat options), so it depends only on the scene.
  The post-mortem's R2: the camera can't see a coin-flip, so the policy averages over them.
- Closed loop. act() works from the MEASURED arm state every frame: it projects the state onto the current leg's
  joint path and commands a point a fixed lead further along it (pure pursuit), so after a push it steers back to
  the path instead of replaying a timed trajectory. That is what makes noise injection (§6.2) and DAgger labels
  (§6.4) possible. resync() re-enters the right leg from a state the teacher didn't drive to (DAgger, later).
- RRT checks, it doesn't drive. ⚠️ Supersedes "RRT for every arm move" (K7 2026-11-01) for the teacher, by K7's
  approval of §6.1 on 2026-11-09: every free move is a fixed template (a straight joint-space line), checked with
  rrt.Checker; a blocked template gets ONE fixed detour, else the candidate is dropped. The drop rate is measured
  (check_teacher.py solve), because a consistent teacher that can't solve some scenes shrinks what the data covers
  (Motion Policy Networks).
- The cut lands where the cue says. ⚠️ v2 cut at a random point 20-80 % along the cut zone while the drawn cross
  and observation.environment_state always marked 50 % (truss_target.cut_pixel, 2026-11-09 finding). The teacher
  tries CUT_U in order, 0.5 first, and reports the one it used (meta["cut_u"]); the recorder must draw and record
  the cut point at that u.

The motion (§6.1 sketch):
  start -> standoff (STANDOFF_M behind the pre-cut pose, straight joint line) -> pre-cut pose (SLIDE_M back along
  the approach axis, straight Cartesian line) -> slide straight in, slowly -> close, squeeze (fixed times) ->
  retreat straight back and up -> over to the release pose above the crate -> open, wait -> back to start ->
  short hold. Each free move (start -> standoff, retreat -> crate, crate -> start) is a straight joint line, or when
  that is blocked the one detour: up over a fixed via-pose high in front of the row, then down (VIA_NORM), or
  two such vias chained.
  ⚠️ §6.1 named a fixed via-pose over the crate; one constant pose 20 cm over the crate's centre hit a neighbour
  truss's fruit with the cut truss in hand in 2 of 3 test scenes (2026-11-09), so the carry uses the same rule.

Reused from truss_expert.py (B.L.U.S.H commit 6d1e02d): the obstacle sets and checker (TrussPlanner.__init__), the
IK, the carried-truss checker (CarryChecker), the release-pose fit (crate_poses + laid_flat, deterministic: its
samples come from a fixed seed). Timing constants come from expert.py; the sag correction is expert.Plan's,
applied to the cross-track error so the lead itself doesn't wind it up.
"""
import collections
import copy
import json
import os
import re
import weakref

import mujoco
import numpy as np

import arm_map
import expert
import scene
import truss_expert as tx
import truss_ik
import truss_scene as ts

FPS = expert.FPS
PlanError = tx.PlanError

# --- fixed priority lists (first that works wins; design choices, 2026-11-09) ---
CUT_U = (0.5, 0.4, 0.6, 0.3, 0.7)          # along the cut zone; 0.5 is where the cue points (truss_target.cut_pixel)
APPROACH_BLEND_DEG = tx.APPROACH_BLEND_DEG  # (0, 30, -30, 50, -50), now in this order, not shuffled
SLIDE_M = (tx.SLIDE_M,) + tx.SLIDE_FALLBACK_M   # 4.5 cm, then 3.5 / 2.5 cm at the reach edge (v2's values)
STANDOFF_M = (0.04, 0.02, 0.0)             # standoff this far behind the pre-cut pose (guess, unverified)
APPROACH_STEPS = 8                         # IK way-points from standoff to pre-cut (as the slide's 10, guess)
RETREAT_M, RETREAT_UP_M = 0.04, 0.03       # v2 drew 3-5 cm back and 2-4 cm up; now the middle of each
# Every free move: the template is one straight joint-space line; the one detour goes over a fixed via-pose high in
# front of the row (§6.1 "up, across, down"), the first of VIA_NORM that's clear. ⚠️ The trusses hang down and out
# into the aisle (fruit 3-22 cm high, out to 6 cm from the arm's base line) while the cut points sit near the stem at
# ~28 cm and the start pose low at 11 cm: a straight line from start to standoff crossed fruit in 3 of 3 test scenes,
# and rising straight up with the orientation held ran out of reach (2026-11-09). The v2 RRT's routes went over such
# a high pose (grip 34-46 cm up, 25-36 cm out) when the straight line was blocked (10 test scenes, 2026-11-09).
# Joint angles in norm units (arm_map, use_degrees=false), copied from three of those v2 routes; the grip point there
# in the comment. Tried nearest the move's goal first (joint space), so the choice depends only on the scene.
# ⚠️ Defined by IK as (point, pitch, tilt 0) first: unreachable, no detour was ever tried (2026-11-09).
VIA_NORM = ((-59.0, -44.0, 8.0, -34.0, 6.0),     # grip (0.27, 0.29, 0.46): high, left of centre (seed 9100009)
            (-54.0, -46.0, 44.0, -35.0, 19.0),   # grip (0.18, 0.36, 0.34): left (seed 9100005)
            (21.0, -64.0, -6.0, 76.0, 61.0),     # grip (0.42, 0.37, 0.32): right (seed 9100005)
            # Added from a search (2026-11-09): v2 RRT routes on seeds 9200000-9200059 (apart from the test seeds),
            # way-points pooled, the one that unblocks most scenes the line and the vias above can't reach kept.
            # 15 of 60 scenes were blocked start -> standoff; this pose unblocks 3, no other pose 2 more.
            (-90.0, -19.0, 41.0, -56.0, 22.0))   # gripper (0.18, 0.21, 0.23): the arm raised in place over its base
# ⚠️ 12 of those 15 scenes had no single clear via in 75 RRT way-points: their routes needed two. So the last
# fallback chains two vias (up out of the start, then over: §6.1's "up, across, down"), pairs in a fixed order.
# Teacher v5 (phase C and rejoin only, after every VIA_NORM route): more fixed via-poses found the same way, from
# offline RRT routes on scenes kept apart from the tuning set and the gate (via_search, 2026-11-18). In 3 of 4 v5
# scenes blocked only by the way in, an offline RRT reached the standoff (046 30 of 30 candidates, 022 15 of 20,
# 004 7 of 15, 081 0 of 12): routes exist, the four poses miss them. Still fixed poses, never an RRT route (rule 3).
# Search (sim/whatif/via_search.py, via_pick.py): 60 scenes from 9940000, 18 with a v5
# candidate blocked only on the way in (103 cases), RRT routes in 11; their 161 way-points clustered (12 norm units)
# into 99 candidates, each scored alone and chained with VIA_NORM's; kept greedily by new scenes unblocked: 7, 2, 1
# (10 of the 18; 2026-11-18).
VIA_NORM_V5 = ((-48.1, -35.1, -61.0, 88.0, 42.8),   # unblocked 7 of the 18 scenes
               (-92.6, -49.7, 77.8, -52.0, 5.3),    # +2
               (-55.0, -41.9, 74.0, -26.4, 17.6))   # +1
# Round 3 (K7 2026-11-26: "apply changes 3 and 4 and see what it does"), old setup (jaw v5, base 0.37/0.20, old start
# pose, crate 30 x 28): at 86 % (155 of 180 tuning scenes) the teacher fails by finding NO plan (22 scenes: the way in
# from the start pose blocked in 13, the carry to the crate in 4), hardly ever once it has one (155 of 157). More fixed
# via-poses, found as before from offline RRT routes (whatif/via_search.py with VIA_EXTRA=1 for the way in,
# whatif/via_search_crate.py for the carry) on scenes 9740000+, which nothing else uses. ⚠️ Used ONLY in a last pass
# after phase C found nothing more (_search_plans), so a scene that had a plan keeps exactly that plan; alone or
# chained with any other via ("v5b:<i>", "v5b:<i>-<j>"). Empty = off (the teacher as it was).
VIA_NORM_V5B = ()
# --- timing: fixed, the middle of v2's ranges ---
CLOSE_S = 0.65                             # v2 0.5-0.8
SQUEEZE_S = 0.75                           # v2 0.6-0.9 (the cut lands ~0.45 s into the close)
SETTLE_S = 0.2                             # v2 0.15-0.3 at the stalk before closing
RELEASE_S = 0.5                            # jaw opening over the crate (v2's fixed 0.5)
DROP_S = 0.65                              # v2 0.5-0.8
HOLD_END_S = 0.3                           # short: identical frames at home froze the n10 ACT run (post-mortem §2)
CLOSE_NORM = 1.0                           # the middle of expert.CLOSE_NORM (0-2)
# --- closed loop ---
# Lead of the commanded point along the path, norm units per frame. expert.JOINT_PEAK_NORM_S 60 /s = 2 per frame;
# carrying at tx.CARRY_SPEED. Cartesian legs are timed by CART_M_S along the line (v2's min-jerk average for
# CART_PEAK_M_S 0.08 m/s: 0.08 / 1.875).
FREE_LEAD = expert.JOINT_PEAK_NORM_S / FPS
CARRY_LEAD = FREE_LEAD * tx.CARRY_SPEED
CART_M_S = expert.CART_PEAK_M_S / 1.875
APPROACH_M_S = 2 * CART_M_S                # standoff -> pre-cut: nothing in reach yet, twice the slide's pace (guess)
END_TOL = {"servo": 0.5, "free": 2.0}      # norm units from a leg's end to call it reached (guess)
STALL_NORM, STALL_FRAMES = 0.05, 8         # ...or the arm moved less than this per frame for this many (sag, contact)
SEARCH_BACK = 0.5                          # projection window: this many leads behind the last projection...
SEARCH_AHEAD = 6.0                         # ...and this many ahead (keeps it from jumping where a path folds back)
MAX_S = 45.0                               # done after this long whatever happens (a stuck leg can't loop forever)
# --- rejoin (2026-11-16): taking over from a state the teacher didn't drive to (hybrid hand-over, DAgger) ---
ESCAPE_M = (0.02, 0.04, 0.06)              # back out along the slide line this far, the first that is clear (guess)
ESCAPE_STEPS = 8                           # IK points along that line
# ⚠️ With the arm pushed 2 cm in a random direction the slide-line escape found no clear pose in 11 of 126 runs
# (2026-11-16): then also straight back along the gripper's own finger axis, then straight up (position only).
ESCAPE_DIRS = ("slide", "back", "up")
ESCAPE_STALL_FRAMES = 30                   # an escape that stops moving for 1 s is over: go on from there (guess)
# ⚠️ No wait at the standoff. A hold there until the stalk hung still (3 mm, 10 mm/s, at most 3 s) was tried for the
# 5 of 18 hard takeovers that escaped and then slid onto empty air (2026-11-16): it fixed none of the 18, and on the
# 10 seeds that changed between rejoin versions it caused 5 failures (neighbours nudged to 15-17 mm, a drop) against
# 2 it prevented (check_teacher takeover, deterministic, same 40 runs with and without it). Removed.
# --- release fallback (teacher v4, K7 2026-11-17; measured first as a what-if,
# sim/whatif/release_fallback.patch) ---
# ⚠️ _release keeps ONE pose (the clear one with most room to the walls) and the carry is planned to it alone: when the
# carry there or the way home from it is blocked, the candidate (cut point, approach) is dropped, though a lower-ranked
# pose may clear. Seed 9900001 (stock crate, strict, u 0.5, straight in): the chosen pose's carry hit s0t0's peduncle
# and its way home was blocked too, the 4th-ranked pose cleared both (2026-11-17). With the bigger crate A all 50
# release calls of seed 9900029 returned the same pose and every carry to it was blocked (agents/crate, 2026-11-16).
# ⚠️ That one this doesn't fix: its carry hits s1t0's fruit 10 % of the way out of the retreat end, whatever the
# pose (all 330 alternates blocked, 2026-11-17). Where the block sits at the retreat end, the retreat is the lever.
# So plan() makes a second pass, only when the first found no plan at all: every scene that plans keeps its plan,
# action for action. ⚠️ A fallback inside each candidate could change at least 4 of the 63 solved held-out plans
# (seeds 9900048, 085, 130, 134: a carry blocked at one approach, solved at a later one; meta "tried" lists only the
# winning mode's failures). The second pass retries only the candidates that got as far as the release, with the
# next-ranked distinct clear poses after the first choice (which failed), ranked past _release's first
# RELEASE_SAMPLES // 4 too. Measured on the 154 held-out seeds, stock crate (agents/release, 2026-11-17): 63 -> 67
# kept, none lost; every first-pass plan kept its release point and result (actions compared on 8: identical).
RELEASE_FALLBACK_N = 10                    # alternates per candidate (guess, between 10 and 20; 9900029 crate A had 9
#                                            clear poses of 24 that fit, 9900001 stock 4 of 4; the kept plans it found
#                                            used alternates 1-3, the highest used 6, 2026-11-17)
RELEASE_DISTINCT_M = 0.03                  # an alternate's grip point this far from every clear pose before it (guess,
#                                            "a few cm"; the clear poses of those two seeds lay 2.4-23 cm apart)
# --- teacher v5 (K7 2026-11-18: "a better teacher which has a success rate of at least 90 %"; v4 kept 80 of 154
# held-out scenes in the 23 x 28 crate world, 2026-11-17) ---
# v5 = v4's search unchanged (phases "A" and "B": every scene v4 plans keeps its plan), then phase "C" when those find
# nothing: the finer cut points below, v4's retreats plus V5_RETREATS, more release poses, and the "push" neighbour
# mode. 52 of v4's 74 failures had no plan, and in 47 of those 52 some candidate's slide was clear: what stopped it
# came after (start -> standoff 21, retreat 15, carry / release 11; 2026-11-18 census).
# ⚠️ Still no RRT: every free move stays the template, a via-pose or two (post-mortem §6.1 rule 3, RRT checks, it
# doesn't drive: its routes are coin-flips the camera can't see, R2). An RRT fallback on every blocked move was tried on
# 2026-11-16 and planned 5 of 15 move-blocked scenes, 2 of which passed their dry run; in the still-blocked ones the
# standoff was free and an 8000-iteration RRT found a way in 1 of 12 tries (post-mortem §6.11).
CUT_U_V5 = (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.25, 0.75, 0.2, 0.8)   # v2 drew u in 0.2-0.8; the
#                                          cue (cross + environment_state) marks the u the plan cuts at (meta cut_u)
RELEASE_ALTERNATES_V5 = 20                 # release poses per candidate in phase C after the first (guess)
# ...then poses the fit rejects (phase C only): the predicted outline up to 1 cm past a wall, the pose itself 1 cm from
# neighbours instead of 2. The dry run judges the landing: the fit's landing model is off by 42-131 mm against the
# 5-33 mm of room it plans for (post-mortem §6.11), and 8-fruit trusses had 10-12 fitting poses of 2144, all blocked
# (9900044, 2026-11-18). RELAXED_N of them per candidate (guess).
RELAXED_SLACK_M = -0.01
RELAXED_N = 10
# Phase C standoffs: v4's, then further back along the approach axis, so the free move can end outside the crowd (in
# 9900004 / 046 the short ones were reachable for no candidate, every route blocked by trusses; guess).
STANDOFF_M_V5 = STANDOFF_M + (0.06, 0.08, 0.10)
# After a release that failed in the physics (generate_truss --retries, same candidate), the next poses go lowest drop
# first (the lowest fruit nearest the crate floor), not most room first: an 8-fruit truss dropped from a grip 19-22 cm
# up tipped out of the crate at 5 alternates and landed in it from 9 cm (9900042, 2026-11-18).
# Heavy trusses are carried slower: retreat and carry at this share of the pace from BIG_TRUSS_N fruit. 9900042
# (8 fruit) lost a fruit on the way to the crate in 7 of 7 plans at the normal pace, none at 0.6, 0.4 or 0.25
# (2026-11-18). The fruit count shows in the picture, so the pace varies with what the camera sees (R2).
BIG_TRUSS_N = 8
CARRY_SLOW_BIG = 0.6
# --- v6 world (jaw "v6", 30 x 28 crate; K7 2026-11-24). Only with the v6 jaw: a v5 scene plans as before. Grid round 2
# (base_grid.py, 30 tuning scenes per base, first plan only): the truss's own fruit came off while LOWERING it into the
# crate in 3-4 of 30 scenes at every base, and on the way HOME in 1-2 (the way home ignores the dropped truss); with the
# new start poses also while carrying (2-4 of 30). The v6 jaw lets the held truss swing more than v5's (catch map: drift
# 0.5 s after the cut median 13.7 mm, v5 8.0). ---
V6_TEACHER = True            # False (A/B only, 2026-11-26): a v6-jaw scene planned with the v5 teacher's logic, i.e.
#                              every V6_* change below off at once; with BLUSH_V6 {"V6_CARRY": ["rigid", 0]} too
#                              that is "old teacher, new jaw" (the aim point stays the v6 jaw's, truss_ik)
V6_MEET_FROM_M = None        # None = off: the slide ends at the aim point. A value (A/B 2026-11-26: 0.008 together with
#                              V6_AIM_FROM_PAD_M 0.002): slide in with the stalk THIS far from the fixed pad (room for
#                              an aim error on the fixed finger's side), then move the jaw sideways until the stalk is
#                              V6_AIM_FROM_PAD_M from the pad, then close. Why: with the stalk 8 mm from the fixed pad
#                              the closing jaw pushes it ~11 mm sideways before the cut (carry_bump_trace.py on the cut
#                              legs, 8 scenes: the stalk tilts 8-15 deg and the truss swings into its neighbours; v5
#                              jaw, aimed 1 mm off its pad: 5-8 deg). Catch map (v6): held 9/9 with the stalk 2-5 mm
#                              from the pad, 4-5/9 at 8-11 mm. Scored on the 180 tuning scenes: 129, the plain 2 mm
#                              aim 131 (won 2, lost 4; two scenes end on a stalled joint) -> stays off; the plain
#                              2 mm aim is the default since 2026-11-26 (truss_ik.V6_AIM_FROM_PAD_M)
V6_MEET_STEPS = 4            # IK way-points along that sideways move (6 mm: 2 mm apart; design choice)
V6_CARRY_SCALE = 0.67        # every carry at 0.67 x v4's pace = 0.4 x the joint speed (v5: 0.6, 8-fruit trusses 0.36)
V6_SETTLE_ABOVE_S = 0.8      # hold above the crate before lowering, so the swing dies down (guess: ~2 swings of a
#                              12-15 cm pendulum)
V6_LIFT_HOME = True          # after the drop: straight back up the lowering line, the way home planned from up there
# ⚠️ The truss does NOT stay as it was at the cut (tx.CARRY "rigid", measured on one seed 2026-11-03): traced on seed
# 9730005 (fruit_off_trace.py, 2026-11-24) it turned 57 deg in the v5 jaw by the carry and 67 deg while lowering, and
# ~110 deg in the v6 jaw, so its tip fruit hung lower than planned and was pressed into the crate floor (1.4 N on the
# fruit, its tether at 0.74 N of a 0.59 N limit). It rolls about its STALK (tx.CARRY_AXIS note). Release fit (v6):
# every pose judged as held at the cut and as hanging under the stalk (tx.CARRY_AXIS 1): the tighter wall room of the
# two, and the hanging truss's lowest fruit V6_FLOOR_CLEAR_M to V6_DROP_MAX_M over the floor / rim.
V6_FIT = "hang"              # "hang" (as cut + fully hanging), "envelope" (+ every angle between and 25 deg past),
#                              "rigid" (v5's fit). "envelope" lost seeds on the 5 traced ones (2026-11-24); hang +
#                              tx.V6_CARRY hang 20/30 vs rigid 18/30 on r2_today (2026-11-25, within the noise)
V6_LEAN = True               # the laid-flat outline may reach tx.LEAN_M past a wall (v5's rule); False lost 3 of 5
V6_REQUIRE_DROP = False      # only release poses with the straight lowering line (untested on its own)
V6_FLOOR_CLEAR_M = 0.015     # the fit: lowest fruit (held or hanging) this far over the floor
V6_DROP_MAX_M = 0.12         # the fit: the held truss's lowest fruit at most this far over the rim
V6_STOP_ABOVE_FLOOR_M = 0.025   # the closed-loop lowering stops with the live lowest fruit this far over the floor
# Live release (2026-11-24): on reaching the crate, re-plan the release from the truss as it hangs NOW (the live bodies,
# held as they are for the last short move), the same fit and checks as at planning time. Motivated by ~22 cm model
# error on 9730015, which was NOT typical (CORRECTED 2026-11-25: that truss had nearly dropped; over 8 seeds the hang
# model is 5-7 cm off, a consistent offset; the stalk is ONE rigid body (truss.py), there is no rachis joint to droop).
# ⚠️ OFF: A/B on the 30 r2_today tuning scenes (2026-11-25, base_grid.py): with it 8/30, without 20/30 (round 2: 19),
# 10 of its failures "not all in the crate". Kept behind the switch, unused.
V6_LIVE_RELEASE = False
# ...and not only fully hanging: on seed 9730005 the truss had turned 62 deg when lowered and its tip fruit was lowest
# part-way (2026-11-24). So the fit takes the worst wall room and the lowest fruit over the whole swing: from as cut to
# fully hanging, plus V6_SWING_OVER_DEG past it either way (it still swings), in V6_SWING_STEPS angles.
V6_SWING_OVER_DEG = 25.0
V6_SWING_STEPS = 9
# ⚠️ The truss also TURNS IN THE GRIP on the way to the crate: traced 2026-11-25 (fruit_off_trace.py, base 0.34/0.14),
# five plans ended the carry with the stalk 55-85 deg off the pads' line (the hang model then 10-40 cm off; the same
# neighbour hit at ~0.9 of to_crate on every retry, 9730016/019), four stayed at 4-33 deg. Nothing touched the truss
# (it turns ~1-1.5 deg/frame under its weight); stiffer pad springs (5.3 / 9.5 N vs 4.25 N) only cut it to 45-68 deg.
# What separates them is how the gripper turns from the cut to the crate (planned joints, FK, 9 seeds): about the
# closing axis (gripper x, the pads' normal) 55-140 deg in the five, 11-26 deg in the four (theirs mostly about the
# stalk, gripper y). The flat pads hold the stalk's roll and slide but not a turn in their own plane: the truss keeps
# hanging as gravity pulls while the pads turn round it. So the fit measures, per pose, how far gravity's direction
# swings in the pads' plane (gripper y-z) from the cut (meta release_swing_deg) and, with V6_PAD_SWING_MAX_DEG set,
# keeps only poses under it. (A first try, the hanging truss's lever about the closing axis, did NOT separate them:
# 41-132 mm for every chosen pose, steady ones included; a 15 / 25 mm cap left 19 / 24 of 28 plans, 2026-11-25.)
V6_PAD_SWING_MAX_DEG = None  # None = off (measured only); value from the 30-scene plan census + traces
V6_PAD_SWING_ORDER = False   # True: the cap ORDERS instead of filtering (poses under it first, the rest after), so a
#                              scene keeps its plans (a 20 deg filter left 26 of 28 plans on 30 scenes, 2026-11-25)

# Gravity-aligned carry (2026-11-25, research_notes/Truss grip and carry research + whatif/pad_hold_torque.py). The pads
# cannot stop the truss turning in the grip (flat 13 mm pads at 4 N: ~0.03 N.m with rubber, the carried truss loads
# them with 0.05-0.07; papers: let a pinched object hang, don't fight it), and a real jaw won't either. But gravity as
# the GRIPPER sees it only depends on the fingers' pitch and the jaw's tilt (rows of R: R[2, :]); the pan joint turns
# about the vertical and changes neither. So with V6_ALIGN the release poses are built at the cut pose's own pitch and
# tilt (ArmIK.solve holds both anywhere it reaches): the truss keeps hanging exactly as it hung at the cut, nothing
# makes it turn, and the carried-truss model "as at the cut" (tx.V6_CARRY rigid) is the right one. Run it with
# BLUSH_V6='{"V6_ALIGN": true, "V6_CARRY": ["rigid", 0]}'. Off by default until the 180-seed A/B.
V6_ALIGN = False
V6_ALIGN_STEP_DEG = 4.0      # release poses: lift, elbow and pan on a grid this fine (design choice)
V6_ALIGN_INSET_M = 0.03      # ... their grip points at least this far inside the crate walls (design choice)
V6_ALIGN_LOW_MAX_M = 0.10    # ... the hanging truss's lowest point at most this far over the crate floor
V6_ALIGN_IK_ITERS = 40       # Levenberg-Marquardt steps per aligned retreat pose, warm-started (ArmIK.solve uses 80)
V6_ALIGN_TUCK_M = (0.20, 0.16, 0.13, 0.25)   # crane move: grip point this far from the pan axis while the arm turns
V6_ALIGN_OVER_RIM_M = (0.03, 0.07)     # ... with the hanging truss's lowest point this far over the crate rim
V6_ALIGN_VIA_TOL_M = 0.02              # ... taken from the grid when a pose is this close (grid: ~1-2 cm)
V6_ALIGN_FALLBACK = False    # True: when no aligned carry is clear, fall back to the old via-poses (they turn the
#                              gripper against gravity on the way: the truss swings, 9730005 / 016 / 027, 2026-11-25)
V6_ALIGN_TOL_DEG = 5.0       # a pose counts as aligned when gravity in its gripper frame is this close to the cut's
# The fruit also dangle on their own pedicels, lower than any rigid model of the truss puts them: with the fit above,
# 9730005 still pressed its tip fruit into the floor while lowering, and 9730018 / 9730015 dragged fruit over the crate
# +y wall at the end of the carry (1.2-2.4 N, 2026-11-24). So in the v6 world the carry ends V6_ABOVE_CRATE_M over the
# release pose (v5: tx.ABOVE_CRATE_M 0.08), and the lowering is closed-loop on the truss as it really hangs (the teacher
# reads the simulator, the policy only sees the cameras): it stops once the target's lowest fruit is V6_FLOOR_CLEAR_M
# over the crate floor, and the release + drop happen there.
V6_ABOVE_CRATE_M = 0.15
# Phase C's order and budget: cut point, then (strict and brush first, each at every stalk tilt, then push at every
# tilt) approach angle, so a clean tilted cut comes before pressing neighbours, and the long tail of cut points
# comes last (every phase-C plan kept on the tuning set so far cut at u 0.5, 2026-11-18). At most PHASE_C_MAX
# candidates past their cut leg per scene, and a standoff within
# STD_MEMO_RAD of one no route reached counts as blocked: a scene with no plan took 600 s, most of it routes to
# near-identical standoffs (guesses; deterministic: they depend only on the scene).
PHASE_C_MAX = 150
STD_MEMO_RAD = 0.02
# What-if only, OFF unless BLUSH_TEACHER_RRT=1 (2026-11-18, for K7's decision; no recording uses it): phase D, after
# phase C finds nothing, lets a seeded RRT-Connect route drive start -> standoff for the first RRT_D_MAX candidates
# whose cut leg is clear. ⚠️ Reverses the post-mortem's rule 3 (RRT checks, it doesn't drive: its routes are
# coin-flips the camera can't see, R2).
RRT_D = os.environ.get("BLUSH_TEACHER_RRT") == "1"
RRT_D_MAX, RRT_D_ITERS = 12, 4000
# Retreats added in phase C, after v4's five (m: back along the approach, toward the aisle, up). Position only: the jaw
# keeps the truss, its angle may turn (v4's retreat IK held the stalk's angle: "no IK for the retreat" in 15 of 154,
# and 10 carries were blocked right where the arm pulls out of the plant, whatever the release pose, 2026-11-17).
V5_RETREATS = ((0.04, 0.0, 0.03), (0.0, 0.04, 0.03), (0.0, 0.0, 0.04), (0.06, 0.0, 0.04), (0.0, 0.06, 0.04),
               (0.0, 0.0, 0.07), (0.03, 0.03, 0.05), (0.06, 0.0, 0.0), (0.0, -0.03, 0.05), (0.08, 0.0, 0.06))
# Neighbour modes: v4's strict and brush (tx.NEIGHBOUR_CLEAR / OWN_FRUIT_CLEAR), then in phase C "push": the arm may
# press other trusses by up to 6 mm and the target's own fruit by 3 mm on the way in (guess; brush is 2 and 2). The dry
# run's judge is unchanged: no fruit off, no neighbour moved more than 15 mm. Crews push trusses aside the same way.
# Not for the carry: the swinging truss in hand keeps tx.CARRY_CLEAR_M (1 cm tore fruit off, 2026-11-03).
MODES = {m: (tx.NEIGHBOUR_CLEAR[m], tx.OWN_FRUIT_CLEAR[m]) for m in tx.NEIGHBOUR_CLEAR}
MODES["push"] = (-0.006, -0.003)
# Stalk tilted in the jaw (truss_ik.tilt_axis): in phase C, after the square cut (0), the stalk may cross the fingers
# tilted about the jaw's closing axis. ⚠️ The square cut pose is unique for a 5-joint arm and can fold it into itself:
# in 9900093, 109, 112 and 116 (v4: no cut leg at all) the only cut pose put the wrist camera 6-15 mm into the shoulder
# and upper arm, while tilted 10-40 deg the cut pose was free and its fingers pointed further down (0.67-0.98 against
# 0.34-0.60; 2026-11-18). Smallest tilt first (guess).
TILT_DEG = (0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0)


# ⚠️ Self-collision (2026-11-09): rrt.Checker skips every contact between two arm parts, so neither the v2 expert's
# RRT nor this teacher saw the arm fold into itself. At the cut pose of seed 9100090 the wrist camera mount pressed
# into the upper arm at 48 N; the elbow servo sat at its full 3.35 N.m (gravity there: 0.4 N.m) and the slide stalled
# until the time limit, the stalk never reached (5 of 95 planned runs, 2026-11-09). Every checker the teacher uses now
# also reports arm parts closer than SELF_CLEARANCE, except pairs inside the hand (HAND_PARTS touch by design).
SELF_CLEARANCE = 0.002                     # m (the planner's soft clearance, rrt.SOFT_CLEARANCE; guess, unverified)
HAND_PARTS = ("gripper", "moving_jaw", "finray", "pad_", "wrist_camera", "blade")


ALIGN_LOG = collections.Counter()   # V6_ALIGN: how each carry was found or what blocked the crane move
_ALIGN_KIN = None            # TrussTeacher._align_kin's result, once per process


def _rotz(a):
    """Rotation by a (rad) about the vertical."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# A/B without editing code: BLUSH_V6='{"V6_LIVE_RELEASE": false, "V6_FIT": "rigid", "V6_CARRY": ["rigid", 0]}' sets
# these switches (and truss_expert.V6_CARRY) for one process (base_grid.py, fruit_off_trace.py runs).
if os.environ.get("BLUSH_V6"):
    for _k, _v in json.loads(os.environ["BLUSH_V6"]).items():
        if _k == "V6_CARRY":
            tx.V6_CARRY = tuple(_v)
        elif _k == "V6_AIM_FROM_PAD_M":       # the aim point in the v6 jaw lives in truss_ik
            truss_ik.V6_AIM_FROM_PAD_M = float(_v)
        elif _k in globals() and _k.startswith("V6_"):
            globals()[_k] = _v
        else:
            raise KeyError(f"BLUSH_V6: unknown switch {_k}")


# What blocked a plan, counted per (reason, what was hit) for base_grid --plan-only's census (2026-11-25): which geoms
# a "slide hits" / "the truss in hand hits" / blocked route runs into. The caller clears it per scene.
HIT_LOG = collections.Counter()
POOL_LOG = []        # every v6 release fit of a plan search: (poses that fit, of them under V6_PAD_SWING_MAX_DEG)


def hit_kind(name, target):
    """A blocked check's geom name -> a kind (the census groups by it)."""
    if name.startswith("self:"):
        return "the arm itself"
    if name.startswith("blade on"):
        return "blade near a stalk"
    m = re.match(r"(s\d+t\d+)_(t\d+)?", name)
    if m:
        own = m.group(1) == target
        return ("own " if own else "neighbour ") + ("fruit" if m.group(2) else "truss stalk")
    for key, kind in (("crate", "crate"), ("table", "table"), ("floor", "table"), ("stem", "stem"),
                      ("knuckle", "stem"), ("leaf", "leaf")):
        if key in name:
            return kind
    return "other: " + name


def _log_hits(reason, chk, target, where=False):
    for n in getattr(chk, "last_hits", ()) or ():
        HIT_LOG[(reason, hit_kind(n, target))] += 1
    if where:                                     # which segment of the path is blocked (-1: its first pose)
        HIT_LOG[(reason + " at segment", str(getattr(chk, "last_seg", None)))] += 1


def add_self_check(chk, prefix="arm/"):
    """Make chk.hits (and so free / segment_free / path_ok) also report self-collisions of the arm. Returns chk.
    ⚠️ The wrapper holds chk only by a weak reference. Holding it strongly (the bound chk.hits, or chk itself in the
    closure) made chk -> chk.hits -> chk a reference cycle, which only Python's rare full gc pass frees, so checkers
    (a deepcopy of the scene's model + MjData, 64 MB each) outlived their teacher: in eval_truss --hybrid --record one
    more stayed alive per planned scene (1 -> 5 over 5 scenes until a full gc) and a failed plan left 20-28 alive at
    once (2026-11-16)."""
    m = chk.m
    body = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
    arm = np.array([b.startswith(prefix) for b in body])
    hand = np.array([any(h in b for h in HAND_PARTS) for b in body])
    orig = type(chk).hits                         # unbound: rrt.Checker.hits or truss_expert.CarryChecker.hits
    ref = weakref.ref(chk)

    def hits(q, **kw):
        chk = ref()
        out = orig(chk, q, **kw)                  # leaves chk.d at q, contacts current
        for c in chk.d.contact[:chk.d.ncon]:
            a, b = c.geom1, c.geom2
            if (arm[a] and arm[b] and m.geom_bodyid[a] != m.geom_bodyid[b] and not (hand[a] and hand[b])
                    and c.dist < SELF_CLEARANCE):
                out.add(f"self: {body[a][len(prefix):]} / {body[b][len(prefix):]}")
        return out
    chk.hits = hits
    return chk


class Leg:
    """One piece of the motion: a joint path (norm units, (n, 5)) or a hold, the jaw command, the pace."""

    def __init__(self, name, W=None, hold=None, jaw=None, jaw_to=None, lead=FREE_LEAD, servo=False, frames=None,
                 checker_ok=True, until=None):
        self.name, self.lead, self.servo, self.frames = name, lead, servo, frames
        self.until = until                          # a hold leg: also done once until() is true
        self.W = None if W is None else np.asarray(W, float)
        self.hold = None if hold is None else np.asarray(hold, float)
        self.jaw, self.jaw_to = jaw, jaw_to         # jaw: fixed command; jaw_to: (from, to) ramped over the leg
        if self.W is not None:
            seg = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
            self.cum = np.r_[0.0, np.cumsum(seg)]
            self.L = float(self.cum[-1])

    def point(self, s):
        s = float(np.clip(s, 0.0, self.L))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.W) - 2))
        seg = self.cum[i + 1] - self.cum[i]
        u = 0.0 if seg < 1e-12 else (s - self.cum[i]) / seg
        return self.W[i] + (self.W[i + 1] - self.W[i]) * u

    def trim_to(self, q):
        """Start the path at q, joined to its nearest point (v6: the lift after the lowering stopped early would
        otherwise first drive back down to the planned bottom, onto the truss it just dropped)."""
        s0 = self.project(q, 0.0, self.L)
        keep = self.W[self.cum > s0 + 1e-9]
        self.W = np.vstack([q, keep]) if len(keep) else np.vstack([q, self.W[-1]])
        seg = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
        self.cum = np.r_[0.0, np.cumsum(seg)]
        self.L = float(self.cum[-1])

    def project(self, q, s_lo, s_hi):
        """Arc length of the point on the path nearest q, searched in [s_lo, s_hi]."""
        best, best_s = np.inf, s_lo
        for i in range(len(self.W) - 1):
            a, b = self.W[i], self.W[i + 1]
            c0, c1 = self.cum[i], self.cum[i + 1]
            if c1 < s_lo or c0 > s_hi:
                continue
            ab = b - a
            n2 = float(ab @ ab)
            u = 0.0 if n2 < 1e-12 else float(np.clip((q - a) @ ab / n2, 0, 1))
            s = float(np.clip(c0 + (c1 - c0) * u, s_lo, s_hi))
            dist = np.linalg.norm(q - self.point(s))
            if dist < best:
                best, best_s = dist, s
        return best_s


class TrussTeacher(tx.TrussPlanner):
    """TrussPlanner's scene sets, checker and IK; deterministic legs instead of its sampled plan."""

    def __init__(self, m, d, info, k):
        # v5: planning reads a frozen copy of the start frame, so the search can go on after a dry run has moved the
        # live scene (next_plan). The live data stays at hand for rejoin (the jaw where it is at a hand-over).
        self.d_live = d
        super().__init__(m, copy.copy(d), info, k)
        add_self_check(self.chk)
        self.legs, self.phases, self.meta = [], [], {}
        self._align, self._q_cut = {}, None
        self._swing = {}                       # release pose (joints) -> gravity's swing in the pads' plane (deg)
        self._R_cut = None                     # the gripper at the cut (planned), for that swing
        self._pool = None                      # the last v6 fit: (poses that fit, of them under V6_PAD_SWING_MAX_DEG)
        self.q_home = np.asarray(scene.START_NORM, float)
        self.v6 = scene.jaw_of(m) == "v6" and V6_TEACHER      # the v6 world's carry, settle and lift (V6_*)
        self._vias = []
        self._search, self._skip, self.n_plans, self._avoid = None, True, 0, set()
        self._legs_c, self._std_c = {}, {}     # (mode, u, beta[, tilt]) -> cut leg / route to the standoff
        self._more = False                     # the last pass with VIA_NORM_V5B's poses (_search_plans)
        self._xcut_c, self._geom_c = {}, {}    # (u, tilt) -> the cut pose; (u, beta, tilt, fast) -> its IK
        self._std_blocked, self._c_count = {}, 0   # phase C: standoffs no route reached, per mode; its candidates

    def close(self):
        """End the search (next_plan can't go on after this). The suspended generator holds the teacher in a reference
        cycle: without this (or a gc.collect()) a planned teacher and its checkers outlive the scene."""
        if self._search is not None:
            self._search.close()
            self._search = None

    def _mode(self, mode):
        self.kw = dict(soft_clearance=MODES[mode][0], firm_clearance=MODES[mode][1])

    # --- planning (at the start frame; next_plan goes on from there) ---
    def plan(self):
        """The first plan in the fixed order: v4's own where v4 finds one. PlanError if none."""
        self._search = self._search_plans()
        self.n_plans = 0
        return self.next_plan()

    def next_plan(self, same_candidate=False, avoid_approach=False):
        """The next plan in the same fixed order, after one whose dry run failed (generate_truss --retries).
        same_candidate: it failed from the carry on (outside the crate, a fruit off or a neighbour moved carrying,
        lowering or dropping it), so the same cut and approach with the next release pose first; else the next
        candidate. avoid_approach: it failed on the way in or out, so skip every later candidate with the same approach
        angle and stalk tilt, in any neighbour mode (they differ only in the cut point, 1-2 mm apart, or reuse the same
        route). PlanError when none is left."""
        if self._search is None:
            raise PlanError("plan() first")
        if avoid_approach and self.meta:
            m = self.meta
            self._avoid.add((float(m["approach_blend_deg"]), float(m.get("tilt_deg", 0.0))))
        self._skip = not same_candidate
        try:
            next(self._search)
        except StopIteration:
            raise PlanError(f"no plan left after {self.n_plans}") from None
        self.n_plans += 1
        self.meta["plan_index"] = self.n_plans
        return self

    def _search_plans(self):
        """Generator: sets the legs + meta of the next plan, then yields. Phase A = v4's first pass (templates, the
        release fit's first pose), B = v4's release fallback (alternates for the candidates that failed at the release
        or the carry), C = v5 (CUT_U_V5, V5_RETREATS, RELEASE_ALTERNATES_V5, then the push mode)."""
        self.q0 = q0 = self.q_now()
        why, late, done = [], {}, set()        # late: v4's release fallback; done: candidates that gave a plan
        for mode in tx.NEIGHBOUR_CLEAR:
            self._mode(mode)
            self.jaw(tx.JAW_OPEN_DEG)
            if not self.chk.free(q0, **self.kw):
                why.append(f"{mode}: start pose not clear")
                continue
            tried = []
            for u in CUT_U:
                for beta in APPROACH_BLEND_DEG:
                    yield from self._candidate("A", mode, u, beta, tried, late, done)
            why.append(f"{mode}: " + ("; ".join(sorted(set(tried))) or "no candidate"))
        todo = [key for key in late if key not in done]
        if todo:                               # v4's release fallback, in v4's order
            why2 = []
            for mode in [m for m in tx.NEIGHBOUR_CLEAR if any(key[0] == m for key in todo)]:
                self._mode(mode)
                tried = []
                for u in CUT_U:
                    for beta in APPROACH_BLEND_DEG:
                        if (mode, u, beta) in todo:
                            yield from self._candidate("B", mode, u, beta, tried, late, done)
                why2.append(f"{mode}: " + ("; ".join(sorted(set(tried))) or "no candidate"))
            why.append("release fallback: " + " / ".join(why2))
        tried3, clear = {m: [] for m in MODES}, {}
        for mode in MODES:                     # v5: strict, brush, then push
            self._mode(mode)
            self.jaw(tx.JAW_OPEN_DEG)
            clear[mode] = self.chk.free(q0, **self.kw)
        order = [(mode, tilt) for mode in ("strict", "brush") for tilt in TILT_DEG]
        order = sorted(order, key=lambda mt: TILT_DEG.index(mt[1])) + [("push", tilt) for tilt in TILT_DEG]
        yield from self._phase_c(order, clear, tried3, late, done)
        why3 = [f"{m}: start pose not clear" if not clear[m] else
                f"{m}: " + ("; ".join(sorted(set(tried3[m]))) or "no candidate") for m in MODES]
        if self._c_count >= PHASE_C_MAX:
            why3.append(f"budget: {PHASE_C_MAX} candidates")
        why.append("v5: " + " / ".join(why3))
        if VIA_NORM_V5B:                       # the last pass: phase C's candidates that gave no plan, once more with
            self._more = True                  # round 3's via-poses on the way in and on the carry (free_move)
            tried4 = {m: [] for m in MODES}
            yield from self._phase_c(order, clear, tried4, late, done)
            self._more = False
            why.append("v5b: " + " / ".join(f"{m}: " + ("; ".join(sorted(set(tried4[m]))) or "no candidate")
                                             for m in MODES if clear[m]))
        if RRT_D:                              # what-if phase D (RRT_D), off by default
            tried_d, n_d = [], 0
            for u in (0.5,):
                for mode, tilt in order:
                    if not clear[mode] or n_d >= RRT_D_MAX:
                        continue
                    self._mode(mode)
                    for beta in APPROACH_BLEND_DEG:
                        if n_d >= RRT_D_MAX:
                            break
                        leg = self._cut_leg(u, beta, self.t["r"], tilt, fast=True)
                        if isinstance(leg, str) or not leg["approaches"]:
                            continue
                        n_d += 1
                        leg = dict(leg)
                        xs, so = leg["approaches"][-1]
                        self.jaw(tx.JAW_OPEN_DEG)
                        import rrt
                        path = rrt.plan(self.chk, q0, xs[0], np.random.default_rng([self._seed_d(), n_d]),
                                        lo=self.lo, hi=self.hi, iters=RRT_D_ITERS, **self.kw)
                        if path is None:
                            tried_d.append("no RRT route")
                            continue
                        leg["approach"], leg["standoff_m"] = xs, so
                        self._at_release = False
                        for rest in self._rests("C", leg, self.t["r"], tried_d):
                            self._parts = dict(leg=leg, rest=rest)
                            self._build_legs(q0, path, leg, rest)
                            self.meta = dict(teacher="v5", phase="D_rrt", neighbours=mode, cut_u=u,
                                             approach_blend_deg=float(beta), tilt_deg=float(tilt),
                                             slide_mm=round(leg["slide_m"] * 1000),
                                             standoff_mm=round(leg["standoff_m"] * 1000),
                                             detours=dict(to_standoff=f"rrt:{len(path) - 2}",
                                                          to_crate=rest["detour_crate"], home=rest["detour_home"]),
                                             retreat_option=rest["retreat_option"],
                                             release_point=rest["release_point"], release_pitch=rest["release_pitch"],
                                             crate_slack_mm=rest["crate_slack_mm"], target=self.t["name"],
                                             n_fruit=self.t["n"], stalk_mm=round(self.t["r"] * 2000, 1),
                                             tried=sorted(set(tried_d)), miss=False,
                                             release_fallback=rest["release_fallback"])
                            yield
                            if self._skip:
                                break
            why.append("rrt what-if: " + ("; ".join(sorted(set(tried_d))) or "no candidate"))
        raise PlanError(" / ".join(why))

    def _phase_c(self, order, clear, tried3, late, done):
        """Generator over phase C's candidates that gave no plan yet, at most PHASE_C_MAX past their cut leg."""
        self._c_count = 0
        for u in CUT_U_V5:
            for mode, tilt in order:
                if not clear[mode]:
                    continue
                self._mode(mode)
                for beta in APPROACH_BLEND_DEG:
                    if ((mode, u, beta) if tilt == 0 else (mode, u, beta, tilt)) not in done:
                        yield from self._candidate("C", mode, u, beta, tried3[mode], late, done, tilt)
                    if self._c_count >= PHASE_C_MAX:
                        return

    def _seed_d(self):
        """The scene's seed for phase D's RRT (deterministic per scene)."""
        return int(self.info.get("seed", 0)) % (2 ** 31)

    def _candidate(self, phase, mode, u, beta, tried, late, done, tilt=0.0):
        """Generator over one candidate's plans (cut point u, approach beta, neighbour mode, stalk tilt): the first,
        then (when next_plan asks for the same candidate) its other release poses. The cut leg and the route to the
        standoff are computed once per candidate and reused by the later phases."""
        r, key = self.t["r"], (mode, u, beta) if tilt == 0 else (mode, u, beta, tilt)
        if (float(beta), float(tilt)) in self._avoid:
            tried.append("approach failed its dry run")
            return
        if phase == "B":
            leg, to_std, detour = late[key]
        else:
            # phase C's own legs (fast IK, STANDOFF_M_V5): reusing phase A's for the square cut at v4's cut points meant
            # the longer standoffs were never tried for them (review, 2026-11-18)
            ckey = key if phase != "C" else (*key, "C")
            if ckey not in self._legs_c:
                self._legs_c[ckey] = self._cut_leg(u, beta, r, tilt, fast=phase == "C")
            leg = self._legs_c[ckey]
            if isinstance(leg, str):
                tried.append(leg)
                return
            if phase == "C":
                self._c_count += 1             # PHASE_C_MAX
            if ckey not in self._std_c:
                self._std_c[ckey] = self._to_standoff(self.q0, leg)   # sets leg's approach + standoff_m
            to_std, detour = self._std_c[ckey]
            if to_std is None and phase == "C" and VIA_NORM_V5:
                leg = dict(leg)                # the cached one keeps v4's route (none)
                to_std, detour = self._to_standoff(self.q0, leg, extra=True,
                                                   memo=(mode, "v5b") if self._more else mode)
            if to_std is None:
                _log_hits("start -> standoff blocked", self.chk, self.t["name"])
                tried.append("start -> standoff blocked, detour too")
                return
        self._at_release = False
        for rest in self._rests(phase, leg, r, tried):
            self._parts = dict(leg=leg, rest=rest)   # for rejoin: the cut and after, on the start-frame scene
            self._build_legs(self.q0, to_std, leg, rest)
            self.meta = dict(teacher="v5", phase=phase, neighbours=mode, cut_u=u, approach_blend_deg=float(beta),
                             tilt_deg=float(tilt),
                             slide_mm=round(leg["slide_m"] * 1000), standoff_mm=round(leg["standoff_m"] * 1000),
                             detours=dict(to_standoff=detour, to_crate=rest["detour_crate"],
                                          home=rest["detour_home"]),
                             retreat_option=rest["retreat_option"], release_point=rest["release_point"],
                             release_pitch=rest["release_pitch"], crate_slack_mm=rest["crate_slack_mm"],
                             release_swing_deg=None if rest.get("release_swing") is None
                             else round(rest["release_swing"], 1), release_pool=rest.get("release_pool"),
                             carry_align_deg=rest.get("carry_align"),
                             target=self.t["name"], n_fruit=self.t["n"], stalk_mm=round(r * 2000, 1),
                             tried=sorted(set(tried)), miss=False)
            if phase != "A" or rest["release_fallback"] is not None:
                self.meta["release_fallback"] = rest["release_fallback"]
            done.add(key)
            yield
            if self._skip:
                return
        if phase == "A" and self._at_release and key not in done:
            late[key] = (leg, to_std, detour)

    def _line(self, p0, p1, s, r, x0, n, tilt=0.0, fast=False):
        """Joint poses along a straight line of the cut point, the stalk across the jaw (tilted tilt deg, v5).
        fast=False: TrussPlanner.line_ik (v4's, phases A and B); fast=True (phase C): each point continues the last
        one's arm branch or the line fails (truss_ik.solve_from)."""
        if not fast:
            return self.line_ik(p0, p1, s, r, x0, n)
        xs, x = [], x0
        for u in np.linspace(0, 1, n):
            x = self.ik.solve_from(p0 + (p1 - p0) * u, s, r, x, tilt_deg=tilt)
            if x is None:
                return None
            xs.append(x)
        return xs

    def _x_cut(self, u, tilt, fast, p_cut, s, r):
        """The cut pose at u (cached per u and tilt: the same for every mode and approach). Phase C (fast) solves u 0.5
        from the IK's samples and continues the other points of the cut zone from it."""
        if (u, tilt) not in self._xcut_c:
            x = None
            if fast and u != 0.5:
                x0 = self._x_cut(0.5, tilt, fast, *ts.cut_target_now(self.t, self.d, self.m, 0.5), r)
                x = None if x0 is None else self.ik.solve_from(p_cut, s, r, x0, tilt_deg=tilt)
            else:
                x = self.ik.solve(p_cut, s, r, tilt_deg=tilt)
            self._xcut_c[(u, tilt)] = x
        return self._xcut_c[(u, tilt)]

    def _cut_geom(self, u, beta, r, tilt, fast):
        """The cut leg's IK, cached per (u, beta, tilt, fast) for every neighbour mode: the cut pose, the approach axis,
        the slide (SLIDE_M, the first that has IK), and lazily the approach lines behind it (STANDOFF_M)."""
        key = (u, beta, tilt, fast)
        if key not in self._geom_c:
            p_cut, s = ts.cut_target_now(self.t, self.d, self.m, u)
            x_cut = self._x_cut(u, tilt, fast, p_cut, s, r)
            if x_cut is None:
                self._geom_c[key] = "no cut pose"
                return self._geom_c[key]
            _, R = self.ik.frame(x_cut, r)
            b = np.radians(beta)
            f = np.cos(b) * -R[:, 2] + np.sin(b) * R[:, 1]
            # V6_MEET_FROM_M: the slide runs with the jaw off sideways (gripper -x: the fixed pad further from the
            # stalk) and its last way-points bring the pad to the stalk
            off = np.zeros(3)
            if truss_ik.JAW == "v6" and V6_MEET_FROM_M is not None:      # the jaw's, also with V6_TEACHER off
                off = -(V6_MEET_FROM_M - truss_ik.V6_AIM_FROM_PAD_M) * R[:, 0]
            slide = None
            for L in SLIDE_M:
                slide = self._line(p_cut + off - f * L, p_cut + off, s, r, x_cut, tx.SLIDE_STEPS, tilt, fast)
                if slide is not None and off.any():
                    meet = self._line(p_cut + off, p_cut, s, r, slide[-1], V6_MEET_STEPS, tilt, fast)
                    slide = None if meet is None else slide + meet[1:-1] + [x_cut]
                if slide is not None:
                    break
            self._geom_c[key] = "no IK along the slide" if slide is None else dict(
                x_cut=x_cut, p_cut=p_cut, s=s, f=f, slide=slide, slide_m=L, off=off, u=u, beta=beta, tilt=tilt,
                lines={})
        return self._geom_c[key]

    def _cut_leg(self, u, beta, r, tilt=0.0, fast=False):
        """Cut pose at u along the cut zone (the stalk tilted tilt deg in the jaw, v5), the slide in along the approach
        blended beta deg toward the stalk, and the standoff behind it. Same checks as TrussPlanner._cut_leg, plus the
        approach to the pre-cut pose. The IK is _cut_geom's; the checks use this mode's clearances."""
        g = self._cut_geom(u, beta, r, tilt, fast)
        if isinstance(g, str):
            return g
        p_cut, s, f, slide, L = g["p_cut"], g["s"], g["f"], g["slide"], g["slide_m"]
        self.jaw(tx.JAW_OPEN_DEG)
        if not self.chk.path_ok(slide, ignore=self.target_stalk, **self.kw):
            _log_hits("slide hits", self.chk, self.t["name"])
            return "slide hits"
        for deg in np.linspace(tx.JAW_OPEN_DEG, -10.0, 6):
            self.jaw(deg)
            if self.chk.hits(slide[-1], ignore=self.target_stalk, **self.kw):
                self.jaw(tx.JAW_OPEN_DEG)
                return "closing jaw hits a neighbour"
        self.jaw(tx.JAW_OPEN_DEG)
        approaches = []                       # every standoff that works, in STANDOFF_M (phase C: _V5) order
        for so in (STANDOFF_M_V5 if fast else STANDOFF_M):
            if so == 0.0:
                approaches.append(([slide[0]], 0.0))
                continue
            if so not in g["lines"]:
                o = g["off"]
                g["lines"][so] = self._line(p_cut + o - f * (L + so), p_cut + o - f * L, s, r, slide[0],
                                            APPROACH_STEPS, tilt, fast)
            xs = g["lines"][so]
            if xs is not None and self.chk.path_ok(xs, **self.kw):
                approaches.append((xs, so))
        return dict(x_cut=g["x_cut"], p_cut=p_cut, s=s, f=f, slide=slide, slide_m=L, approaches=approaches, u=u,
                    beta=beta, tilt=tilt, fast=fast, meet_m=float(np.linalg.norm(g["off"])))

    def vias(self):
        """The fixed via-poses, 5 joints in rad (VIA_NORM)."""
        if not self._vias:
            self._vias = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in VIA_NORM]
        return self._vias

    def vias_v5(self):
        """VIA_NORM_V5's poses, 5 joints in rad."""
        if not hasattr(self, "_vias5"):
            self._vias5 = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in VIA_NORM_V5]
        return self._vias5

    def vias_v5b(self):
        """VIA_NORM_V5B's poses, 5 joints in rad."""
        if not hasattr(self, "_vias5b"):
            self._vias5b = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in VIA_NORM_V5B]
        return self._vias5b

    def _cut_pitch_tilt(self):
        """The cut pose's finger pitch and jaw tilt (deg, as ArmIK.solve takes them) = gravity in the gripper frame."""
        Rc = self._R_cut
        return (float(np.degrees(np.arcsin(np.clip(Rc[2, 2], -1, 1)))),
                float(np.degrees(np.arcsin(np.clip(Rc[2, 0], -1, 1)))))

    def _is_aligned(self, x):
        """Gravity in x's gripper frame as at the cut, AND on the cut pose's branch. ⚠️ Two arm shapes give the same
        gravity: (pitch, roll) and (180 - pitch, roll + 180), the arm bent over backwards with the wrist flipped. A
        straight joint move between the two turned the gripper 105-116 deg on the way (9730016 / 019 / 027,
        2026-11-25); within one branch the wrist roll is constant and a straight joint move keeps gravity fixed."""
        tol = np.radians(V6_ALIGN_TOL_DEG)
        roll = (float(x[4]) - float(self._q_cut[4]) + np.pi) % (2 * np.pi) - np.pi
        return bool(self.ik_free.fk(x)[1][2] @ self._R_cut[2] >= np.cos(tol)) and abs(roll) <= 2 * tol

    def _align_solve(self, p, point, x0):
        """ArmIK's solve for grip point p at the cut's pitch and tilt, from x0 ONLY. ⚠️ ArmIK.solve restarts a miss
        from 8 sampled poses: that jumps branch (_is_aligned) and made one plan take > 9 min (2026-11-25)."""
        ikf = self.ik_free
        pitch, tilt = self._cut_pitch_tilt()
        x, e = ikf._lm(np.asarray(x0, float).copy(), np.asarray(p, float), point, -np.sin(np.radians(pitch)),
                       np.sin(np.radians(tilt)), V6_ALIGN_IK_ITERS)
        if np.linalg.norm(e[:3]) < 0.002 and np.all(np.abs(e[3:]) < ikf.W_ANG * 0.05) and self._is_aligned(x):
            return x
        return None

    def _align_kin(self):
        """How this arm keeps gravity fixed in the gripper frame, found once per process from its own kinematics:
        (s1, s2) with s1 * lift + s2 * elbow + wrist_flex = constant (the three pitch joints are parallel), the pan
        joint's sense about the vertical and where its axis stands (x, y)."""
        global _ALIGN_KIN
        if _ALIGN_KIN is None:
            ikf = self.ik_free
            q = 0.5 * (ikf.lo + ikf.hi)
            o0, R0 = ikf.fk(q)
            best = None
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    q2 = q.copy()
                    q2[1], q2[2], q2[3] = q[1] + 0.3, q[2] - 0.2, q[3] - s1 * 0.3 + s2 * 0.2
                    err = float(np.linalg.norm(ikf.fk(q2)[1][2] - R0[2]))
                    best = min(best, (err, s1, s2)) if best else (err, s1, s2)
            if best[0] > 1e-6:
                raise tx.PlanError("V6_ALIGN: the arm's pitch joints are not parallel")
            qb = q.copy()
            qb[0] += 0.4
            ob, Rb = ikf.fk(qb)
            sgn = next(sg for sg in (1.0, -1.0) if np.allclose(Rb, _rotz(sg * 0.4) @ R0, atol=1e-6))
            Rz = _rotz(sgn * 0.4)[:2, :2]
            c = np.linalg.solve(np.eye(2) - Rz, ob[:2] - Rz @ o0[:2])
            _ALIGN_KIN = (best[1], best[2], sgn, c)
        return _ALIGN_KIN

    def _aligned(self, q, point=None):
        """V6_ALIGN: pose q put on the cut pose's pitch and tilt: its pan, lift and elbow kept, the wrist bend set to
        compensate them and the wrist roll as at the cut. None if the wrist can't bend that far."""
        s1, s2, _, _ = self._align_kin()
        q, qc = np.asarray(q, float), self._q_cut
        x = np.array([q[0], q[1], q[2], qc[3] + s1 * (qc[1] - q[1]) + s2 * (qc[2] - q[2]), qc[4]])
        ikf = self.ik_free
        return x if ikf.lo[3] <= x[3] <= ikf.hi[3] and self._is_aligned(x) else None

    def _align_nodes(self, point):
        """Every aligned arm shape at the cut's pan (lift x elbow on the V6_ALIGN_STEP_DEG grid, _aligned): joints X,
        gripper origins O, rotations R, and the grip point's distance RHO from the pan axis. Cached per cut pose."""
        key = ("nodes", tuple(np.round(self._R_cut, 5).ravel()))
        if self._align.get("nkey") != key:
            ikf, qc, step = self.ik_free, self._q_cut, np.radians(V6_ALIGN_STEP_DEG)
            c = self._align_kin()[3]
            X, O, R, RHO = [], [], [], []
            for lift in np.arange(ikf.lo[1], ikf.hi[1] + 1e-9, step):
                for elbow in np.arange(ikf.lo[2], ikf.hi[2] + 1e-9, step):
                    x = self._aligned(np.array([qc[0], lift, elbow, 0.0, 0.0]))
                    if x is None:
                        continue
                    o, Rm = ikf.fk(x)
                    X.append(x), O.append(o), R.append(Rm)
                    RHO.append(float(np.linalg.norm((o + Rm @ point)[:2] - c)))
            self._align["nkey"] = key
            self._align["nodes"] = (np.array(X).reshape(-1, 5), np.array(O).reshape(-1, 3),
                                    np.array(R).reshape(-1, 3, 3), np.array(RHO))
        return self._align["nodes"]

    def _tuck_via(self, pan, rho, o_z, point):
        """The aligned arm shape whose grip point is nearest `rho` from the pan axis with the gripper origin nearest
        height o_z, at pan angle `pan`; None when the grid has nothing within V6_ALIGN_VIA_TOL_M."""
        X, O, _, RHO = self._align_nodes(point)
        if not len(X):
            return None
        d = np.hypot(RHO - rho, O[:, 2] - o_z)
        i = int(d.argmin())
        return np.r_[pan, X[i, 1:]] if d[i] <= V6_ALIGN_VIA_TOL_M else None

    def _seg_ok(self, chk, a, b, kw):
        """chk.path_ok([a, b]) remembered ON the checker: the crane move's pull-out and turn are the same for many
        release poses (re-checking them made one plan take > 9 min, 2026-11-25). ⚠️ On the checker, not keyed by
        id(chk): each cut candidate has its own CarryChecker (its own carried truss), and ids are reused."""
        memo = chk.__dict__.setdefault("_seg_memo", {})
        key = (np.asarray(a, float).round(5).tobytes(), np.asarray(b, float).round(5).tobytes(), repr(kw))
        if key not in memo:
            memo[key] = bool(chk.path_ok([a, b], **kw))
        return memo[key]

    def _move_point_aligned(self, q, delta, n, point):
        """As _move_point (the grip point moved by delta in n steps), at the cut pose's pitch and tilt all the way."""
        o, R = self.ik_free.fk(q)
        p0, x, xs = o + R @ point, np.asarray(q, float), []
        for u in np.linspace(0, 1, n + 1)[1:]:
            x = self._align_solve(p0 + np.asarray(delta) * u, point, x)
            if x is None:
                return None
            xs.append(x)
        return xs

    def free_move(self, xa, xb, chk, r, extra=False, align=False, **kw):
        """The template (one straight joint line xa -> xb), else the detour over a fixed via-pose (VIA_NORM, nearest
        xb first), else two of them chained. Returns (way-points, 0 for the line, the via's number from 1, or
        10 * first + second for a chain) or (None, None). extra (v5: phase C, rejoin): then VIA_NORM_V5's poses alone
        (nearest xb first), then chained with any other via ("v5:<i>", "v5:<i>-<j>", indices after VIA_NORM's)."""
        if chk.path_ok([xa, xb], **kw):
            if align:
                ALIGN_LOG["line: clear"] += 1
            return [xa, xb], 0
        vias = self.vias()
        dist = lambda u, v: float(np.abs(u - v).max())   # noqa: E731
        if align:                  # V6_ALIGN: first the same via-poses moved onto the cut's pitch and tilt ("al:")
            # the crane move: pull the hanging truss straight back out of the row and up until its lowest point
            # clears the crate rim, turn, reach out over the crate. Fixed radii and heights, tried in this order.
            point = truss_ik.cut_point_in_gripper(r)
            low = self._align.get("low")
            if low is not None:
                z_a = float(self.ik_free.fk(xa)[0][2])              # the gripper's height where the retreat ended
                for over in V6_ALIGN_OVER_RIM_M:
                    for rho in V6_ALIGN_TUCK_M:
                        o_z = ts.CRATE_SIZE[2] + over - low
                        va, vb = self._tuck_via(xa[0], rho, o_z, point), self._tuck_via(xb[0], rho, o_z, point)
                        v0 = self._tuck_via(xa[0], rho, z_a, point)     # straight back first, then up or down
                        if va is None or vb is None:
                            ALIGN_LOG["crane: no pose at that radius and height"] += 1
                            continue
                        out = None
                        if self._seg_ok(chk, xa, va, kw):
                            out = [xa, va]
                        elif v0 is not None and self._seg_ok(chk, xa, v0, kw) and self._seg_ok(chk, v0, va, kw):
                            out = [xa, v0, va]
                        if out is None:
                            ALIGN_LOG["crane: pulling out of the row blocked"] += 1
                        elif not self._seg_ok(chk, va, vb, kw):
                            ALIGN_LOG["crane: turning blocked"] += 1
                        elif not self._seg_ok(chk, vb, xb, kw):
                            ALIGN_LOG["crane: reaching out over the crate blocked"] += 1
                        else:
                            ALIGN_LOG["crane: clear"] += 1
                            return out + [vb, xb], f"crane:{rho}:{over}"
            if not V6_ALIGN_FALLBACK:
                return None, None
        for i in sorted(range(len(vias)), key=lambda j: dist(vias[j], xb)):
            if chk.path_ok([xa, vias[i], xb], **kw):
                return [xa, vias[i], xb], i + 1
        pairs = sorted(((i, j) for i in range(len(vias)) for j in range(len(vias)) if i != j),
                       key=lambda ij: dist(xa, vias[ij[0]]) + dist(vias[ij[0]], vias[ij[1]]) + dist(vias[ij[1]], xb))
        for i, j in pairs:
            if chk.path_ok([xa, vias[i], vias[j], xb], **kw):
                return [xa, vias[i], vias[j], xb], 10 * (i + 1) + j + 1
        if not extra or not VIA_NORM_V5:
            return None, None
        n4, allv = len(vias), vias + self.vias_v5()
        for i in sorted(range(n4, len(allv)), key=lambda j: dist(allv[j], xb)):
            if chk.path_ok([xa, allv[i], xb], **kw):
                return [xa, allv[i], xb], f"v5:{i}"
        pairs = sorted(((i, j) for i in range(len(allv)) for j in range(len(allv)) if i != j and max(i, j) >= n4),
                       key=lambda ij: dist(xa, allv[ij[0]]) + dist(allv[ij[0]], allv[ij[1]]) + dist(allv[ij[1]], xb))
        for i, j in pairs:
            if chk.path_ok([xa, allv[i], allv[j], xb], **kw):
                return [xa, allv[i], allv[j], xb], f"v5:{i}-{j}"
        if not (self._more and VIA_NORM_V5B):
            return None, None
        n7, allw = len(allv), allv + self.vias_v5b()          # the last pass only: round 3's poses, same two shapes
        for i in sorted(range(n7, len(allw)), key=lambda j: dist(allw[j], xb)):
            if chk.path_ok([xa, allw[i], xb], **kw):
                return [xa, allw[i], xb], f"v5b:{i}"
        pairs = sorted(((i, j) for i in range(len(allw)) for j in range(len(allw)) if i != j and max(i, j) >= n7),
                       key=lambda ij: dist(xa, allw[ij[0]]) + dist(allw[ij[0]], allw[ij[1]]) + dist(allw[ij[1]], xb))
        for i, j in pairs:
            if chk.path_ok([xa, allw[i], allw[j], xb], **kw):
                return [xa, allw[i], allw[j], xb], f"v5b:{i}-{j}"
        return None, None

    def _to_standoff(self, q0, leg, extra=False, memo=None):
        """The first standoff (STANDOFF_M order) the arm can get to from the start; sets leg approach/standoff_m.
        extra: v5's via-poses too (free_move). memo (phase C, the mode): skip a standoff within STD_MEMO_RAD of one no
        route reached in this mode, and remember the ones none reaches."""
        self.jaw(tx.JAW_OPEN_DEG)
        blocked = self._std_blocked.setdefault(memo, []) if memo is not None else None
        for xs, so in leg["approaches"]:
            goal = np.asarray(xs[0], float)
            if blocked is not None and any(np.abs(goal - b).max() < STD_MEMO_RAD for b in blocked):
                continue
            path, how = self.free_move(q0, xs[0], self.chk, self.t["r"], extra=extra, **self.kw)
            if path is not None:
                leg["approach"], leg["standoff_m"] = xs, so
                return path, how
            if blocked is not None:
                blocked.append(goal)
        return None, None

    def _rests(self, phase, leg, r, tried):
        """Generator over the ways on from the cut (v4's _after_cut, one per yield): a retreat, a release pose, the
        carry onto it, the way home. Phase A: the release fit's first pose, and only when next_plan asks for the same
        candidate again its alternates (RELEASE_FALLBACK_N); B (v4's release fallback): the alternates; C: v4's
        retreats then V5_RETREATS, the first pose then RELEASE_ALTERNATES_V5 alternates. Why each failed goes to
        tried; self._at_release says it got past the retreat (v4's late list)."""
        x_cut = leg["x_cut"]
        carry = self._carry(x_cut)
        self._R_cut, self._q_cut = self.ik.fk(x_cut)[1], np.asarray(x_cut, float)
        point = truss_ik.cut_point_in_gripper(r)
        cchk = add_self_check(tx.CarryChecker(self.m, self.d, carry[0], carry[1], point, hard=self.hard,
                                              soft=self.soft))
        cchk.d.qpos[self.g_adr] = np.radians(-10.0)
        self.jaw(-10.0)
        ret, opt = self._retreat(leg, r, cchk, tried, extra=phase == "C")
        if ret is None:
            return
        self._at_release = True               # from here on it failed at the release or the carry
        seen, gave, first = [], False, None   # grip points of this candidate's poses tried in a plan
        n_alt = RELEASE_FALLBACK_N if phase in "AB" else RELEASE_ALTERNATES_V5
        if phase != "B":
            first = self._release(carry, cchk, r)
            if isinstance(first, str):
                tried.append(first)
                # ⚠️ only v4 (phase A) stops here (its alternates wait for phase B). Phase C stopping here too left the
                # relaxed poses dead for exactly their case, an 8-fruit truss whose few fitting poses all hit (review,
                # 2026-11-18: 9900044, 9930075)
                if phase == "A":
                    return
                first = None
        if first is not None:
            rest = self._via(first, ret, opt, cchk, r, None, extra=phase == "C")
            if isinstance(rest, str):
                tried.append(rest)
                if phase == "A":               # v4: the alternates wait for the release fallback (phase B)
                    return
            else:
                seen.append(first[3])
                gave = True
                yield rest
                if self._skip:
                    return
        why = "release fallback: no other clear pose"
        if not gave:                           # no plan from this candidate yet: v4's order, most room first
            for j, rel in enumerate(self._release_alternates(carry, cchk, r, n_alt)):
                rest = self._via(rel, ret, opt, cchk, r, dict(alternate=j + 1, rank=self._rel_rank),
                                 extra=phase == "C")
                if isinstance(rest, str):
                    why = rest
                    continue
                why, gave = None, True
                seen.append(rel[3])
                yield rest
                if self._skip:
                    return
                break                          # it failed in the physics: the others lowest first, below
        if gave:                               # a plan of this candidate failed at the release: lowest drop first
            for j, rel in enumerate(self._release_alternates(carry, cchk, r, n_alt, low=True, exclude=seen)):
                rest = self._via(rel, ret, opt, cchk, r, dict(alternate=j + 1, rank=self._rel_rank, low=True),
                                 extra=phase == "C")
                if isinstance(rest, str):
                    why = rest
                    continue
                why = None
                seen.append(rel[3])
                yield rest
                if self._skip:
                    return
        if phase == "C":                       # then poses the fit rejects (RELAXED_*), lowest first
            for j, rel in enumerate(self._release_alternates(carry, cchk, r, RELAXED_N, relaxed=True, low=True,
                                                             exclude=seen)):
                rest = self._via(rel, ret, opt, cchk, r, dict(alternate=j + 1, rank=self._rel_rank, relaxed=True),
                                 extra=True)
                if isinstance(rest, str):
                    why = rest
                    continue
                why = None
                seen.append(rel[3])
                yield rest
                if self._skip:
                    return
        if why is not None and phase != "A":
            tried.append(why)

    def _retreat(self, leg, r, cchk, tried, extra=False):
        """The retreat out of the plant with the truss in hand: v4's five (line IK holding the stalk's angle), then
        with extra (phase C) V5_RETREATS, position only. Returns (joint way-points, option) or (None, None)."""
        x_cut, p_cut, f, s = leg["x_cut"], leg["p_cut"], leg["f"], leg["s"]
        aisle = np.array([0, -np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]), 0.0])
        dist, up = RETREAT_M, RETREAT_UP_M
        options = (p_cut - f * dist + [0, 0, up], p_cut + aisle * dist + [0, 0, up], p_cut + [0, 0, dist],
                   p_cut + aisle * dist * 1.5, p_cut - f * dist * 0.6 + aisle * dist * 0.6 + [0, 0, up])
        why = "no IK for the retreat"
        for i, back in enumerate(options):
            xs = self._line(p_cut, back, s, r, x_cut, 8, leg.get("tilt", 0.0), leg.get("fast", False))
            if xs is None:
                continue
            if not cchk.path_ok(xs, **self.kw):
                why = "retreat: the truss in hand hits"
                _log_hits(why, cchk, self.t["name"], where=True)
                continue
            return xs, i
        if extra:
            for j, (b, a, z) in enumerate(V5_RETREATS):
                delta = -f * b + aisle * a + np.array([0.0, 0.0, z])
                xs = (self._move_point_aligned(x_cut, delta, 8, truss_ik.cut_point_in_gripper(r))
                      if self.v6 and V6_ALIGN else None) or self._move_point(x_cut, delta, 8)
                if xs is None:
                    continue
                path = [np.asarray(x_cut, float)] + xs
                if not cchk.path_ok(path, **self.kw):
                    why = "retreat: the truss in hand hits"
                    _log_hits(why, cchk, self.t["name"], where=True)
                    continue
                return path, f"v5_{j}"
        tried.append(why)
        return None, None

    def _via(self, rel, ret, opt, cchk, r, alternate, extra=False):
        """One release pose: the carry onto it (templates, CARRY_CLEAR then CARRY_CLEAR_MIN) and the way home. The rest
        of a plan as a dict, or why not."""
        x_rel, x_above, lower, p, pitch, slack = rel
        to_crate, detour_crate = None, None
        for clear in (tx.CARRY_CLEAR_M, tx.CARRY_CLEAR_MIN_M):   # the retreat can end < 2 cm from a neighbour
            to_crate, detour_crate = self.free_move(ret[-1], x_above, cchk, r, extra=extra,
                                                    align=self.v6 and V6_ALIGN, soft_clearance=clear)
            if to_crate is not None:
                break
        if to_crate is None:
            _log_hits("retreat -> crate blocked", cchk, self.t["name"])
            return "retreat -> crate blocked, via-pose too"
        self.jaw(tx.JAW_OPEN_DEG)
        q_start = self.fm.norm_to_rad(self.q_home)[:5]
        lift = list(reversed(lower)) if self.v6 and V6_LIFT_HOME and len(lower) > 1 else None
        home, detour_home = self.free_move((lift or lower)[-1], q_start, self.chk, r, extra=extra,
                                           ignore=self.target_all, firm=False, **self.kw)
        if home is None:
            return "crate -> start blocked, via-pose too"
        return dict(ret=ret, retreat_option=opt, to_crate=to_crate, lower=lower, home=home, lift=lift,
                    detour_crate=detour_crate, detour_home=detour_home, release_point=np.round(p, 4).tolist(),
                    release_pitch=round(pitch, 1), crate_slack_mm=round(slack * 1000, 1), release_fallback=alternate,
                    release_swing=self._swing.get(tuple(np.asarray(x_rel, float))), release_pool=self._pool,
                    carry_align=[self._align_dev(ret), self._align_dev(to_crate), self._align_dev(lower)])

    def _align_dev(self, path):
        """The most gravity turns in the gripper frame along joint way-points, against the cut pose (deg; mid-points
        too, the moves between way-points are straight in joint space). 0 = the truss has no reason to turn."""
        if self._R_cut is None or path is None or not len(path):
            return None
        qs = [np.asarray(q, float) for q in path]
        qs += [(a + b) / 2 for a, b in zip(qs[:-1], qs[1:])]
        return round(float(max(np.degrees(np.arccos(np.clip(self.ik.fk(q)[1][2] @ self._R_cut[2], -1, 1)))
                               for q in qs)), 1)

    def _release_fit(self, carry, r, slack_min=tx.RELEASE_SLACK_M, live=False):
        """Every sampled pose over the crate (tx.crate_poses) the carried truss fits at, most room to the walls first:
        (their indices, slack, Qs, Os, Rs, grip point in the gripper frame). slack_min: v4's, or RELAXED_SLACK_M."""
        (cx, cy), (sx, sy, h) = ts.CRATE_CENTRE, ts.CRATE_SIZE
        point = truss_ik.cut_point_in_gripper(r)
        radii = np.array([fr["radius"] for fr in self.t["fruit"]])
        inner = np.array([sx / 2 - scene.CRATE_WALL, sy / 2 - scene.CRATE_WALL])
        ikf = self.ik_free
        rel = np.array([q for (_, q) in carry[1][1:]])
        Rt, pt = carry[1][0]
        stalk = np.array([pt + Rt @ (q - self.t["knuckle"]) for q in
                          (self.t["knuckle"], self.t["peduncle_end"], self.t["rachis_end"])])
        rel = np.vstack([rel, stalk])
        radii = np.r_[radii, [r] * len(stalk)]
        aligned = self.v6 and V6_ALIGN and not live and self._R_cut is not None
        Qs, Os, Rs = self._aligned_poses(ikf, point, rel, radii) if aligned else tx.crate_poses(ikf, point)
        G = Os[:, None, :] + np.einsum("nij,kj->nki", Rs, rel)
        held = (inner[None, None, :] - (np.abs(G[:, :, :2] - [cx, cy]) + radii[None, :, None])).min((1, 2))
        flat = (inner[None, None, :] - (np.abs(tx.laid_flat(G) - [cx, cy]) + radii[None, :, None])).min((1, 2))
        slack = np.minimum(held, flat + tx.LEAN_M)
        low_z = (G[:, :, 2] - radii).min(1) - h
        high_z = (G[:, :, 2] + radii).max(1) - h
        if aligned:        # V6_ALIGN: the truss as cut IS how it hangs here; lowest first (2 cm bins), then most room
            ok = np.flatnonzero((slack >= slack_min) & (low_z >= scene.CRATE_WALL + V6_FLOOR_CLEAR_M - h)
                                & (low_z <= V6_DROP_MAX_M) & (high_z <= tx.ABOVE_RIM_MAX_M))
            self._pool = (len(Qs), len(ok))
            POOL_LOG.append(self._pool)
            self._swing.update({tuple(np.asarray(Qs[i], float)): 0.0 for i in ok})
            return ok[np.lexsort((-slack[ok], np.round(low_z[ok] / 0.02)))], slack, Qs, Os, Rs, point, low_z
        if not self.v6:
            ok = np.flatnonzero((slack >= slack_min) & (low_z >= scene.CRATE_WALL + 0.01 - h)
                                & (low_z <= tx.RELEASE_ABOVE_M[1] + 0.04) & (high_z <= tx.ABOVE_RIM_MAX_M))
            return ok[np.argsort(-slack[ok], kind="stable")], slack, Qs, Os, Rs, point, low_z
        if live:                                                           # the truss as it hangs now: held as is
            ok = np.flatnonzero((slack >= slack_min) & (low_z >= scene.CRATE_WALL + V6_FLOOR_CLEAR_M - h)
                                & (low_z <= V6_DROP_MAX_M) & (high_z <= tx.ABOVE_RIM_MAX_M))
            # the live shape is right, so centre the TRUSS (its fruit), not the grip or the modelled room: the most
            # "room" put 9730018's release 8 cm off centre and it settled over the -x wall (2026-11-24)
            n_fr = len(self.t["fruit"])
            fc = G[:, :n_fr, :2].mean(1)
            off = np.hypot(fc[:, 0] - cx, fc[:, 1] - cy)
            return ok[np.argsort(off[ok], kind="stable")], slack, Qs, Os, Rs, point, low_z
        if V6_FIT == "rigid":                                              # v5's fit, unchanged
            ok = np.flatnonzero((slack >= slack_min) & (low_z >= scene.CRATE_WALL + 0.01 - h)
                                & (low_z <= tx.RELEASE_ABOVE_M[1] + 0.04) & (high_z <= tx.ABOVE_RIM_MAX_M))
            return ok[np.argsort(-slack[ok], kind="stable")], slack, Qs, Os, Rs, point, low_z
        # v6: also as it hangs under its stalk (V6_FIT note)
        n_fr = len(self.t["fruit"])
        mass = np.array([fr["radius"] ** 3 for fr in self.t["fruit"]], float)    # ~ fruit mass (same density)
        P = Os + np.einsum("nij,j->ni", Rs, point)                         # grip points
        a = Rs[:, :, 1]                                                    # the stalk's axis per pose (tx.CARRY_AXIS)
        c = (G[:, :n_fr, :] * mass[None, :, None]).sum(1) / mass.sum() - P # centre of mass from the grip
        c_perp = c - (c * a).sum(1, keepdims=True) * a
        down = np.array([0.0, 0.0, -1.0])[None, :] + a[:, 2:3] * a
        nc, nd = np.linalg.norm(c_perp, axis=1), np.linalg.norm(down, axis=1)
        swing = (nc > 1e-6) & (nd > 1e-6)
        c_perp = c_perp / np.maximum(nc, 1e-9)[:, None]
        down = down / np.maximum(nd, 1e-9)[:, None]
        th_hang = np.arctan2((np.cross(c_perp, down) * a).sum(1), (c_perp * down).sum(1)) * swing
        v = G - P[:, None, :]                                              # Rodrigues about a through P
        ax = a[:, None, :]
        lean = tx.LEAN_M if V6_LEAN else 0.0
        if not V6_LEAN:
            slack = np.minimum(held, flat)
        if V6_FIT == "envelope":                                           # as cut .. hanging, then past it both ways
            over = np.radians(V6_SWING_OVER_DEG) * np.sign(th_hang)
            angles = [u * th_hang for u in np.linspace(0.0, 1.0, V6_SWING_STEPS)] + [th_hang - over, th_hang + over]
        else:                                                              # "hang": fully hanging only
            angles = [th_hang]
        low_h = np.full(len(G), np.inf)
        for th in angles:
            th = th * swing
            ct, st = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
            Gh = P[:, None, :] + v * ct + np.cross(ax, v) * st + ax * (ax * v).sum(2, keepdims=True) * (1 - ct)
            held_h = (inner[None, None, :] - (np.abs(Gh[:, :, :2] - [cx, cy]) + radii[None, :, None])).min((1, 2))
            flat_h = (inner[None, None, :]
                      - (np.abs(tx.laid_flat(Gh) - [cx, cy]) + radii[None, :, None])).min((1, 2))
            slack = np.minimum(slack, np.minimum(held_h, flat_h + lean))
            low_h = np.minimum(low_h, (Gh[:, :, 2] - radii).min(1) - h)
        low_all = np.minimum(low_z, low_h)
        # gravity in each pose's gripper frame (-row 2 of R) vs at the cut: its angle in the pads' plane (y-z), i.e. how
        # far the truss would swing about the closing axis to keep hanging (V6_PAD_SWING_MAX_DEG note)
        pad_swing = np.zeros(len(Rs))
        if self._R_cut is not None:
            g_cut = -self._R_cut[2, :]
            a_cut = np.arctan2(g_cut[2], g_cut[1])
            a_rel = np.arctan2(-Rs[:, 2, 2], -Rs[:, 2, 1])
            pad_swing = np.degrees(np.abs((a_rel - a_cut + np.pi) % (2 * np.pi) - np.pi))
        fits = ((slack >= slack_min) & (low_all >= scene.CRATE_WALL + V6_FLOOR_CLEAR_M - h)
                & (low_z <= V6_DROP_MAX_M) & (high_z <= tx.ABOVE_RIM_MAX_M))
        capped = fits & ((pad_swing <= V6_PAD_SWING_MAX_DEG) if V6_PAD_SWING_MAX_DEG is not None else True)
        ok = np.flatnonzero(fits if V6_PAD_SWING_ORDER else capped)
        self._pool = (int(fits.sum()), int(capped.sum()))          # poses that fit the truss, and under the cap
        POOL_LOG.append(self._pool)
        self._swing.update({tuple(np.asarray(Qs[i], float)): float(pad_swing[i]) for i in ok})
        # Most room first, as v5. ⚠️ NOT nearest the crate's centre by grip point: the truss sticks out 10-20 cm along
        # its stalk from the grip, so centred grips put it over the +y wall (5 of 5 seeds, 2026-11-24).
        if V6_PAD_SWING_ORDER and V6_PAD_SWING_MAX_DEG is not None:   # under the cap first, each part most room first
            return (ok[np.lexsort((-slack[ok], pad_swing[ok] > V6_PAD_SWING_MAX_DEG))], slack, Qs, Os, Rs, point,
                    low_all)
        return ok[np.argsort(-slack[ok], kind="stable")], slack, Qs, Os, Rs, point, low_all

    def _aligned_poses(self, ikf, point, rel, radii):
        """V6_ALIGN: arm poses (Q, gripper o, R) over the crate on the cut pose's pitch, tilt and branch, so gravity in
        the gripper frame is as at the cut. Built, not solved: lift and elbow on a V6_ALIGN_STEP_DEG grid with the
        wrist bend compensating (_aligned), kept when the hanging truss's lowest point is V6_FLOOR_CLEAR_M ..
        V6_ALIGN_LOW_MAX_M over the crate floor; each then turned about the pan axis in V6_ALIGN_STEP_DEG steps, kept
        when the grip point is V6_ALIGN_INSET_M inside the walls. Cached per cut pose."""
        Rc, qc = self._R_cut, self._q_cut
        # ⚠️ per cut pose AND carried shape: two cut candidates with the same orientation at another point of the stalk
        # carry the truss elsewhere in the gripper (its lowest point, the grip point), so neither may reuse the other's
        key = (np.round(qc, 6).tobytes(), np.round(rel, 5).tobytes(), np.round(point, 6).tobytes())
        if self._align.get("key") != key:
            (cx, cy), (sx, sy, _) = ts.CRATE_CENTRE, ts.CRATE_SIZE
            _, _, sgn, c = self._align_kin()
            low = float((rel @ Rc[2] - radii).min())           # the truss's lowest point below the gripper's origin
            step = np.radians(V6_ALIGN_STEP_DEG)
            Xn, On, Rn, _ = self._align_nodes(point)
            hgt = On[:, 2] + low - scene.CRATE_WALL if len(Xn) else np.zeros(0)
            nodes = [(Xn[i], On[i], Rn[i]) for i in np.flatnonzero((hgt >= V6_FLOOR_CLEAR_M)
                                                                    & (hgt <= V6_ALIGN_LOW_MAX_M))]
            Qs, Os, Rs = [], [], []
            if nodes:
                X, O, R = (np.array([n[k] for n in nodes]) for k in range(3))
                c3 = np.array([c[0], c[1], 0.0])
                for pan in np.arange(ikf.lo[0], ikf.hi[0] + 1e-9, step):
                    Rz = _rotz(sgn * (pan - qc[0]))
                    O2 = c3 + (O - c3) @ Rz.T
                    R2 = np.einsum("ij,njk->nik", Rz, R)
                    g = O2 + R2 @ point
                    ok = ((np.abs(g[:, 0] - cx) <= sx / 2 - V6_ALIGN_INSET_M)
                          & (np.abs(g[:, 1] - cy) <= sy / 2 - V6_ALIGN_INSET_M))
                    for i in np.flatnonzero(ok):
                        Qs.append(np.r_[pan, X[i, 1:]])
                        Os.append(O2[i])
                        Rs.append(R2[i])
            self._align.update(key=key, low=low,
                               poses=(Qs, np.array(Os).reshape(-1, 3), np.array(Rs).reshape(-1, 3, 3)))
        return self._align["poses"]

    def _release_pose(self, fit, i, cchk, clear=tx.CARRY_CLEAR_M):
        """Pose i of _release_fit's: (x_rel, x_above, lower, grip point, pitch, slack), or None if it hits."""
        _, slack, Qs, Os, Rs, point, _ = fit
        ikf = self.ik_free
        xr = np.asarray(Qs[i])
        p = Os[i] + Rs[i] @ point
        pitch = np.degrees(np.arcsin(np.clip(Rs[i][2, 2], -1, 1)))
        tilt = np.degrees(np.arcsin(np.clip(Rs[i][2, 0], -1, 1)))
        if cchk.hits(xr, soft_clearance=clear):
            return None
        above = V6_ABOVE_CRATE_M if self.v6 else tx.ABOVE_CRATE_M
        if self.v6 and V6_ALIGN and self._q_cut is not None and self._is_aligned(xr):
            up = self._move_point_aligned(xr, np.array([0.0, 0.0, above]), 7, point)   # chained from the release pose
            low = [None] if up is None else up[::-1] + [xr]
        else:
            low = [ikf.solve(p + [0, 0, above * (1 - v)], pitch, tilt, point, x0=xr)
                   for v in np.linspace(0, 1, 8 if self.v6 else 6)]
        if any(x is None for x in low) or not cchk.path_ok(low[:-1] + [xr], soft_clearance=clear):
            if self.v6 and V6_REQUIRE_DROP:
                return None                    # only poses with the straight closed-loop lowering (V6_ABOVE_CRATE_M)
            low = [xr]                         # no straight drop for this pose: the carry goes to it directly
        else:
            low[-1] = xr
        return xr, low[0], low, p, pitch, float(slack[i])

    def _release(self, carry, cchk, r):
        """The release pose: TrussPlanner._after_cut's fit (seen from above every carried fruit and the stalk inside
        the walls, also laid flat; most room to the walls first), deterministic. Returns (x_rel, x_above, lower,
        grip point, pitch, slack) or why not."""
        fit = self._release_fit(carry, r)
        for i in fit[0][:tx.RELEASE_SAMPLES // 4]:
            rel = self._release_pose(fit, i, cchk)
            if rel is not None:
                return rel
        return f"no release pose: {len(fit[0])} of {len(fit[2])} poses over the crate fit the truss"

    def _release_alternates(self, carry, cchk, r, n_max=RELEASE_FALLBACK_N, relaxed=False, low=False, exclude=()):
        """Release fallback (v4's second pass, phase B; and phase C): after _release's choice, which failed, the next
        clear poses in the same order, each RELEASE_DISTINCT_M or more from every clear pose before it (that choice
        too), at most n_max, past its first RELEASE_SAMPLES // 4 too. Yields what _release returns (rank in
        self._rel_rank). relaxed (phase C, last): only the poses v4's fit rejects and RELAXED_SLACK_M lets in, checked
        at tx.CARRY_CLEAR_MIN_M. low: lowest drop first (after a release that failed in the physics), skipping the
        grip points in exclude (tried already) instead of _release's choice."""
        fit = self._release_fit(carry, r, RELAXED_SLACK_M if relaxed else tx.RELEASE_SLACK_M)
        order, slack, Qs, Os, Rs, point, low_z = fit
        if low:
            order = order[np.argsort(low_z[order], kind="stable")]
        clear = tx.CARRY_CLEAR_MIN_M if relaxed else tx.CARRY_CLEAR_M
        seen, n = [np.asarray(q) for q in exclude], 0
        for rank, i in enumerate(order):
            if relaxed and slack[i] >= tx.RELEASE_SLACK_M:
                continue                       # tried already with the normal fit
            p = Os[i] + Rs[i] @ point
            if any(np.linalg.norm(p - q) < RELEASE_DISTINCT_M for q in seen):
                continue
            if cchk.hits(np.asarray(Qs[i]), soft_clearance=clear):
                continue
            seen.append(p)
            if not relaxed and not low and len(seen) == 1 and rank < tx.RELEASE_SAMPLES // 4:
                continue                       # _release's own choice: it failed in the first pass
            self._rel_rank = rank
            yield self._release_pose(fit, i, cchk, clear)
            n += 1
            if n >= n_max:
                return

    def _build_legs(self, q0, to_std, leg, rest, d_jaw=None):
        """d_jaw: the MjData whose jaw the first leg ramps from (default the start frame's copy; rejoin passes
        the live data, where the jaw is at the hand-over)."""
        fm = self.fm
        n5 = lambda xs: np.array([fm.rad_to_norm(np.r_[x, 0])[:5] for x in xs])   # noqa: E731
        g_open = float(fm.rad_to_norm(np.r_[np.zeros(5), np.radians(tx.JAW_OPEN_DEG)])[5])
        g_now = float(fm.rad_to_norm(np.r_[np.zeros(5), (self.d if d_jaw is None else d_jaw).qpos[self.g_adr]])[5])
        self.g_open, self.g_shut = g_open, CLOSE_NORM

        def cart_lead(xs, metres, speed):
            W = n5(xs)
            L = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
            return max(L / max(metres / speed * FPS, 1.0), 1e-3)

        f = lambda s: max(1, int(round(s * FPS)))   # noqa: E731
        cut = n5([leg["x_cut"]])[0]
        L = []
        L.append(Leg("to_standoff", n5(to_std), jaw_to=(g_now, g_open), lead=FREE_LEAD))
        if len(leg["approach"]) > 1:
            L.append(Leg("approach", n5(leg["approach"]), jaw=g_open, servo=True,
                         lead=cart_lead(leg["approach"], leg["standoff_m"], APPROACH_M_S)))
        L.append(Leg("slide_in", n5(leg["slide"]), jaw=g_open, servo=True,
                     lead=cart_lead(leg["slide"], leg["slide_m"] + leg.get("meet_m", 0.0), CART_M_S)))
        L.append(Leg("settle_at_stalk", hold=cut, jaw=g_open, servo=True, frames=f(SETTLE_S)))
        L.append(Leg("close_cut", hold=cut, jaw_to=(g_open, CLOSE_NORM), servo=True, frames=f(CLOSE_S)))
        L.append(Leg("squeeze", hold=cut, jaw=CLOSE_NORM, servo=True, frames=f(SQUEEZE_S)))
        ret_m = float(np.linalg.norm([RETREAT_M, RETREAT_UP_M]))
        k = CARRY_SLOW_BIG if self.t["n"] >= BIG_TRUSS_N else None     # v5: heavy trusses slower (None: v4's pace)
        lead_ret, lead_car = cart_lead(rest["ret"], ret_m, CART_M_S), CARRY_LEAD
        if k is not None:
            lead_ret, lead_car = lead_ret * k, lead_car * k
        if self.v6:
            lead_car = min(lead_car, CARRY_LEAD * V6_CARRY_SCALE)
        L.append(Leg("retreat", n5(rest["ret"]), jaw=CLOSE_NORM, lead=lead_ret))
        self._lead_car = lead_car
        L += self._tail_legs(rest, lead_car)
        self.legs = L
        self.reset()

    def _tail_legs(self, rest, lead_car):
        """The legs from the carry to the crate on: to_crate, (settle_above, lower), release, drop, (lift), return,
        hold_end. _build_legs's tail, also used by the live release re-plan."""
        fm = self.fm
        n5 = lambda xs: np.array([fm.rad_to_norm(np.r_[x, 0])[:5] for x in xs])   # noqa: E731
        g_open, g_start = self.g_open, float(self.q_home[5])

        def cart_lead(xs, metres, speed):
            W = n5(xs)
            Lp = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
            return max(Lp / max(metres / speed * FPS, 1.0), 1e-3)

        f = lambda s: max(1, int(round(s * FPS)))   # noqa: E731
        L = []
        L.append(Leg("to_crate", n5(rest["to_crate"]), jaw=CLOSE_NORM, lead=lead_car))
        if len(rest["lower"]) > 1:
            if self.v6:
                L.append(Leg("settle_above", hold=n5([rest["lower"][0]])[0], jaw=CLOSE_NORM,
                             frames=f(V6_SETTLE_ABOVE_S)))
            L.append(Leg("lower", n5(rest["lower"]), jaw=CLOSE_NORM,
                         lead=cart_lead(rest["lower"], V6_ABOVE_CRATE_M if self.v6 else tx.ABOVE_CRATE_M, CART_M_S)))
        x_rel = n5([rest["lower"][-1]])[0]
        L.append(Leg("release", hold=x_rel, jaw_to=(CLOSE_NORM, g_open), frames=f(RELEASE_S)))
        L.append(Leg("drop", hold=x_rel, jaw=g_open, frames=f(DROP_S)))
        if rest.get("lift"):
            L.append(Leg("lift", n5(rest["lift"]), jaw=g_open,
                         lead=cart_lead(rest["lift"], V6_ABOVE_CRATE_M, CART_M_S)))
        L.append(Leg("return", n5(rest["home"]), jaw_to=(g_open, g_start), lead=FREE_LEAD))
        L.append(Leg("hold_end", hold=self.q_home[:5], jaw=g_start, frames=f(HOLD_END_S)))
        return L

    def _replan_release_live(self, q_norm):
        """V6_LIVE_RELEASE: re-plan the release from the truss as it hangs now, from the arm where it is. Replaces the
        legs from here on (to_crate onto the new release's above-pose, settle, lower, release, drop, lift, return);
        keeps the old ones if nothing fits. Returns True if it re-planned."""
        self.meta["live_release_tried"] = True
        m, d = self.m, self.d_live
        G = m.body("arm/gripper").id
        o, R = d.xpos[G].copy(), d.xmat[G].reshape(3, 3).copy()
        bodies = [m.body(self.t["body"]).id] + [m.body(fr["tomato"]).id for fr in self.t["fruit"]]
        rel = [(R.T @ d.xmat[b].reshape(3, 3), R.T @ (d.xpos[b] - o)) for b in bodies]
        masses = [float(m.body_subtreemass[bodies[0]])] + [float(m.body_mass[b]) for b in bodies[1:]]
        carry = (bodies, rel, masses)
        x_now = self.fm.norm_to_rad(np.r_[q_norm, 0.0])[:5]
        r = self.t["r"]
        saved = (tx.CARRY, tx.CARRY_AXIS)
        tx.CARRY, tx.CARRY_AXIS = "rigid", 0                 # held as it is now for the last short move
        try:
            point = truss_ik.cut_point_in_gripper(r)
            cchk = add_self_check(tx.CarryChecker(self.m, self.d, bodies, rel, point, hard=self.hard,
                                                  soft=self.soft))
            cchk.d.qpos[self.g_adr] = np.radians(-10.0)
            fit = self._release_fit(carry, r, live=True)
            rest = None
            for i in fit[0][:tx.RELEASE_SAMPLES // 4]:
                rel_pose = self._release_pose(fit, i, cchk)
                if rel_pose is None:
                    continue
                out = self._via(rel_pose, [x_now], "live", cchk, r, "live", extra=True)
                if isinstance(out, dict):
                    rest = out
                    break
        finally:
            tx.CARRY, tx.CARRY_AXIS = saved
        self.meta["live_release"] = rest is not None
        if rest is None:
            return False
        self.legs = self.legs[:self.i] + self._tail_legs(rest, getattr(self, "_lead_car", CARRY_LEAD))
        self.n_in_leg, self.s, self.stall, self.q_prev = 0, 0.0, 0, None
        self.meta.update(live_release_point=rest["release_point"], live_crate_slack_mm=rest["crate_slack_mm"])
        return True

    # --- running (every frame) ---
    def reset(self):
        self.i, self.n_in_leg, self.n_total = 0, 0, 0
        self.s, self.I, self.stall, self.q_prev = 0.0, np.zeros(5), 0, None
        self.phases = []

    @property
    def done(self):
        return self.i >= len(self.legs) or self.n_total >= MAX_S * FPS

    @property
    def leg(self):
        return self.legs[min(self.i, len(self.legs) - 1)].name

    def _next_leg(self):
        self.phases.append((self.legs[self.i].name, self.n_in_leg))
        self.i += 1
        self.n_in_leg, self.s, self.stall, self.q_prev = 0, 0.0, 0, None
        if self.done:
            self.meta.update(frames=self.n_total, seconds=round(self.n_total / FPS, 2),
                             timed_out=self.i < len(self.legs))

    def _low_fruit_m(self):
        """v6: how far the target truss's lowest fruit (live, as it really hangs) is above the crate floor (m)."""
        d, m = self.d_live, self.m
        return min(d.xpos[m.body(f["tomato"]).id][2] - f["radius"] for f in self.t["fruit"]) - scene.CRATE_WALL

    def act(self, state_norm):
        """The action for this frame from the measured state (norm, 6): {joint.pos: norm}."""
        q = np.asarray(state_norm, float)[:5]
        g = self.legs[self.i]
        if (self.v6 and V6_LIVE_RELEASE and not self.meta.get("live_release_tried") and self.n_in_leg == 0
                and g.name in ("settle_above", "release")):
            self._replan_release_live(q)
            g = self.legs[self.i]
        if self.v6 and g.name == "lower" and self._low_fruit_m() <= V6_STOP_ABOVE_FLOOR_M:
            for h in self.legs[self.i + 1:]:          # release and drop where it stopped (V6_ABOVE_CRATE_M note)
                if h.name in ("release", "drop"):
                    h.hold = q.copy()
                elif h.name == "lift" and h.W is not None:
                    h.trim_to(q.copy())
            self.meta["lower_stopped_early"] = True
            self._next_leg()
            g = self.legs[self.i]
        if g.W is not None:
            s = g.project(q, self.s - SEARCH_BACK * g.lead, self.s + SEARCH_AHEAD * g.lead)
            self.s = s
            target = g.point(s + g.lead)
            ref = g.point(s)
            prog = 1.0 if g.L < 1e-9 else min(1.0, (s + g.lead) / g.L)
        else:
            target = ref = g.hold
            prog = min(1.0, (self.n_in_leg + 1) / g.frames)
        if g.servo:
            # the sag correction on the cross-track error (path point nearest the arm minus the arm), not on the lead
            self.I = np.clip(self.I + expert.SAG_KI * (ref - q), -expert.SAG_MAX_NORM, expert.SAG_MAX_NORM)
        else:
            self.I *= 0.9
        jaw = g.jaw if g.jaw_to is None else g.jaw_to[0] + (g.jaw_to[1] - g.jaw_to[0]) * prog
        a = np.r_[target + self.I, jaw]
        self.n_in_leg += 1
        self.n_total += 1
        # leg finished?
        if g.W is not None:
            end = g.W[-1]
            moved = np.inf if self.q_prev is None else float(np.abs(q - self.q_prev).max())
            self.stall = self.stall + 1 if moved < STALL_NORM else 0
            self.q_prev = q
            at_end = self.s + g.lead >= g.L or (g.name == "escape" and self.stall >= ESCAPE_STALL_FRAMES)
            close = float(np.abs(q - end).max()) < END_TOL["servo" if g.servo else "free"]
            if at_end and (close or self.stall >= STALL_FRAMES):
                self._next_leg()
        elif self.n_in_leg >= g.frames or (g.until is not None and g.until()):
            self._next_leg()
        if not self.done and self.n_total >= MAX_S * FPS:
            self.meta.update(frames=self.n_total, seconds=round(self.n_total / FPS, 2), timed_out=True)
        return {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, a)}

    def rejoin(self, state_norm):
        """Take over from a state the teacher didn't drive to, before the cut (a hybrid hand-over, a DAgger
        takeover), keeping the start-frame plan's cut: if the arm is in among the plant, back out along the slide
        line to the first clear pose (ESCAPE_M), then a free move to the plan's standoff, then the plan's own approach,
        slide, cut, carry and return. The cut and everything after it stay planned on the start-frame scene, where
        the stalk hangs again once the arm has let go of it. Evidence (check_teacher takeover, 63 held-out scenes,
        2026-11-16): re-entering the old plan's nearest leg (resync) failed 29 % with the arm 2 cm off; planning
        afresh on the live scene failed 37 % with the stalk swung 5 mm (it springs back to where the old plan aims)
        and found no cut pose for a tilted stalk. Returns the first leg's name; raises PlanError."""
        leg, rest = self._parts["leg"], self._parts["rest"]
        r = self.t["r"]
        q = self.fm.norm_to_rad(np.r_[np.asarray(state_norm, float)[:5], 0.0])[:5]
        self.jaw(tx.JAW_OPEN_DEG)
        esc, how_out = [], None
        if not self.chk.free(q, **self.kw):
            p_now, R_now = self.ik.frame(q, r)
            for how_out in ESCAPE_DIRS:
                for back in ESCAPE_M:
                    # position only: the slide line's IK (stalk-aligned) turned the wrist inside the plant on its
                    # first point and the arm jammed there until MAX_S (seed 9900089, 2026-11-16)
                    v = {"slide": -leg["f"], "back": R_now[:, 2], "up": np.array([0.0, 0.0, 1.0])}[how_out]
                    xs = self._move_point(q, v * back, ESCAPE_STEPS)     # fingers run along gripper -z
                    if xs is not None and self.chk.free(xs[-1], **self.kw):
                        esc = [q] + xs
                        break
                if esc:
                    break
            else:
                raise PlanError("rejoin: no clear pose backing out (slide line, finger axis or up)")
        start = esc[-1] if esc else q
        path, how = self.free_move(start, leg["approach"][0], self.chk, r, extra=True, **self.kw)
        if path is None:
            raise PlanError("rejoin: no free move to the standoff")
        self._build_legs(start, path, leg, rest, d_jaw=self.d_live)
        if esc:
            fm = self.fm
            W = np.array([fm.rad_to_norm(np.r_[x, 0])[:5] for x in esc])
            L = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
            g_now = float(fm.rad_to_norm(np.r_[np.zeros(5), self.d_live.qpos[self.g_adr]])[5])
            self.legs[0].jaw, self.legs[0].jaw_to = self.g_open, None
            self.legs.insert(0, Leg("escape", W, jaw_to=(g_now, self.g_open), servo=True,
                                    lead=max(L / max(back / CART_M_S * FPS, 1.0), 1e-3)))
        self.meta.update(rejoined=dict(escape_mm=round(back * 1000) if esc else 0, escape_dir=how_out if esc else None,
                                       to_standoff_via=how))
        self.reset()
        return self.legs[0].name

    def _move_point(self, q, delta, n):
        """Joint poses (rad) moving the gripper by `delta` (world, m) in n steps, orientation free: damped least
        squares on the planning copy (self.chk), so the live scene isn't touched. None if it can't follow."""
        m, d = self.chk.m, self.chk.d
        b = m.body("arm/gripper").id
        dadr = [m.jnt_dofadr[m.joint("arm/" + j).id] for j in arm_map.JOINTS[:5]]
        d.qpos[self.chk.adr] = q
        mujoco.mj_fwdPosition(m, d)
        p0, x, xs = d.xpos[b].copy(), np.asarray(q, float).copy(), []
        for u in np.linspace(0, 1, n + 1)[1:]:
            goal = p0 + np.asarray(delta) * u
            for _ in range(5):
                jacp = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jacp, None, b)
                J = jacp[:, dadr]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), goal - d.xpos[b])
                x = np.clip(x + dq, self.lo, self.hi)
                d.qpos[self.chk.adr] = x
                mujoco.mj_fwdPosition(m, d)
            if np.linalg.norm(goal - d.xpos[b]) > 0.003:
                return None
            xs.append(x.copy())
        return xs

    def resync(self, state_norm, cut, released):
        """Re-enter from a state the teacher didn't drive to (a DAgger takeover): the eligible leg whose path passes
        nearest the arm. cut: the target's stalk is cut; released: it lies in the crate. ⚠️ Not used by step 0's
        checks yet; written now so DAgger (§6.4) has one entry point."""
        q = np.asarray(state_norm, float)[:5]
        names = [g.name for g in self.legs]
        if released:
            pool = [names.index("return")]
        elif cut:
            pool = [k for k, n in enumerate(names) if n in ("squeeze", "retreat", "to_crate", "lower")]
        else:
            pool = [k for k, n in enumerate(names) if n in ("to_standoff", "approach", "slide_in")]

        def dist(k):
            g = self.legs[k]
            if g.W is None:
                return float(np.linalg.norm(q - g.hold))
            return float(np.linalg.norm(q - g.point(g.project(q, 0.0, g.L))))
        k = min(pool, key=dist)
        self.i, self.n_in_leg, self.stall, self.q_prev = k, 0, 0, None
        g = self.legs[k]
        self.s = 0.0 if g.W is None else g.project(q, 0.0, g.L)
        self.I = np.zeros(5)
        return g.name
