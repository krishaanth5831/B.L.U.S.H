# Post-mortem diagnostics for `act_teacher_v5` (19-20 Nov 2026)

The measurements behind the v5 post-mortem (vault: `08 Modules/(C) BLUSH v5 Policy Post-Mortem ...`). Every script
writes its result files to `outputs/notes/2026-11-19_v5_postmortem/` (git-ignored), never into the repo tree.
Run from the `lerobot` env; the `offline_*.py` scripts need the GPU for 2-4 minutes each and no simulator.

| Script | Question it answers | Number it gave for the 100k checkpoint |
|---|---|---|
| `analyse_dataset.py` | What is in dataset v5: positions, timing, how alike the teacher's paths are, how well the cue pins the target | cut pixel alone pins the cut point to 3.3 mm; paths 46 mm apart at 2 s for near-identical targets |
| `offline_chunk_error.py` | On its own TRAINING frames, how far is the policy's command from the teacher's? | jaw 11.0 mm off at the teacher's close (13.5 at 20k) |
| `offline_sensitivity.py` | Which inputs decide where the jaw goes? | cue numbers followed 0.00-0.19 of the way; whole-image shift of 30 px moves the jaw 2-5 mm |
| `offline_smoothness.py` | Are the commands jittery, do chunks join up? | speed changes 4x the teacher's; 18 mm jump at a hand-over; first command 12.7 mm from the arm |
| `cell_geometry_and_claw_view.py` | Does the arm have room at the cut, and what does the wrist camera see of the stalk? | wrist joint within 10 units of its end stop in 201 of 736 cuts; wrist camera has the stalk at 4-5 px per mm from 1.5 s before the jaw shuts |
| `offline_state_follow.py` | Does the policy steer by its own joint readings? | near the cut it follows a changed reading 0.6-0.9, the wrist joint only 0.36-0.48 |
| `offline_offset_shape.py` | Is the miss one constant offset, and which way does it point? | 10.3 of the 10.8 mm is one shift of the whole plan; it points AWAY from the dataset's average command (cosine -0.91), 8 mm low |
| `offline_train_vs_inference.py` | Does the network answer differently while it trains (dropout on) and when it runs? | as it runs 11.0 mm, mean of 16 dropout-on answers 4.9 mm, stretch taken out 6.1 mm; the commands are 4-6 % further out as it runs |
| `run_diag_batch1.sh` | Rollout batch: same test twice, teacher alone, training scenes, shifted cue / images | results in `outputs/diag/runs/<name>/` |
| `analyse_batch1.py` | Reads that batch: noise floor, teacher vs policy aim, training vs held-out, does the jaw follow the cue | teacher 86/100; policy 13/50 on training scenes; 3 of 50 scenes flip between identical runs |
| `analyse_after_cut.py` | After the cut, when is the truss lost? | policy: 19 of 64 cuts lost within 0.5 s; teacher: 11 of 100 miss the crate at release; `--group` adds pad force: in 46 of 51 trusses lost at the cut only one pad pressed the stalk (six traced runs) |
| `where_at_shut.py` | Where is the stalk, in the gripper's frame, when the jaw first shuts? | 150 runs: outside the fixed finger 84, between the fingers at blade depth 30 (20 successes), deeper than the blade 26 |
| `turn_reach_height.py` | Which motion leaves the stalk beside the jaw: turn, reach or height? | 5.8 deg of turn left at 3.33 s, 3-3.5 deg at the closing; reach within 2.5 mm; jaw 12-15 mm low |
| `offline_first_plan_turn.py` | Does the first plan finish the turn, on recorded frames? | yes, within about 1 deg (on RECORDED frames, which a test never gives it; see below) |
| `first_frame_check.py` | Does the simulator's first observation match the recorded frame 0 of the same scene? | joints and cue identical; pictures 2.2 / 1.2 grey levels apart; the claw picture alone moves the plan 107 mm at 1 s |
| `claw_frame_leak.py` | Does the claw camera's start picture tell the policy how the teacher will start? | yes: with the raw picture the first plan ends 6.8 deg short (11.7 at 20k), through the encoder alone 0.5 |
| `replay_raw_vs_recorded.py` | Through a replayed episode: raw against recorded pictures; which frames of a clip shape a stored frame | plans 46-110 mm apart at frame 0, 2-5 mm later; a stored frame depends on the next frame and nothing further |
| `hold_or_go.py`, `hold_start_pictures.py` | Do the start pictures decide stay-home against go? | not on training episodes; held-out: the claw picture through the encoder sends 3 of 20 nothing-ripe plans off |
| `encoder_leak_test.py` | Which encoder settings make a stored frame depend on its neighbours? | v5's: a trace of the next frame in every key frame; frame-independent: AV1 low delay + every frame a key frame (26 kB per frame), H.264 qp 18 every frame a key frame, lossless |
| `start_of_run.py` | The measures written down before the picture tests: travel, turn at 3.33 s, stalk outside the fixed finger, paired success; `--by-opening` splits by the teacher's opening move | training scenes 13 -> 20 -> 24 of 50; held-out with the encoder: travel 99 -> 190 mm, turn left 4.8 -> 1.4 deg, outside 60 -> 36 %, success 19 -> 21 |
| `run_traced_test.sh`, `run_std_test.sh`, `paired_vs_baseline.py` | Launch a traced test / a standard 100-scene test; compare every 100-scene variant with the baseline scene by scene | |

⚠️ The checkpoint's settings show the 6 cue numbers are NOT rescaled by the library (their input type is missing
from `normalization_mapping`), while joints and actions are. Add `ENV` to that map in the next training command.

⚠️ **Training pictures are read back from the dataset's video; a test gives the policy raw renders** (2026-11-20).
Every key frame of the video carries a trace of the next frame (SVT-AV1's look-ahead), and at the start pose that trace
is the only difference between the claw camera's pictures of different episodes. Every offline number above was
computed on recorded frames: it describes the policy on its training input, not in a test. `eval_truss.py
--codec-still 1` passes live pictures through the encoder before the policy plans; `--start-frame recorded` (training
scenes only) uses the dataset's own frame 0. Results in the vault note, sections 4.5-4.6.

⚠️ With dropout 0.1 the running answer differs from the training-mode answer: on v5's training frames that gap is
about half of the miss (`offline_train_vs_inference.py`). `eval_truss.py --unstretch 1.035` and `--mc-dropout 16`
(`--fix-from 1` to leave the first plan alone) are two ways to run the policy closer to how it was trained, with no
training. In rollouts: alone, no clear gain on held-out scenes (21 and 12 of 100 against 19); with the right start
picture on training scenes, `--unstretch` took 20/50 to 24/50. The one-position control,
which worked, had a gap of the same size at 20k steps.

`train_clean_50.jsonl`: the 50 clean training scenes used by the offline scripts and by the `train_clean` rollouts.

## The offline gates for the next training run

Before any rollout test of a new checkpoint, run `offline_chunk_error.py`, `offline_sensitivity.py`,
`offline_smoothness.py` and `offline_train_vs_inference.py` on it. The v5 values to beat:

| Gate | v5 at 100k | Why it matters |
|---|---|---|
| jaw error at the teacher's close, training frames | 11.0 mm (90 %: 16.7) | the tool needs about 6 mm |
| first command of a chunk vs where the arm is | 12.7 mm (teacher 3.4) | a general lack of fit, before any target finding |
| (once per DATASET) a stored frame against its own raw picture sent through the encoder alone, first frame of 20 episodes | v5: 0.9 grey levels (`encoder_leak_test.py`, column "alone") | must be 0 grey levels: the dataset's pictures depend on nothing but their own picture, so a test can reproduce them |
| gap between the running answer and the mean of 16 dropout-on answers, at the teacher's close | 11.0 against 4.9 mm | the tests run the network the first way; training only fits the second |
| share of a 30 px cue shift the jaw follows | 0.00-0.19 | the policy must use the cue it is given |

⚠️ The scripts are tied to dataset v5 (`REPO`, `SNAP`, the attempt logs in `~/.cache/blush_sim/truss/`) and to the
two-camera rig. For another dataset change those constants at the top.
