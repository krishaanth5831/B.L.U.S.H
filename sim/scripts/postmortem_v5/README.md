# Post-mortem diagnostics for `act_teacher_v5` (19-20 Nov 2026)

The measurements behind the v5 post-mortem (vault: `08 Modules/(C) BLUSH v5 Policy Post-Mortem ...`). Every script
writes its result files to `outputs/notes/2026-11-19_v5_postmortem/` (git-ignored), never into the repo tree.
Run from the `lerobot` env; the three `offline_*.py` scripts need the GPU for about 2 minutes each and no simulator.

| Script | Question it answers | Number it gave for the 100k checkpoint |
|---|---|---|
| `analyse_dataset.py` | What is in dataset v5: positions, timing, how alike the teacher's paths are, how well the cue pins the target | cut pixel alone pins the cut point to 3.3 mm; paths 46 mm apart at 2 s for near-identical targets |
| `offline_chunk_error.py` | On its own TRAINING frames, how far is the policy's command from the teacher's? | jaw 11.0 mm off at the teacher's close (13.5 at 20k) |
| `offline_sensitivity.py` | Which inputs decide where the jaw goes? | cue numbers followed 0.00-0.19 of the way; whole-image shift of 30 px moves the jaw 2-5 mm |
| `offline_smoothness.py` | Are the commands jittery, do chunks join up? | speed changes 4x the teacher's; 18 mm jump at a hand-over; first command 12.7 mm from the arm |
| `cell_geometry_and_claw_view.py` | Does the arm have room at the cut, and what does the wrist camera see of the stalk? | wrist joint within 10 units of its end stop in 201 of 736 cuts; wrist camera has the stalk at 4-5 px per mm from 1.5 s before the jaw shuts |
| `offline_state_follow.py` | Does the policy steer by its own joint readings? | near the cut it follows a changed reading 0.6-0.9, the wrist joint only 0.36-0.48 |
| `run_diag_batch1.sh` | Rollout batch: same test twice, teacher alone, training scenes, shifted cue / images | results in `outputs/diag/runs/<name>/` |
| `analyse_batch1.py` | Reads that batch: noise floor, teacher vs policy aim, training vs held-out, does the jaw follow the cue | teacher 86/100; policy 13/50 on training scenes; 3 of 50 scenes flip between identical runs |
| `analyse_after_cut.py` | After the cut, when is the truss lost? | policy: 19 of 64 cuts lost within 0.5 s; teacher: 11 of 100 miss the crate at release |

⚠️ The checkpoint's settings show the 6 cue numbers are NOT rescaled by the library (their input type is missing
from `normalization_mapping`), while joints and actions are. Add `ENV` to that map in the next training command.

`train_clean_50.jsonl`: the 50 clean training scenes used by the offline scripts and by the `train_clean` rollouts.

## The three offline gates for the next training run

Before any rollout test of a new checkpoint, run `offline_chunk_error.py`, `offline_sensitivity.py` and
`offline_smoothness.py` on it. The v5 values to beat:

| Gate | v5 at 100k | Why it matters |
|---|---|---|
| jaw error at the teacher's close, training frames | 11.0 mm (90 %: 16.7) | the tool needs about 6 mm |
| first command of a chunk vs where the arm is | 12.7 mm (teacher 3.4) | a general lack of fit, before any target finding |
| share of a 30 px cue shift the jaw follows | 0.00-0.19 | the policy must use the cue it is given |

⚠️ The scripts are tied to dataset v5 (`REPO`, `SNAP`, the attempt logs in `~/.cache/blush_sim/truss/`) and to the
two-camera rig. For another dataset change those constants at the top.
