# Dataset + checkpoint manifest

LeRobot datasets (AV1 video) and policy checkpoints are too big for git, so they live on
the Hugging Face Hub under `krish5831`. This file is the record of what exists and
what each one is for. Add a row the day something is pushed.

No local copies are kept (K7, 2026-11-16): everything is on the Hub, private; lerobot fetches what a script needs
into `~/.cache/huggingface/lerobot/krish5831/<name>/`.

## Naming (K7, 2026-11-16: "a strict and good naming convention which makes sense for all these data sets")

`krish5831/<module>_<world>_<source>_v<N>[_<subset>][_<YYYYMMDDhhmm>]`

| Field | Values |
|---|---|
| module | `blush`, `shear`, `arm` (an SO-101 test that belongs to no module) |
| world | `sim`, `real` |
| source | `teacher` (closed-loop scripted teacher), `scripted` (open-loop scripted expert), `teleop` (a human on the leader arm) |
| vN | the version of that source: teacher v3, the v2 truss planner, camera rig v1 / v2 |
| subset | only for a partial or special set: `pilot`, `onelayout`, `onerow`, `markercup`, ... |
| time | only for one-off real recordings: when the recording started |

All repos are private and carry a short card naming what they are and their former name.

## All datasets (2026-11-17)

| Repo | Former name | Episodes | Cameras | What |
|---|---|---|---|---|
| `blush_sim_teacher_v3` | | 610 (525 cuts, 85 holds) | base + claw | Step 2 of the post-mortem ladder: teacher v3, half with DART noise, seeds 9300000 to 9722070 |
| `blush_sim_teacher_v3_pilot` | | 5 | base + claw | Teacher v3 pilot |
| `blush_sim_teacher_v3_onelayout` | `blush_sim_teacher_onelayout_v1` | 50 | base + claw | Step 1: one layout (seed 9100009), all noisy |
| `blush_sim_teacher_v3_finisher_pilot` | | 10 (3951 frames) | base + claw | Last-stage finisher pilot (2026-11-16): the step-3 ACT policy approaches, teacher v3 takes over 20-50 mm from the cue's cut point (`rejoin`, its normal plan unchanged) and finishes; recorded from the hand-over on, kept only if the whole scene judged clean (`eval_truss.py --record`). Seeds 9890000 to 9890027; the finisher's range is 9800000 to 9899999 |
| `blush_sim_teacher_v3_finisher` | | 201 (78357 frames) | base + claw | The last-stage finisher's dataset (2026-11-17), same recipe as its pilot (ACT 100k approaches, teacher v3 takes over at 20-50 mm, recorded from the hand-over, whole scene judged clean). Recorded at dev 0fba9b1, before the `add_self_check` leak fix (160bbbd, no plan changes); merged from 33 parts with `merge_parts.py`. Seeds from 9800000, 9830000, 9860000 and 9870000 + 500 j, all inside 9800000 to 9899999 |
| `blush_sim_scripted_v2` | `blush_sim_truss_v2` | 200 | base + claw | The v2 scripted truss planner (random RRT routes, 25 % misses), seeds from 7600000; SmolVLA era |
| `shear_sim_teacher_v3` | | 600 (540 cuts, 60 holds) | base + claw | S.H.E.A.R step 2 |
| `shear_sim_teacher_v3_onerow` | `shear_sim_teacher_onerow_v1` | 50 | base + claw | S.H.E.A.R step 1, row 910057 |
| `shear_sim_scripted_v1` | | 200 | base + claw | S.H.E.A.R's v1 scripted expert; SmolVLA era |
| `arm_real_teleop_v1_markercup_202609202013` | `marker_test_20260920_201308` | | | Real SO-101, leader-arm teleop: the marker into the cup, camera rig v1 |
| `arm_real_teleop_v1_markercup_202609202136` | `marker_test_20260920_213650` | | | The same |
| `arm_real_teleop_v2_markercup_202609202219` | `marker_test_new_camera_setup_20260920_221944` | 5 | claw | The same, new camera rig (v2) |

⚠️ Renamed on the Hub (`blush_sim_scripted_v2`, the three `arm_*`) redirect from their old names. Uploaded fresh under
the new name (`*_onelayout`, `*_onerow`) do not: checkpoints trained on the old names (`act_teacher_onelayout_v1`)
find their dataset only under the new one.

Watch an episode (a window per camera): `lerobot-dataset-viz --repo-id krish5831/blush_sim_teacher_v3 --episode-index 0`

Deleted:
- 2026-11-08: `blush_sim_truss_v1` (birdseye + claw, 200 scripted + 100 DAgger, local + Hub), its backup,
  `blush_sim_truss_v1_pilot`, `blush_sim_truss_v2_pilot`.
- Single-fruit picking, replaced by trusses: `blush_sim_scripted_v0` (2026-10-28), `blush_sim_scripted_v1`
  (200 episodes, local + Hub, 2026-11-03) and its local copy `blush_sim_scripted_v1_box`.

## Real demonstration datasets

| Hub repo | Date | Episodes | Cells | Recorded with | Notes |
|---|---|---|---|---|---|
| | | | | | |

⚠️ Don't mix in `krish5831/tissues_into_box_20260928_195253`. It was recorded with
`use_degrees=true` on raw ttyACM ports.

## Policies

Sim policies train into `outputs/train/<run>/` (local, gitignored; `sim/scripts/train_sim.sh`
names the run). Add a row here when one is pushed or evaluated with `sim/eval_truss.py`.

None right now. Every earlier B.L.U.S.H policy (single-fruit ACT runs, SmolVLA LoRA / expert v1 / v2, and
`act_truss_v1`, stopped before its first checkpoint) was deleted locally and on the Hub on 2026-11-08. Next:
`sim/scripts/train_sim.sh act-truss` on `blush_sim_scripted_v2`.

| Hub repo | Type | Trained on | Steps | Where trained | Eval result (`plan/eval_trials.csv`) |
|---|---|---|---|---|---|
| | | | | | |

## Rollout / eval datasets

| Hub repo | Policy + checkpoint | Trials | Notes |
|---|---|---|---|
| | | | |
