# Dataset + checkpoint manifest

LeRobot datasets (AV1 video) and policy checkpoints are too big for git, so they live on
the Hugging Face Hub under `krish5831`. This file is the record of what exists and
what each one is for. Add a row the day something is pushed.

Local copies: `~/.cache/huggingface/lerobot/krish5831/<name>/`.

## Sim datasets (MuJoCo, `../sim/`)

Scripted by the sim's expert, not teleoperated. Same robot_type (`so_follower`), units
(`use_degrees=false`), fps (30) and codec as real recordings. Each run's attempt log is
`~/.cache/blush_sim/truss/<name>_<stamp>.jsonl`.

| Repo | Date | Episodes | Cameras | Made with | Notes |
|---|---|---|---|---|---|
| `krish5831/blush_sim_truss_v1` | 2026-11-03, DAgger 11-05/06 | 300 | birdseye + claw | `generate_truss.py --episodes 200 --seed-start 210000`, then `dagger_truss.py` (100, seeds 7000000+) | Whole-truss harvesting. `observation.environment_state` = target box + cut point. Kept 200 of 513 scripted attempts (`run_v1.log`) |
| `krish5831/blush_sim_truss_v2` | 2026-11-06 | 200 | base + claw | `generate_truss.py --episodes 200 --repo-id krish5831/blush_sim_truss_v2 --rig base --start-jitter-deg 3 --hold-share 0.10 --balance --seed-start 7600000` | Rig "base". 20 stay-home episodes, 25 % miss + recovery. Private on the Hub. What `train_sim.sh act-truss` trains on |

Watch an episode (a window per camera): `lerobot-dataset-viz --repo-id krish5831/blush_sim_truss_v2 --episode-index 0`

Deleted (single-fruit picking, replaced by trusses): `blush_sim_scripted_v0` (2026-10-28),
`blush_sim_scripted_v1` (200 episodes, local + Hub, 2026-11-03); its copy
`blush_sim_scripted_v1_box` is gone from the local cache too.
`train_sim.sh act-laptop` / `act-box` still default to those names.

## Real demonstration datasets

| Hub repo | Date | Episodes | Cells | Recorded with | Notes |
|---|---|---|---|---|---|
| | | | | | |

⚠️ Don't mix in `krish5831/tissues_into_box_20260928_195253`. It was recorded with
`use_degrees=true` on raw ttyACM ports.

## Policies

Sim policies train into `outputs/train/<run>/` (local, gitignored; `sim/scripts/train_sim.sh`
names the run). Add a row here when one is pushed or evaluated with `sim/eval_truss.py`.

| Hub repo | Type | Trained on | Steps | Where trained | Eval result (`plan/eval_trials.csv`) |
|---|---|---|---|---|---|
| | | | | | |

## Rollout / eval datasets

| Hub repo | Policy + checkpoint | Trials | Notes |
|---|---|---|---|
| | | | |
