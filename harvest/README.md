# Harvest pipeline

Teleop demonstrations → ACT baseline (data check) → π0.5 fine-tune → gated end-to-end runs. The policy
learns **how** to pick a single cherry tomato. The ripeness gate (`../ripeness/`)
decides **whether** to.

⚠️ The vault notes 03–06 were written for whole-truss cutting. The module now picks
single fruits, so names and the task string here are placeholders (`harvest.env`).
Lock them before the first episode.

## Layout

```
harvest/
├── harvest.env           dataset name, task string, policy repos, episode timing (LOCK before recording)
├── datasets.md           manifest of every Hub dataset + checkpoint (the data itself lives on the Hub)
├── scripts/
│   ├── common.sh         shared robot/teleop/camera flags (from config/hardware.env)
│   ├── teleop.sh         pre-flight check
│   ├── record.sh         one recording session (--resume for later ones)
│   ├── check_dataset.py  episode count, camera keys, normalised-not-degrees
│   ├── train_act.sh      smoke | full
│   ├── train_smolvla.sh  superseded by π0.5 (π0.5 training script not written yet)
│   └── rollout.sh        eval (episodic, saved) | run (one shot)
├── control/              target-centring controller (hybrid approach, see below)
├── harvest_loop.py       gated v1 loop: gate → rollout → log
├── plan/                 episode_plan.csv, eval_trials.csv, system_trials.csv
└── logs/                 session_log.md, harvest_log.csv
```

## Decision to make before recording: how does the policy know *which* fruit?

On a vine with ten fruits, the policy doesn't know which one the gate chose.

| Option | How | Catch |
|---|---|---|
| Overlay | Draw the target box onto the frames fed to the policy | The gate must also run **while recording**, so the overlay is in the training frames |
| **Hybrid** (vault recommendation) | Gate gives the target pixel → `control/` centres the claw camera on it → the policy only does the final grasp and pick | More code, but a much smaller, easier job for the policy |

**Chosen (2026-10-10): overlay, two cameras.** The birdseye gate selects the target
(recall-tuned) and its box, detected on the start frame, is drawn onto the birdseye frames
for the whole episode. The claw gate confirms just before the close (precision-tuned) or
aborts and logs `claw_veto`. v1 scenes hold exactly one ripe fruit, so the box can be drawn
after recording. ⚠️ That box-drawing script isn't written yet, and `harvest_loop.py`
and `control/` still follow the older hybrid design. The box **style** is fixed in
`target_box.py` (magenta, 5 px, outside the fruit), the one function the sim, the overlay script
and the rollout all draw with. The birdseye selection rule is `ripeness/gate/birdseye.py`.

## Locked recording decisions

| Decision | Value |
|---|---|
| End effector | _(final version from `hardware/end_effector/`)_ |
| Cameras | `birdseye` (fixed mast) + `claw` (wrist), 640×480 @ 30 fps |
| `use_degrees` | `false`, robot and teleop |
| Task string | `harvest.env` `TASK` |
| Dataset | `harvest.env` `DATASET`, `--dataset.no_stamp=true` |

## Variation schedule

Grid on the rig: **3 heights × 3 lateral × 3 rotations = 27 cells**, plus distractors
(none ~40 % / unripe fruit nearby ~60 %) and lighting (standard ~70 % / varied ~30 %).
Every recorded cell gets ≥ 4 episodes. **Hold out 3 cells**: never record them, and
only evaluate on them.

Fill `plan/episode_plan.csv` (one row per planned episode), **shuffle it**, then
record in that order. Plan 150 so that ≥ 100 survive deleting bad ones.

## One episode

Rest pose → approach → align (target centred in the claw view) → one decisive close →
retract straight back → carry to the crate → release → rest pose. Deliberate, not
slow. Fumbled the align? Press ← and re-record. Don't correct on camera.

## Steps

1. `scripts/teleop.sh`: pre-flight (wrist roll OK, both streams live, cable free).
2. `scripts/record.sh` per session of 25. Log each one in `logs/session_log.md`.
3. Watch every episode (`lerobot-dataset-viz`). Delete the bad ones into `$DATASET_CLEAN`
   with `lerobot-edit-dataset --operation.type delete_episodes`.
4. `scripts/check_dataset.py $DATASET_CLEAN`. Push to the Hub, add it to `datasets.md`.
5. `scripts/train_act.sh smoke`, then `full`. Run `scripts/rollout.sh eval` on the 40k and
   80k checkpoints, and log them in `plan/eval_trials.csv`.
6. π0.5: fine-tune `lerobot/pi05_base` on a rented GPU (or the hosted π API) and serve it
   with lerobot `async_inference`, which supports pi05 + so101_follower at commit
   `5aa74557`. Same eval grid.
7. `harvest_loop.py` over `plan/system_trials.csv`. Metrics: harvest success, gate
   recall, end-to-end ripe yield, **false-pick rate (target: 0)**, median time per pick.

⚠️ Scripts are seeded from the vault notes (checked there against lerobot commit
`5aa74557`) and **have not been run from this repo yet**.
