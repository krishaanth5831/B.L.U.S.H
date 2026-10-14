# CLAUDE.md — B.L.U.S.H

Vinea harvest module: SO-101 ripeness gate + cherry-tomato picking policy.
Read `README.md` for the pipeline and the data layout.

## Git workflow (hard rules)

- Commit and push straight to `dev`. **No feature branches**, not even for a PR (the
  user's rule, 2026-10-14). Never push to `main`. It's protected (PR required,
  enforce_admins on).
- Every PR is `dev` → `main`, and only when asked: `gh pr create --base main --head dev`.
  Always pass `--base` explicitly.
- **Never merge a PR**, to any branch. The user merges.

## Hardware rules that cost real debugging time

- ⚠️ Address the arms by `/dev/serial/by-id/...` only (see `config/hardware.env`).
  `ttyACM0/1` swap between boots and once inverted leader/follower.
- ⚠️ `use_degrees=false` on robot and teleop: record, train data, deploy. The
  degrees branch of lerobot's `_unnormalize` skips range clamping, which caused
  the wrist_roll runaway. Never mix in data recorded with `use_degrees=true`
  (e.g. the tissues dataset).
- ⚠️ Video codec `libsvtav1`. `h264_nvenc` dies on frame 1 on this laptop.
- ⚠️ Calibration ids `follower_white` / `leader_black`. Omitting `--*.id` writes
  `None.json` and silently uses the wrong calibration.
- ⚠️ `lerobot-record` stamps `_YYYYMMDD_HHMMSS` onto `repo_id` unless
  `--dataset.no_stamp=true`.

## Code style

Every constant carries a comment with where its value came from (measured,
copied from a doc, or a guess marked `unverified`). Use ⚠️ on traps. Snippets
seeded from the vault have **not been run yet**. Don't claim otherwise until
they have.

## Envs

- `lerobot`: recording, training, rollout (editable checkout at
  `~/Desktop/personal_projects/github/lerobot`).
- `ripeness`: YOLO only (`envs/ripeness.yml`). The two pin different
  torch/opencv versions.

Hugging Face user is `krish5831`. GitHub user is `krishaanth5831`.
