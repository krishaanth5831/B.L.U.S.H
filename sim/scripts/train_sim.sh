#!/usr/bin/env bash
# train_sim.sh: train on sim data, or on sim + real merged (lerobot env). Same recipes as harvest/scripts.
#   sim/scripts/train_sim.sh act-smoke        500 ACT steps on the laptop: data loads, both cameras, loss falls
#   sim/scripts/train_sim.sh act-full         ACT on the sim dataset (rented 4090)
#   sim/scripts/train_sim.sh merge <real_ds>  sim + real into $COTRAIN_DATASET (lerobot-train takes one dataset)
#   sim/scripts/train_sim.sh pi05-smoke [ds]  200 π0.5 steps, rented GPU (default ds: $COTRAIN_DATASET)
#   sim/scripts/train_sim.sh pi05-full  [ds]  the real π0.5 run, rented GPU
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
OUT=outputs/train
DS=${2:-$COTRAIN_DATASET}
# π0.5 recipe from the vault's Throwaway Pilot step 11 (checked against lerobot 5aa74557).
PI05=(--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base
      --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
      --policy.n_action_steps=10 --policy.empty_cameras=1 --policy.gradient_checkpointing=true
      --policy.dtype=bfloat16 --policy.device=cuda)
case "${1:-}" in
  act-smoke)
    lerobot-train --dataset.repo_id="$SIM_DATASET" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --batch_size=8 --steps=500 --save_freq=500 \
      --output_dir=$OUT/act_sim_smoke --job_name=act_sim_smoke --wandb.enable=false ;;
  act-full)
    lerobot-train --dataset.repo_id="$SIM_DATASET" --policy.type=act --policy.device=cuda \
      --policy.repo_id="$SIM_ACT_POLICY" --batch_size=32 --steps=80000 --save_freq=20000 --num_workers=8 \
      --output_dir=$OUT/act_sim_v1 --job_name=act_sim_v1 --wandb.enable=true ;;
  merge)
    REAL=${2:?real dataset, e.g. \$DATASET_CLEAN}
    # ⚠️ Refuses if fps, robot_type or features differ. Both use 30 fps, so_follower, birdseye + claw at 640x480.
    lerobot-edit-dataset --new_repo_id "$COTRAIN_DATASET" \
      --operation.type merge --operation.repo_ids "['$SIM_DATASET', '$REAL']" ;;
  pi05-smoke)
    lerobot-train --dataset.repo_id="$DS" "${PI05[@]}" --policy.push_to_hub=false \
      --output_dir=$OUT/pi05_sim_smoke --job_name=pi05_sim_smoke --batch_size=8 --steps=200 --save_freq=100 ;;
  pi05-full)
    lerobot-train --dataset.repo_id="$DS" "${PI05[@]}" --policy.repo_id="$SIM_PI05_POLICY" \
      --output_dir=$OUT/pi05_sim_v1 --job_name=pi05_sim_v1 --batch_size=32 --steps=30000 --save_freq=5000 \
      --num_workers=8 --wandb.enable=true ;;
  *) echo "usage: $0 act-smoke | act-full | merge <real_ds> | pi05-smoke [ds] | pi05-full [ds]"; exit 1 ;;
esac
