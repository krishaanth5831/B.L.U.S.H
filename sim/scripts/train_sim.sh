#!/usr/bin/env bash
# train_sim.sh: train on sim data, or on sim + real merged (lerobot env). Same recipes as harvest/scripts.
#   sim/scripts/train_sim.sh act-smoke        500 ACT steps on the laptop: data loads, both cameras, loss falls
#   sim/scripts/train_sim.sh act-laptop [ds] ACT on the scripted episodes on the laptop GPU (100k steps)
#   sim/scripts/train_sim.sh act-box [ds]    act-laptop + the box as numbers + temporal ensembling (v1_box)
#   sim/scripts/train_sim.sh smolvla-truss [ds] SmolVLA + LoRA on the truss episodes, laptop GPU (20k steps)
#   sim/scripts/train_sim.sh smolvla-truss-expert [ds] SmolVLA, whole action expert trained, 60k steps (laptop)
#   sim/scripts/train_sim.sh smolvla-truss-expert-v2 [ds] the same on the 300 episodes (200 scripted + 100 DAgger)
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
  act-laptop)
    # ACT on the scripted sim episodes, on the laptop's 6 GB RTX 1000 Ada (K7, 2026-10-31). Batch 8 like
    # act-smoke; 100k steps is a guess (unverified), checkpoints every 10k so any of them can be tried.
    DS=${2:-$HF_USER/blush_sim_scripted_v1}
    lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=50 --num_workers=8 \
      --output_dir=$OUT/act_scripted_v1 --job_name=act_scripted_v1 --wandb.enable=false ;;
  act-box)
    # act-laptop + the target box as 4 numbers (observation.environment_state, sim/add_box_feature.py) and
    # ACT's temporal ensembling (paper's 0.01: re-plan every step, average the overlapping chunks) baked into
    # the checkpoint config. K7, 2026-11-01, after sim/analyze_policy.py: the v1 policy closed a median 37 mm
    # from new targets (23 mm with ensembling); neighbours hang 21-30 mm apart.
    DS=${2:-$HF_USER/blush_sim_scripted_v1_box}
    lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 \
      --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=50 --num_workers=8 \
      --output_dir=$OUT/act_box_v1 --job_name=act_box_v1 --wandb.enable=false ;;
  smolvla-truss)
    # SmolVLA (pretrained lerobot/smolvla_base) + LoRA on the 200 truss-harvesting episodes, laptop GPU (K7,
    # 2026-11-04). Recipe = S.H.E.A.R's shear_smolvla_lora_v0 run, which fits the 6 GB RTX 1000 at batch 16.
    # Cameras renamed to the base model's camera1/camera2. ⚠️ SmolVLA doesn't read observation.environment_state
    # (the box + cut point numbers): it sees the green box and cross drawn into the birdseye frames instead.
    DS=${2:-$HF_USER/blush_sim_truss_v1}
    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id="$DS" --policy.device=cuda \
      --policy.push_to_hub=false \
      --rename_map='{"observation.images.birdseye": "observation.images.camera1", "observation.images.claw": "observation.images.camera2"}' \
      --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=64 \
      --batch_size=16 --steps=20000 --save_freq=2000 --log_freq=100 --num_workers=8 \
      --output_dir=$OUT/smolvla_truss_lora_v1 --job_name=smolvla_truss_lora_v1 --wandb.enable=false ;;
  smolvla-truss-expert)
    # SmolVLA with its whole action expert trained (no LoRA; the VLM stays frozen, SmolVLA's default
    # train_expert_only / freeze_vision_encoder), 60k steps (K7, 2026-11-04: the LoRA run reached for the truss but
    # never cut it). Batch 8: S.H.E.A.R's expert run on this 6 GB laptop GPU uses 8 (BATCH= overrides). The LR
    # schedule decays over the whole run (SmolVLA's default decays over 30k).
    DS=${2:-$HF_USER/blush_sim_truss_v1}
    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id="$DS" --policy.device=cuda \
      --policy.push_to_hub=false --policy.scheduler_decay_steps=60000 \
      --rename_map='{"observation.images.birdseye": "observation.images.camera1", "observation.images.claw": "observation.images.camera2"}' \
      --batch_size=${BATCH:-8} --steps=60000 --save_freq=5000 --log_freq=100 --num_workers=8 \
      --output_dir=$OUT/smolvla_truss_expert_v1${RUN_SUFFIX:-} --job_name=smolvla_truss_expert_v1 --wandb.enable=false ;;
  smolvla-truss-expert-v2)
    # The same recipe on the dataset after DAgger: the 200 scripted episodes + 100 where smolvla_truss_expert_v1
    # drove until it veered off and the scripted expert took over (sim/dagger_truss.py, 2026-11-05/06; K7: "retrain
    # smolvla on all 300 episodes with 60k steps"). Trained from smolvla_base again, not from v1 (one clean run).
    DS=${2:-$HF_USER/blush_sim_truss_v1}
    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id="$DS" --policy.device=cuda \
      --policy.push_to_hub=false --policy.scheduler_decay_steps=60000 \
      --rename_map='{"observation.images.birdseye": "observation.images.camera1", "observation.images.claw": "observation.images.camera2"}' \
      --batch_size=${BATCH:-8} --steps=60000 --save_freq=5000 --log_freq=100 --num_workers=8 \
      --output_dir=$OUT/smolvla_truss_expert_v2${RUN_SUFFIX:-} --job_name=smolvla_truss_expert_v2 --wandb.enable=false ;;
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
  *) echo "usage: $0 act-smoke | act-laptop [ds] | act-box [ds] | act-full | merge <real_ds> | pi05-smoke [ds] | pi05-full [ds]"; exit 1 ;;
esac
