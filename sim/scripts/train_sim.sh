#!/usr/bin/env bash
# train_sim.sh: train on sim data, or on sim + real merged (lerobot env). Same recipes as harvest/scripts.
#   sim/scripts/train_sim.sh act-smoke [ds]   500 ACT steps on the laptop: data loads, both cameras, loss falls
#   sim/scripts/train_sim.sh act-truss [ds]   ACT + box numbers + temporal ensembling on the rig "base" truss episodes (100k)
#   sim/scripts/train_sim.sh act-teacher <ds> ACT on teacher-v3 data, augmentation on (STEPS, SAVE, RUN env vars)
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
    DS=${2:-$HF_USER/blush_sim_truss_v2}
    lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --batch_size=8 --steps=500 --save_freq=500 \
      --output_dir=$OUT/act_sim_smoke --job_name=act_sim_smoke --wandb.enable=false ;;
  act-truss)
    # ACT on the rig "base" truss episodes (K7 2026-11-08: "train this with act policy... we will not be using smolvla
    # this time"). The recipe of the best single-fruit ACT (act-box, 2026-11-01: ensembling cut the median miss from 37 to
    # 23 mm): base + claw cameras, the target box + cut point as numbers (observation.environment_state, read by ACT
    # automatically) and temporal ensembling 0.01. Batch 8,
    # 100k steps, a checkpoint every 10k (pick by graded rollouts, not loss).
    DS=${2:-$HF_USER/blush_sim_truss_v2}
    lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 \
      --batch_size=${BATCH:-8} --steps=${STEPS:-100000} --save_freq=10000 --log_freq=50 --num_workers=8 \
      --output_dir=$OUT/act_truss_v1${RUN_SUFFIX:-} --job_name=act_truss_v1 --wandb.enable=false ;;
  act-teacher)
    # The post-mortem recipe (§6.3, vault, 2026-11-11) on teacher-v3 data: ACT defaults (chunk 100, batch 8, lr 1e-5,
    # the box + cut point as numbers through observation.environment_state), lerobot's image augmentation on (colour,
    # sharpness, RandomAffine +-5 deg / 5 % shift). How chunks are executed (ensembling / full chunk / every 25) is set
    # at evaluation (eval_truss.py --ensemble / --n-action-steps), not here. Step 1 (one layout): STEPS=20000.
    # Saves every 5k: pick the checkpoint by rollouts (sim/pick_checkpoint.py), never "last". Step 1 went
    # 74 % at 20k, 95 % at 40k, 5 % at 50k with the training loss flat (post-mortem section 6.9).
    # Learning rate (step 3, 2026-11-14): that drift past the peak happened at a constant 1e-5 (S.H.E.A.R's too:
    # 45k 18 of 19, 50k 8 of 19), so the default is now a cosine decay from 1e-5 to LR_FLOOR (1e-6) at STEPS, after
    # WARMUP steps (lerobot's cosine_decay_with_warmup; LR_SCHED=constant for the old recipe). To a floor, not to
    # zero: 100k steps are only 2.9 passes over dataset v3 (step 1 peaked at 14), so a run that is still improving
    # at the end can be resumed with more steps at the floor rate. ACT's preset is AdamW lr 1e-5, weight decay 1e-4
    # and no scheduler (configuration_act.py), so the same optimiser is spelled out. Without the preset lerobot
    # passes policy.parameters() as one group instead of ACT's two (backbone apart): the same thing, since
    # optimizer_lr_backbone is 1e-5 too. Warmup 1000 = pi0's value (guess, unverified). EMA=1 adds lerobot's weight
    # average (pretrained_model_ema per checkpoint); AMP=1 trains in mixed precision.
    #   RUN=act_teacher_v3 sim/scripts/train_sim.sh act-teacher $HF_USER/blush_sim_teacher_v3     step 3
    DS=${2:?dataset, e.g. $HF_USER/blush_sim_teacher_onelayout_v1}
    SCHED=()
    if [[ "${LR_SCHED:-cosine}" == cosine ]]; then
      SCHED=(--use_policy_training_preset=false --optimizer.type=adamw --optimizer.lr=1e-5 --optimizer.weight_decay=1e-4
             --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr=1e-5 --scheduler.decay_lr=${LR_FLOOR:-1e-6}
             --scheduler.num_warmup_steps=${WARMUP:-1000} --scheduler.num_decay_steps=${STEPS:-100000})
    fi
    [[ -n "${EMA:-}" ]] && SCHED+=(--ema.enable=true)
    [[ -n "${AMP:-}" ]] && SCHED+=(--policy.use_amp=true)
    lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --dataset.image_transforms.enable=true "${SCHED[@]}" \
      --batch_size=${BATCH:-8} --steps=${STEPS:-100000} --save_freq=${SAVE:-5000} --log_freq=100 --num_workers=8 \
      --output_dir=$OUT/${RUN:-act_teacher_v1} --job_name=${RUN:-act_teacher_v1} --wandb.enable=false ;;
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
  *) echo "usage: $0 act-smoke [ds] | act-truss [ds] | act-teacher <ds> | act-full | merge <real_ds> | pi05-smoke [ds] | pi05-full [ds]"; exit 1 ;;
esac
