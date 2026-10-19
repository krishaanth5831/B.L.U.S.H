#!/usr/bin/env bash
# rollout_sim.sh: run a trained policy on the sim arm (lerobot env). Same modes as harvest/scripts/rollout.sh.
#   sim/scripts/rollout_sim.sh eval <policy> <name>   20 saved episodes with resets (press N between them)
#   sim/scripts/rollout_sim.sh run  <policy>          one 40 s run, nothing saved
#   sim/scripts/rollout_sim.sh pi05 <policy>          π0.5 through async inference on the rented GPU
#                                                     (policy_server + ssh tunnel first, see the vault note)
# ⚠️ The overlay flag must match what the policy was trained with (none until step 4 exists).
# The terminal prints "✅ target in the crate" per scene: that's the sim success signal.
set -euo pipefail
source "$(dirname "$0")/common_sim.sh"
MODE=${1:?eval|run|pi05}; POLICY=${2:?policy repo}
ROBOT=("${SIM_ROBOT_FLAGS[@]}" --robot.snap_to_first_action=false)
case "$MODE" in
  eval)
    lerobot-rollout --strategy.type=episodic "${ROBOT[@]}" \
      --policy.path="$POLICY" --task="$TASK" \
      --dataset.repo_id="$HF_USER/rollout_sim_${3:?eval name, e.g. act_sim_v1_40k}" --dataset.no_stamp=true \
      --dataset.single_task="$TASK" --dataset.num_episodes=20 \
      --dataset.episode_time_s=40 --dataset.reset_time_s=20 \
      --dataset.rgb_encoder.vcodec="$VCODEC" --dataset.push_to_hub=false \
      --display_data=true ;;
  run)
    lerobot-rollout --strategy.type=base "${ROBOT[@]}" \
      --policy.path="$POLICY" --task="$TASK" --duration=40 --display_data=true ;;
  pi05)
    python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 \
      "${ROBOT[@]}" --task="$TASK" \
      --policy_type=pi05 --pretrained_name_or_path="$POLICY" --policy_device=cuda \
      --actions_per_chunk=50 --chunk_size_threshold=0.5 --fps="$CAM_FPS" ;;
  *) echo "usage: $0 eval <policy> <name> | run <policy> | pi05 <policy>"; exit 1 ;;
esac
