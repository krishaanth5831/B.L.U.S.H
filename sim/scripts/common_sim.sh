# common_sim.sh: sourced by the sim scripts. Same idea as harvest/scripts/common.sh, with the real
# follower swapped for the MuJoCo one (--robot.type=so101_mujoco, from sim/plugin). The leader is real.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../config/hardware.env"
source "$HERE/../../harvest/harvest.env"     # TASK, timings: identical strings to the real recordings

# Sim datasets get their own names, so sim and real never mix by accident. Same robot_type
# (so_follower) and features as real, so they can be merged on purpose for co-training.
SIM_DATASET=$HF_USER/blush_sim_pick_v1     # placeholder, lock before the first real session
COTRAIN_DATASET=$HF_USER/blush_cotrain_v1  # sim + real merged (train_sim.sh merge)
SIM_ACT_POLICY=$HF_USER/act_blush_sim_v1
SIM_PI05_POLICY=$HF_USER/pi05_blush_sim_v1

# A fresh random first layout per session unless SEED=... is given. Each N in the MuJoCo window
# moves to the next seed. Seeds + full sim state land in ~/.cache/blush_sim/states/<session>/.
SEED=${SEED:-$RANDOM}

SIM_ROBOT_FLAGS=(
  --robot.type=so101_mujoco
  --robot.id=sim
  --robot.seed="$SEED"
  --robot.use_degrees="$USE_DEGREES"
  --robot.overlay="${OVERLAY:-yolo}"   # birdseye target box (sim/target.py); OVERLAY=none for raw frames
)
TELEOP_FLAGS=(
  --teleop.type=so101_leader
  --teleop.port="$LEADER_PORT"
  --teleop.id="$LEADER_ID"
  --teleop.use_degrees="$USE_DEGREES"
)

need_leader() {
  [[ -n "${DRYRUN:-}" ]] && return 0
  [[ -e "$LEADER_PORT" ]] || { echo "⚠️ leader not found: $LEADER_PORT (plugged in?)"; exit 1; }
}
# DRYRUN=1 sim/scripts/<script>.sh ...  prints the lerobot command instead of running it.
if [[ -n "${DRYRUN:-}" ]]; then
  for c in lerobot-teleoperate lerobot-record lerobot-replay lerobot-train lerobot-edit-dataset lerobot-rollout; do
    eval "$c() { printf 'CMD %s' $c; printf ' %q' \"\$@\"; echo; }"
  done
  python() { if [[ "$1 $2" == "-m lerobot.async_inference.robot_client" ]]; then shift 2; printf 'CMD robot_client'; printf ' %q' "$@"; echo; else command python "$@"; fi; }
fi
python -c "import lerobot_robot_so101_mujoco" 2>/dev/null || {
  echo "⚠️ sim plugin not installed in this env: conda activate lerobot && pip install -e $HERE/../plugin"; exit 1; }
echo "sim seed $SEED"
