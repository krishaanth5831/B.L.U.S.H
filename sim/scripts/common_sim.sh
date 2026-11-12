# shellcheck shell=bash
# shellcheck disable=SC2034  # the names set here are read by the scripts that source this file
# common_sim.sh: sourced by the sim scripts. Same idea as harvest/scripts/common.sh, with the real
# follower swapped for the MuJoCo one (--robot.type=so101_mujoco, from sim/plugin). The leader is real.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../config/hardware.env"
source "$HERE/../../harvest/harvest.env"     # episode / reset timings, as the real recordings
# The task string: generate_truss.py's TASK (the scripted truss episodes'), read from there so it stays one string.
# ⚠️ Not harvest.env's TASK: that's the real rig's single-fruit placeholder.
SIM_TASK=$(sed -n 's/^TASK = "\(.*\)"$/\1/p' "$HERE/../generate_truss.py")
[[ -n "$SIM_TASK" ]] || { echo "⚠️ no TASK = \"...\" line in sim/generate_truss.py"; exit 1; }

# Sim datasets get their own names, so sim and real never mix by accident. Same robot_type
# (so_follower) and features as real, so they can be merged on purpose for co-training.
# These wrappers run the truss scene (the plugin's only task since 2026-11-08), rig "birdseye" unless RIG=base.
# ⚠️ lerobot-record doesn't record observation.environment_state (the box + cut point as numbers) the way
#    generate_truss.py does; a policy trained with them can't be fed by these wrappers' recordings as is.
SIM_DATASET=$HF_USER/blush_sim_pick_v1     # placeholder, never recorded: truss teleop demos
COTRAIN_DATASET=$HF_USER/blush_cotrain_v1  # sim + real merged (train_sim.sh merge)
SIM_ACT_POLICY=$HF_USER/act_blush_sim_v1
SIM_PI05_POLICY=$HF_USER/pi05_blush_sim_v1

# A fresh random first layout per session unless SEED=... is given. Each N in the MuJoCo window
# moves to the next seed. Seeds + full sim state land in ~/.cache/blush_sim/states/<session>/.
SEED=${SEED:-$RANDOM}
# The start-frame target: the truss YOLO if trained (YOLO_WEIGHTS=... for another one), else the sim's true box.
YOLO_WEIGHTS=${YOLO_WEIGHTS:-$HERE/../../runs/yolo_truss/truss_v1/weights/best.pt}   # = generate_truss.YOLO_TRUSS
if [[ -f "$YOLO_WEIGHTS" ]]; then OVERLAY=${OVERLAY:-yolo}; else OVERLAY=${OVERLAY:-true_box}; fi
[[ "$OVERLAY" == yolo && ! -f "$YOLO_WEIGHTS" ]] && { echo "⚠️ OVERLAY=yolo but no truss YOLO at $YOLO_WEIGHTS"; exit 1; }

SIM_ROBOT_FLAGS=(
  --robot.type=so101_mujoco
  --robot.id=sim
  --robot.seed="$SEED"
  --robot.use_degrees="$USE_DEGREES"
  --robot.rig="${RIG:-birdseye}"       # birdseye + claw, or RIG=base: base + claw (2026-11-06)
  --robot.overlay="$OVERLAY"           # target box + cut point (sim/truss_target.py); OVERLAY=none for raw frames
)
[[ "$OVERLAY" == yolo ]] && SIM_ROBOT_FLAGS+=(--robot.yolo_weights="$YOLO_WEIGHTS")
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
