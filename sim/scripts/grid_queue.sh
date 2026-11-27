#!/usr/bin/env bash
# grid_queue.sh TAG FIRST LAST ENVJSON-or-none "extra base_grid.py args": the teacher on a seed range in 15-scene
# chunks, each a systemd unit blush-v6-TAG_<first seed> under the same memory rule as sim_queue.sh. Resumable: seeds
# already in outputs/v6/base_grid_TAG_*.jsonl are not run again. Compare runs with sim/scripts/grid_ab.py.
#   MAXJ=5 sim/scripts/grid_queue.sh r38_TRIO 9730000 9730089 none \
#     "--jaw v5 --base 0.37,0.20 --old-start --world crate23 --rig trio --layout-rig base"
# ⚠️ A unit imports the code when it starts: change sim code between two chunks of one tag and the tag is mixed.
R=$(cd "$(dirname "$0")/../.." && pwd)
TAG=${1:?tag}; FIRST=${2:?first seed}; LAST=${3:?last seed}; ENVJ=${4:?BLUSH_V6 json or none}; EXTRA=${5:-}
MAXJ=${MAXJ:-4}          # as sim_queue.sh
if [ "$ENVJ" = "none" ]; then SETENV="unset BLUSH_V6"; else SETENV="export BLUSH_V6='$ENVJ'"; fi
active() { systemctl --user list-units 'blush-v6-*' --state=active --no-legend | wc -l; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
wait_slot() { while [ "$(active)" -ge "$MAXJ" ] || [ "$(avail_gb)" -lt 13 ]; do sleep 30; done; }
todo() {
  python3 - "$1" "$2" "$3" <<'PY'
import glob, json, sys
tag, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
done = set()
for f in glob.glob(f"outputs/v6/base_grid_{tag}_[0-9]*.jsonl"):   # ⚠️ not _*: a tag can be the prefix of another
    for l in open(f):
        if l.strip():
            try:
                r = json.loads(l)
            except ValueError:      # a line cut short by a stopped job
                continue
            if "seed" in r:
                done.add(r["seed"])
print(",".join(str(s) for s in range(a, b + 1) if s not in done))
PY
}
cd "$R" || exit 1
CHUNK=${CHUNK:-15}
for st in $(seq "$FIRST" "$CHUNK" "$LAST"); do
  en=$((st + CHUNK - 1)); [ "$en" -gt "$LAST" ] && en=$LAST
  seeds=$(todo "$TAG" "$st" "$en")
  [ -z "$seeds" ] && continue
  wait_slot
  tag=${TAG}_${st}
  systemd-run --user --unit="blush-v6-$tag" --collect -q -p MemoryMax=7G -p MemorySwapMax=0 systemd-inhibit --what=sleep:idle --why="B.L.U.S.H v6 $TAG" bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-mj4 && cd $R/sim && export MUJOCO_GL=egl OMP_NUM_THREADS=2 && $SETENV && python -u base_grid.py --retries 6 $EXTRA --seeds $seeds --out $R/outputs/v6/base_grid_$tag.jsonl > $R/outputs/v6/base_grid_$tag.log 2>&1"
  echo "started $tag $(date +%H:%M) avail $(avail_gb) GB"; sleep 30
done
while systemctl --user list-units "blush-v6-${TAG}_*" --state=active --no-legend | grep -q .; do sleep 60; done
echo "all done $(date +%H:%M)"
