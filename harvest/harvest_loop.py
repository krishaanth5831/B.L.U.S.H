"""harvest_loop.py: gated harvest, v1. The ripeness gate decides, lerobot-rollout runs one
policy episode. One trial per run. Follow harvest/plan/system_trials.csv. Run from the repo
root in an env that has both lerobot and ultralytics, or split the gate into its own process.

    python harvest/harvest_loop.py <policy_repo>

Per trial:
  1. read a few claw frames at the approach pose (lets exposure settle), gate the last one
  2. RELEASE the camera, because the rollout process needs it
  3. target found -> one `harvest/scripts/rollout.sh run`. No target -> log `skip`
  4. append the decision + outcome to harvest/logs/harvest_log.csv

One process per trial is slow, but it's the exact rollout path the policy was evaluated with,
so the numbers stay comparable. Move to an in-process loop only once v1 numbers exist.
"""
import csv
import datetime as dt
import pathlib
import subprocess
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ripeness.gate.pick_target import Gate  # noqa: E402

CLAW_CAM = 2          # config/hardware.env; recheck per boot
SETTLE_FRAMES = 15    # flush the buffer + let exposure settle (vault note 06)
LOG = ROOT / "harvest/logs/harvest_log.csv"
FRAMES = ROOT / "harvest/gate_frames"   # gitignored. One jpg per trial, for checking wrong calls.
OUTCOMES = ("success", "miss_approach", "bad_pick", "drop", "collision", "wrong_target")

policy = sys.argv[1]
gate = Gate()


def look():
    cap = cv2.VideoCapture(CLAW_CAM)
    ok, f = False, None
    for _ in range(SETTLE_FRAMES):
        ok, f = cap.read()
    cap.release()
    return f if ok else None


trial = int(input("trial number from system_trials.csv: "))
truth = input("ground truth for the target fruit (ripe/turning/unripe): ").strip()
frame = look()
if frame is None:
    sys.exit("claw camera read failed")
FRAMES.mkdir(exist_ok=True)
cv2.imwrite(str(FRAMES / f"trial_{trial:03d}.jpg"), frame)

target = gate(frame)
action = "pick" if target else "skip"
print(f"gate: {action}" + (f"  conf={target['conf']:.2f}" if target else ""))
if action == "pick":
    subprocess.run([str(ROOT / "harvest/scripts/rollout.sh"), "run", policy], check=False)
    outcome = ""
    while outcome not in OUTCOMES:
        outcome = input(f"outcome {OUTCOMES}: ").strip()
else:
    outcome = "skipped"

with open(LOG, "a", newline="") as fh:
    csv.writer(fh).writerow([dt.datetime.now().isoformat(timespec="seconds"), trial, truth, action,
                             f"{target['conf']:.3f}" if target else "", policy, outcome, ""])
