#!/usr/bin/env python3
"""Post-mortem 2026-11-20: where the stalk is, seen from the jaw, at the moment the jaw first shuts.

    python where_at_shut.py                 the policy as trained, the teacher, the corrected-inference tests
    python where_at_shut.py --first50       the 100-scene tests cut down to the first 50 held-out scenes
                                            (numpy only; reads the eval rows' aim_close, no simulation)

eval_truss.py stores per run aim_close = the first frame in which the jaw reading drops below 10 after it was above
20, with off_grip_mm = (stalk's cut point as it hangs now) - (jaw's cut point), in the gripper's frame.

⚠️ The gripper frame, read off the model (scene.compose(arm=True, cut_hold=True), geom positions, 2026-11-20):
    x  the closing direction. Fixed pad face at x = -3.0 mm, the moving pad at x > 0. The jaw's cut point sits
       radius + 1 mm inside the fixed pad's face, so an offset below about -3 mm = the stalk's centre line is BEHIND
       the fixed pad's face: the stalk is outside the jaw, on the fixed finger's outer side.
    y  along the stalk (the stalk passes through the jaw along y).
    z  along the fingers, and the fingers run along MINUS z: blade centre at z = -101.5 mm (blade 16 mm long, so
       -109.5 to -93.5), fingertips at -110.5, finger roots near -60. So an offset z > 0 = the stalk is nearer the
       gripper's body than the blade = DEEPER in the jaw than the blade; z < 0 = toward and past the fingertips.
       ⚠️ FINDINGS section 11 first had this sign the wrong way round ("15 beyond the fingertips, 2 deeper");
       corrected on 2026-11-20 after reading the geometry off the model.
"""
import collections
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"
OUTSIDE_X = -3.0     # mm, offset below this = stalk centre behind the fixed pad's face (the exact value is
                     # -(stalk radius + 1 mm) = -3.3 to -3.5 for our 4.5-4.9 mm stalks; truss_ik.cut_point_in_gripper)
OPEN_X = 26.5        # mm, offset above this = beyond the open moving finger: tips ~30 mm apart at the 25 deg opening
                     # (truss_expert.JAW_OPEN_DEG comment), minus the 3.5 mm from the fixed pad's face to the cut point
BLADE_HALF = 8.0     # mm, half the blade's length along the finger (geom arm/blade_fixed, size z = 8 mm)
CLASSES = ("outside the fixed finger", "between, at blade depth", "between, deeper than the blade",
           "between, past the fingertips", "beyond the open moving finger")


def rows(f):
    R = [json.loads(ln) for ln in open(f) if ln.strip()]
    return [r for r in R if not r.get("summary") and not r.get("hold")], [r for r in R if r.get("summary")]


def traced(name):
    fs = sorted(glob.glob(str(RUNS / name / "act_*.jsonl")))
    return rows(fs[-1])[0] if fs else []


def cls(og):
    x, _, z = og
    if x < OUTSIDE_X:
        return CLASSES[0]
    if x > OPEN_X:
        return CLASSES[4]
    return CLASSES[1] if abs(z) <= BLADE_HALF else CLASSES[2] if z > 0 else CLASSES[3]


def show(label, R):
    v = [r for r in R if r.get("aim_close")]
    if not v:
        print(f"\n{label}: no closing recorded in {len(R)} runs")
        return
    og = np.array([r["aim_close"]["off_grip_mm"] for r in v])
    ow = np.array([r["aim_close"]["off_world_mm"] for r in v])
    t = np.array([r["aim_close"]["t"] for r in v])
    d = np.linalg.norm(ow, axis=1)
    print(f"\n{label}: {len(R)} runs, success {sum(bool(r['success']) for r in R)}, cut {sum(bool(r['cut']) for r in R)}, "
          f"in crate {sum(bool(r['in_crate']) for r in R)}")
    print(f"   jaw first shuts at {np.median(t):.2f} s, {np.median(d):.1f} mm from the stalk's cut point (25-75 %: "
          f"{np.percentile(d, 25):.1f}-{np.percentile(d, 75):.1f})")
    print(f"   median offset, gripper frame: sideways {np.median(og[:, 0]):+.1f} mm (25-75 %: {np.percentile(og[:, 0], 25):+.1f} to "
          f"{np.percentile(og[:, 0], 75):+.1f}; - = toward and past the fixed finger), along the stalk {np.median(og[:, 1]):+.1f}, "
          f"depth {np.median(og[:, 2]):+.1f} (25-75 %: {np.percentile(og[:, 2], 25):+.1f} to {np.percentile(og[:, 2], 75):+.1f}; "
          f"+ = deeper in the jaw than the blade)")
    print(f"   median offset, world: {np.median(ow, axis=0).round(1)} (x along the row, y across it, z up; + z = the stalk is above "
          f"the jaw point)")
    by = collections.defaultdict(list)
    for r in v:
        by[cls(r["aim_close"]["off_grip_mm"])].append(r)
    print(f"   {'where the stalk is at the first closing':38s} {'runs':>5s} {'cut':>5s} {'crate':>6s} {'success':>8s}   of the runs without a cut")
    n_nocut = sum(not r["cut"] for r in v)
    for c in CLASSES:
        g = by[c]
        if g:
            print(f"   {c:38s} {len(g):5d} {sum(bool(r['cut']) for r in g):5d} {sum(bool(r['in_crate']) for r in g):6d} "
                  f"{sum(bool(r['success']) for r in g):8d}   {sum(not r['cut'] for r in g):3d} of {n_nocut}")
    out = [r for r in by[CLASSES[0]] if not r["cut"]]
    if out:
        o = np.array([r["aim_close"]["off_grip_mm"] for r in out])
        print(f"   the {len(out)} misses outside the fixed finger: stalk centre {np.median(OUTSIDE_X - o[:, 0]):.0f} mm behind the pad's face "
              f"(median); also deeper than the blade {int((o[:, 2] > BLADE_HALF).sum())}, at blade depth "
              f"{int((np.abs(o[:, 2]) <= BLADE_HALF).sum())}, past the fingertips {int((o[:, 2] < -BLADE_HALF).sum())}")
    deep = [r for r in by[CLASSES[2]] if not r["cut"]]
    if deep:
        z = np.array([r["aim_close"]["off_grip_mm"][2] for r in deep])
        print(f"   the {len(deep)} misses between the fingers but deeper than the blade: {np.median(z):.0f} mm deeper than the blade's "
              f"centre (median; range {z.min():.0f} to {z.max():.0f})")


def main():
    first50 = "--first50" in sys.argv
    base = traced("base_a") + traced("base_b")
    seeds50 = {r["seed"] for r in base}
    show("TEACHER (50 held-out + 50 training scenes)", traced("teacher") + traced("train_teacher"))
    show("POLICY AS TRAINED, 150 runs (base_a + base_b: the first 50 held-out scenes twice; train_clean: 50 training scenes)",
         base + traced("train_clean"))
    show("POLICY AS TRAINED, the first 50 held-out scenes twice (base_a + base_b)", base)
    for n, lab in (("base_c", "policy as trained, traced again 20 Nov"), ("fix_mcdropout", "mean of 16, every plan (traced)"),
                   ("fix_mcdropout2", "mean of 16, from plan 2 (traced)")):
        if traced(n):
            show(f"{lab}: first 50 held-out scenes ({n})", traced(n))
    for f in sorted(glob.glob(str(ROOT / "outputs" / "eval" / "act_teacher_v5_100000_*.jsonl")), key=os.path.getmtime):
        r, s = rows(f)
        if not r or r[0].get("hybrid") or not any(x.get("aim_close") for x in r):
            continue
        tag = s[-1].get("infer") if s else "(still running) " + os.path.basename(f)[-14:-6]
        if not tag:
            if len(r) >= 100:
                show("100-scene test, policy as trained" + (" (first 50 scenes)" if first50 else ""),
                     [x for x in r if x["seed"] in seeds50] if first50 else r)
            continue
        if first50:
            r = [x for x in r if x["seed"] in seeds50]
        show(f"100-scene test, {tag}" + (" (first 50 scenes)" if first50 else ""), r)


if __name__ == "__main__":
    main()
