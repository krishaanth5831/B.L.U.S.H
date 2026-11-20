#!/usr/bin/env python3
"""Post-mortem 2026-11-19: read the overnight diagnostic batch (outputs/diag/runs/<name>/) and print the comparisons.

    python analyse_batch1.py          (system python is enough: numpy only)

Runs (all on the 100k checkpoint, policy only unless said): base_a / base_b = the same 50 held-out scenes twice;
teacher / train_teacher = the teacher driving (held-out / 50 clean training scenes); train_clean = the policy on 50
clean TRAINING scenes; cue_* / img_* = first 25 held-out scenes with the cue or one camera image moved.
"""
import collections
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"


def load(name):
    f = sorted(glob.glob(str(RUNS / name / "act_*.jsonl")))[-1]
    rows = [json.loads(ln) for ln in open(f) if ln.strip()]
    return {r["seed"]: r for r in rows if not r.get("summary") and not r.get("hold")}


def trace(name, times=(1.0, 2.0, 3.3, 4.0, 4.5, 5.0, 6.0)):
    """jaw position (mm) per seed at a few clock times."""
    out = collections.defaultdict(dict)
    want = {round(t, 2): t for t in times}
    with open(RUNS / name / "trace.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            t = round(r["t"], 1)
            if abs(r["t"] - t) < 0.02 and t in want and t not in out[r["seed"]]:
                out[r["seed"]][t] = (np.array(r["jaw_mm"]), r.get("stalk_to_jaw_mm"), r.get("cut"))
    return out


def q(a, p):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


def cls(r):
    if r["success"]:
        return "success"
    if r["cut"] and not r["in_crate"]:
        return "cut, lost"
    if r["cut"]:
        return "cut, other"
    return "no cut"


def line(name, R):
    n = len(R)
    return (f"{name:16s} n {n:3d}  success {sum(r['success'] for r in R.values()):2d}  cut {sum(bool(r['cut']) for r in R.values()):2d}  "
            f"in crate {sum(bool(r['in_crate']) for r in R.values()):2d}  closest {q([r['closest_live_mm'] for r in R.values()], 50):5.1f} mm "
            f"(25-75 %: {q([r['closest_live_mm'] for r in R.values()], 25):.1f}-{q([r['closest_live_mm'] for r in R.values()], 75):.1f})  "
            f"fruit knocked off in {sum(bool(r['fruit_off']) for r in R.values()):2d}  neighbours moved {q([r['neighbours_moved_mm'] for r in R.values()], 50):.1f} mm")


def main():
    names = ["base_a", "base_b", "teacher", "train_clean", "train_teacher", "cue_both_u30", "cue_num_u30", "cue_ovl_u30",
             "cue_both_u100", "cue_both_v-30", "img_base_u30", "img_claw_u30"]
    R = {n: load(n) for n in names}
    print("1. All runs")
    for n in names:
        print("  ", line(n, R[n]))
    first25 = sorted(R["base_a"])[:25]
    for n in ("base_a", "base_b"):
        print("  ", line(n + " first 25", {s: R[n][s] for s in first25}))

    print("\n2. The same test twice (base_a vs base_b, 50 held-out scenes)")
    a, b = R["base_a"], R["base_b"]
    both = sum(a[s]["success"] and b[s]["success"] for s in a)
    print(f"   success in both {both}, only first {sum(a[s]['success'] and not b[s]['success'] for s in a)}, "
          f"only second {sum(b[s]['success'] and not a[s]['success'] for s in a)}; cut in both "
          f"{sum(bool(a[s]['cut']) and bool(b[s]['cut']) for s in a)}, only first {sum(bool(a[s]['cut']) and not b[s]['cut'] for s in a)}, "
          f"only second {sum(bool(b[s]['cut']) and not a[s]['cut'] for s in a)}")
    d = [abs(a[s]["closest_live_mm"] - b[s]["closest_live_mm"]) for s in a]
    print(f"   closest approach differs by {q(d, 50):.1f} mm (median; 90 %: {q(d, 90):.1f})")
    ta, tb = trace("base_a"), trace("base_b")
    for t in (1.0, 2.0, 3.3, 4.0, 4.5, 5.0):
        dd = [np.linalg.norm(ta[s][t][0] - tb[s][t][0]) for s in a if t in ta[s] and t in tb[s]]
        print(f"   jaw position at {t:.1f} s differs by {q(dd, 50):5.2f} mm (90 %: {q(dd, 90):5.1f}, max {max(dd):5.1f})")
    dc = [np.linalg.norm(np.array(a[s]["aim_close"]["jaw_mm"]) - np.array(b[s]["aim_close"]["jaw_mm"]))
          for s in a if a[s].get("aim_close") and b[s].get("aim_close")]
    print(f"   jaw position when the jaw shuts differs by {q(dc, 50):.1f} mm (90 %: {q(dc, 90):.1f})")

    print("\n3. Where the jaw is when it shuts, policy (base_a + base_b pooled) vs teacher; offset = stalk cut point - jaw")
    pool = [r for n in ("base_a", "base_b") for r in R[n].values()]
    by = collections.defaultdict(list)
    for r in pool:
        by[cls(r)].append(r)
    print("   gripper frame: x = closing direction, y = along the stalk, z = along the fingers, + = toward the gripper body (the fingers run along -z)")
    for k in ("success", "cut, lost", "cut, other", "no cut"):
        v = [r for r in by[k] if r.get("aim_close")]
        if not v:
            continue
        og = np.array([r["aim_close"]["off_grip_mm"] for r in v])
        ow = np.array([r["aim_close"]["off_world_mm"] for r in v])
        t = [r["aim_close"]["t"] for r in v]
        print(f"   {k:11s} n {len(by[k]):3d}: shuts at {q(t, 50):.2f} s; distance {q(np.linalg.norm(ow, axis=1), 50):5.1f} mm "
              f"(25-75 %: {q(np.linalg.norm(ow, axis=1), 25):.1f}-{q(np.linalg.norm(ow, axis=1), 75):.1f}); "
              f"across the stalk (x,z only) {q(np.linalg.norm(og[:, [0, 2]], axis=1), 50):5.1f} mm; "
              f"median offset gripper frame {np.median(og, axis=0).round(1)}, world {np.median(ow, axis=0).round(1)}")
    v = [r for r in pool if r.get("aim_close")]
    ow = np.array([r["aim_close"]["off_world_mm"] for r in v])
    og = np.array([r["aim_close"]["off_grip_mm"] for r in v])
    print(f"   all {len(v)} closings: world offset mean {ow.mean(0).round(1)} std {ow.std(0).round(1)} median {np.median(ow, axis=0).round(1)}; "
          f"gripper frame mean {og.mean(0).round(1)} std {og.std(0).round(1)}")
    for n in ("teacher", "train_teacher"):
        v = [r for r in R[n].values() if r.get("aim_close")]
        ow = np.array([r["aim_close"]["off_world_mm"] for r in v])
        oc = np.array([r["aim_cut"]["off_world_mm"] for r in R[n].values() if r.get("aim_cut")])
        t = [r["aim_close"]["t"] for r in v]
        print(f"   {n}: shuts at {q(t, 50):.2f} s (5-95 %: {q(t, 5):.2f}-{q(t, 95):.2f}); distance at shutting "
              f"{q(np.linalg.norm(ow, axis=1), 50):.1f} mm (90 %: {q(np.linalg.norm(ow, axis=1), 90):.1f}); just before the cut "
              f"{q(np.linalg.norm(oc, axis=1), 50):.1f} mm (90 %: {q(np.linalg.norm(oc, axis=1), 90):.1f})")
        bad = [r for r in R[n].values() if not r["success"]]
        print(f"      teacher failures {len(bad)}: " + "; ".join(
            f"{r['seed']} {r['target']} fruit {r.get('n_fruit')} cut {bool(r['cut'])} crate {bool(r['in_crate'])} "
            f"off {len(r['fruit_off'])} wrong {len(r['wrong_cut'])}" for r in bad))
    # success vs distance across the stalk at shutting
    print("   policy, by distance from the stalk when the jaw shuts (base_a + base_b + train_clean):")
    pool2 = pool + list(R["train_clean"].values())
    edges = [0, 4, 8, 12, 16, 24, 40, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = [r for r in pool2 if r.get("aim_close") and lo <= np.linalg.norm(r["aim_close"]["off_world_mm"]) < hi]
        if v:
            print(f"      {lo:>3.0f}-{hi if hi < 1e8 else 999:<4.0f} mm: n {len(v):3d}, cut {sum(bool(r['cut']) for r in v):3d}, "
                  f"success {sum(r['success'] for r in v):3d}")

    print("\n4. Policy on scenes it was TRAINED on (train_clean) vs held-out (base_a + base_b)")
    tc = R["train_clean"]
    print("   training scenes by outcome:", dict(collections.Counter(cls(r) for r in tc.values())))
    print("   held-out (two runs pooled) :", dict(collections.Counter(cls(r) for r in pool)))
    for nm, rows in (("training", list(tc.values())), ("held-out", pool)):
        v = [np.linalg.norm(r["aim_close"]["off_world_mm"]) for r in rows if r.get("aim_close")]
        print(f"   {nm}: distance from the stalk when the jaw shuts {q(v, 50):.1f} mm (25-75 %: {q(v, 25):.1f}-{q(v, 75):.1f}); "
              f"closest at any time {q([r['closest_live_mm'] for r in rows], 50):.1f} mm")
    sl = collections.defaultdict(lambda: [0, 0, 0])
    for r in tc.values():
        sl[r["target"]][0] += 1
        sl[r["target"]][1] += bool(r["success"])
        sl[r["target"]][2] += bool(r["cut"])
    print("   training scenes by slot (n/success/cut): " + ", ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in sorted(sl.items())))

    print("\n5. Moving the cue or an image by 30 px: does the jaw follow? (first 25 held-out scenes, paired with base_a)")
    tr = {n: trace(n) for n in ("cue_both_u30", "cue_num_u30", "cue_ovl_u30", "cue_both_u100", "cue_both_v-30",
                                "img_base_u30", "img_claw_u30")}
    tr["base_b"] = tb
    for n in ("base_b", "cue_both_u30", "cue_num_u30", "cue_ovl_u30", "cue_both_u100", "cue_both_v-30", "img_base_u30",
              "img_claw_u30"):
        seeds = [s for s in first25 if s in R[n]]
        dc = np.array([np.array(R[n][s]["aim_close"]["jaw_mm"]) - np.array(a[s]["aim_close"]["jaw_mm"])
                       for s in seeds if R[n][s].get("aim_close") and a[s].get("aim_close")])
        d33 = np.array([tr[n][s][3.3][0] - ta[s][3.3][0] for s in seeds if 3.3 in tr[n][s] and 3.3 in ta[s]])
        d45 = np.array([tr[n][s][4.5][0] - ta[s][4.5][0] for s in seeds if 4.5 in tr[n][s] and 4.5 in ta[s]])
        print(f"   {n:14s} vs base_a: jaw at 3.3 s moved {q(np.linalg.norm(d33, axis=1), 50):5.1f} mm, at 4.5 s "
              f"{q(np.linalg.norm(d45, axis=1), 50):5.1f} mm, when it shuts {q(np.linalg.norm(dc, axis=1), 50):5.1f} mm "
              f"(90 %: {q(np.linalg.norm(dc, axis=1), 90):5.1f}); median move xyz at shutting {np.median(dc, axis=0).round(1)}; "
              f"success {sum(R[n][s]['success'] for s in seeds)}/{len(seeds)}, cut {sum(bool(R[n][s]['cut']) for s in seeds)}")


if __name__ == "__main__":
    main()
