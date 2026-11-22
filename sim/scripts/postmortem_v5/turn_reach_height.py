#!/usr/bin/env python3
"""Post-mortem 2026-11-20: split the jaw's miss into TURN, REACH and HEIGHT around the shoulder's turning axis, and
follow the three over time. Reads the per-frame traces (eval_truss.py --trace) and the eval rows; no simulation.

    python turn_reach_height.py                         the teacher, the policy as trained, the traced re-runs
    python turn_reach_height.py --group "label" run [run ...]   any traced runs in outputs/diag/runs/ (repeatable)

Why: where_at_shut.py says the stalk is beside the jaw (outside the fixed finger) in most misses. This says which
motion of the arm put it there. Seen from above, with the shoulder_pan axis as the centre:
    turn    how many degrees further round the stalk is than the jaw's cut point (+ = the arm has not turned far
            enough from its start pose, which is at the far left). 1 deg is about 2.8 mm sideways at the stalk.
    reach   how many mm further out from the axis the stalk is than the jaw (+ = the arm has not reached far enough)
    height  how many mm higher the stalk is than the jaw (+ = the jaw is too low)

What it found on 2026-11-20 (base_a + base_b + train_clean, 150 runs; teacher + train_teacher, 100 runs):
    the teacher finishes its turn by 3.0 s (median 0.04 deg left), then reaches in and comes down, and closes at 4.5 s
    the policy still has 5.8 deg left at 3.33 s (when it makes its second plan), and from 3.67 s to the closing the turn
    stays 3.0-3.5 deg short in every bin of "how far the arm had to turn" (45-138 deg). Reach ends within 2.5 mm. The
    jaw ends 12-15 mm too low. The stalk has moved 1.7 mm from where it hung (median) when the jaw closes, so the
    arm is not pushing it away: the arm stops short. With the stretch divided out or the mean of 16 dropout answers
    the height error drops to 2-5 mm and the turn error stays 1.8-2.8 deg.
"""
import argparse
import collections
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"
PAN_XY = np.array([370.0, 238.8])   # mm, the shoulder_pan axis in the world: joint anchor read off the model
                                    # (scene.compose(arm=True), d.xanchor of arm/shoulder_pan, axis vertical; 2026-11-20)
UNIT_DEG = 1.116                    # deg of shoulder_pan per norm unit: joint range +-111.6 deg mapped to +-100 units
                                    # (model jnt_range; arm_map's norm units). Only used to print units as degrees.
LOOKS = (1.0, 2.0, 3.0, 3.33, 3.67, 4.0, 4.5, 5.0, 5.5, 6.0)   # s; 3.33 s = when the second 100-step plan is made
TURN_BINS = ((40, 60), (60, 80), (80, 110), (110, 150))        # deg the arm has to turn from its start pose
DEFAULT = (("TEACHER (50 held-out + 50 training scenes)", ("teacher", "train_teacher")),
           ("POLICY AS TRAINED, 150 runs (base_a, base_b, train_clean)", ("base_a", "base_b", "train_clean")),
           ("policy as trained, traced again on 20 Nov (base_c)", ("base_c",)),
           ("mean of 16 dropout answers, every plan (fix_mcdropout)", ("fix_mcdropout",)),
           ("mean of 16 dropout answers, from plan 2 (fix_mcdropout2)", ("fix_mcdropout2",)))


def rows(name):
    fs = sorted(glob.glob(str(RUNS / name / "act_*.jsonl")))
    if not fs:
        return {}
    return {r["seed"]: r for r in (json.loads(ln) for ln in open(fs[-1]) if ln.strip())
            if not r.get("summary") and not r.get("hold")}


def traces(name):
    out = collections.defaultdict(list)
    f = RUNS / name / "trace.jsonl"
    if not f.exists():
        return out
    with open(f) as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:      # the last line of a run that is still being written
                continue
            out[r["seed"]].append(r)
    return out


def az(p):
    """Angle of a world point around the shoulder axis, deg, 0 = straight at the row (+y), + = toward +x."""
    return float(np.degrees(np.arctan2(p[0] - PAN_XY[0], p[1] - PAN_XY[1])))


def split(jaw, stalk):
    """(turn deg, reach mm, height mm) of the stalk seen from the jaw."""
    jaw, stalk = np.asarray(jaw, float), np.asarray(stalk, float)
    return (az(stalk) - az(jaw), float(np.hypot(*(stalk[:2] - PAN_XY)) - np.hypot(*(jaw[:2] - PAN_XY))),
            float(stalk[2] - jaw[2]))


def frame_at(fr, t, tol=0.03):
    f = min(fr, key=lambda x: abs(x["t"] - t))
    return f if abs(f["t"] - t) <= tol else None


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


def iqr(a):
    return f"{np.percentile(a, 25):+.1f} to {np.percentile(a, 75):+.1f}" if len(a) else "n/a"


def group(label, names):
    T, R = {}, {}
    for n in names:
        for s, fr in traces(n).items():
            T[(n, s)] = fr
        for s, r in rows(n).items():
            R[(n, s)] = r
    T = {k: v for k, v in T.items() if k in R}
    if not T:
        print(f"\n{label}: no finished traced run yet")
        return
    print(f"\n{label}: {len(T)} runs, success {sum(bool(R[k]['success']) for k in T)}, cut {sum(bool(R[k]['cut']) for k in T)}")

    # 1. over time, runs not yet cut
    print("   clock    runs   turn left   reach left   jaw too low   distance   jaw reading"
          + ("   shoulder: command - reading | command" if "cmd" in next(iter(T.values()))[0] else ""))
    for t in LOOKS:
        a, extra = [], []
        for fr in T.values():
            f = frame_at(fr, t)
            if f is None or f["cut"] or f.get("stalk_mm") is None:
                continue
            a.append(split(f["jaw_mm"], f["stalk_mm"]) + (float(np.linalg.norm(np.subtract(f["stalk_mm"], f["jaw_mm"]))),
                                                          f["gripper"]))
            if "cmd" in f:
                extra.append((f["cmd"][0] - f["q"][0], f["cmd"][0]))
        if len(a) < 5:
            continue
        a = np.array(a)
        line = (f"   {t:5.2f} s  {len(a):4d}   {med(a[:, 0]):+6.2f} deg   {med(a[:, 1]):+6.1f} mm   {med(a[:, 2]):+7.1f} mm   "
                f"{med(a[:, 3]):6.1f} mm   {med(a[:, 4]):6.1f}")
        if extra:
            e = np.array(extra)
            line += f"        {med(e[:, 0]):+5.2f} units | {med(e[:, 1]):+6.1f}"
        print(line)

    # 2. at the first closing of the jaw (eval row aim_close), by where the stalk ends up
    D = []
    for k, fr in T.items():
        a = R[k].get("aim_close")
        rest = next((f["stalk_mm"] for f in fr if f.get("stalk_mm") is not None), None)
        start = fr[0]["jaw_mm"]
        if not a or rest is None:
            continue
        jaw = np.array(a["jaw_mm"])
        stalk = jaw + np.array(a["off_world_mm"])
        f33, f45 = frame_at(fr, 3.33), frame_at(fr, 4.5)
        D.append(dict(now=split(jaw, stalk), rest=split(jaw, rest), moved=float(np.linalg.norm(stalk - np.array(rest))),
                      need=az(rest) - az(start), x=a["off_grip_mm"][0], cut=bool(R[k]["cut"]), ok=bool(R[k]["success"]), tc=a["t"],
                      t33=None if f33 is None or f33["cut"] or f33.get("stalk_mm") is None else split(f33["jaw_mm"], f33["stalk_mm"])[0],
                      t45=None if f45 is None or f45["cut"] or f45.get("stalk_mm") is None else split(f45["jaw_mm"], f45["stalk_mm"])[0]))
    if not D:
        return
    print("   at the first closing of the jaw:")
    for name, sel in (("all runs", D), ("stalk outside the fixed finger", [d for d in D if d["x"] < -3.0]),
                      ("stalk between the fingers", [d for d in D if d["x"] >= -3.0]), ("successes", [d for d in D if d["ok"]])):
        if not sel:
            continue
        now = np.array([d["now"] for d in sel])
        rest = np.array([d["rest"] for d in sel])
        print(f"      {name:31s} n {len(sel):3d}: turn left {med(now[:, 0]):+5.2f} deg (25-75 %: {iqr(now[:, 0])}), reach left "
              f"{med(now[:, 1]):+5.1f} mm, jaw too low by {med(now[:, 2]):+5.1f} mm; the stalk has moved {med([d['moved'] for d in sel]):.1f} mm "
              f"from where it hung; measured to where it HUNG: turn {med(rest[:, 0]):+5.2f} deg")
    need = np.array([d["need"] for d in D])
    print(f"   by how far the arm has to turn from its start pose (median {med(need):.0f} deg, {need.min():.0f} to {need.max():.0f}):")
    for lo, hi in TURN_BINS:
        sel = [d for d in D if lo <= d["need"] < hi]
        a33 = [d["t33"] for d in sel if d["t33"] is not None]
        a45 = [d["t45"] for d in sel if d["t45"] is not None]
        if sel:
            print(f"      {lo:3d}-{hi:3d} deg: n {len(sel):3d}, turn left at 3.33 s {med(a33):+6.2f} deg, at 4.5 s {med(a45):+5.2f} deg; at the "
                  f"closing: turn {med([d['now'][0] for d in sel]):+5.2f} deg, reach {med([d['now'][1] for d in sel]):+5.1f} mm, too low "
                  f"{med([d['now'][2] for d in sel]):+5.1f} mm, closes at {med([d['tc'] for d in sel]):.2f} s, stalk outside the fixed finger in "
                  f"{sum(d['x'] < -3.0 for d in sel)}; cut {sum(d['cut'] for d in sel)}, success {sum(d['ok'] for d in sel)}")

    # 3. does the shoulder's command move after the second plan is made? (traces from 2026-11-20 on carry cmd and q)
    hold = []
    for k, fr in T.items():
        a = R[k].get("aim_close")
        if not a or "cmd" not in fr[0]:
            continue
        f0, f1 = frame_at(fr, 3.4, 0.05), frame_at(fr, a["t"], 0.05)
        if f0 is None or f1 is None or f1["t"] <= f0["t"] or f0.get("stalk_mm") is None:
            continue
        hold.append(((f1["cmd"][0] - f0["cmd"][0]) * UNIT_DEG, split(f0["jaw_mm"], f0["stalk_mm"])[0],
                     (f1["cmd"][0] - f1["q"][0]) * UNIT_DEG))
    if hold:
        h = np.array(hold)
        left = h[:, 1] > 1.0
        print(f"   shoulder command from 3.4 s to the first closing ({len(h)} runs): it moves {med(h[:, 0]):+.2f} deg (25-75 %: {iqr(h[:, 0])}) "
              f"while {med(h[:, 1]):+.2f} deg of turn is left; command minus reading at the closing {med(h[:, 2]):+.2f} deg")
        if left.sum() > 3:
            share = h[left, 0] / h[left, 1]
            print(f"      runs with more than 1 deg left at 3.4 s ({int(left.sum())}): the command makes up {med(share):.2f} of what is left "
                  f"(25-75 %: {np.percentile(share, 25):.2f} to {np.percentile(share, 75):.2f}; 1 = it finishes the turn, 0 = it holds still)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar=("LABEL", "RUN"),
                    help="a label, then the traced runs (folders in outputs/diag/runs/) to pool under it")
    a = ap.parse_args()
    for label, names in ([(g[0], g[1:]) for g in a.group] if a.group else DEFAULT):
        group(label, names)


if __name__ == "__main__":
    main()
