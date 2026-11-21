#!/usr/bin/env python3
"""MuJoCo 4.0.1 port (2026-11-21): the same runs under MuJoCo 3.8.1 and 4.0.1, compared scene by scene. No simulation.

    python compare_versions.py                         teacher + train_teacher against teacher_mj4 + train_teacher_mj4
    python compare_versions.py --pair old new [--pair old new ...]   any traced runs in outputs/diag/runs/

Per scene in both runs: the outcome (success, cut, in the crate), when the cut and the crate happen, and how far the
jaw's cut point drifts between the two versions over time (the same seed, the same commands from a deterministic
teacher: any difference is the simulator). The teacher re-plans from what it sees, so a small physics difference can
also change its plan; the share of scenes with the same number of frames says how often that happened.
"""
import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postmortem_v5"))
import turn_reach_height as trh  # noqa: E402

LOOKS = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0)   # s


def first_cut_t(fr):
    return next((f["t"] for f in fr if f["cut"]), None)


def compare(old, new):
    Ro, Rn = trh.rows(old), trh.rows(new)
    To, Tn = trh.traces(old), trh.traces(new)
    seeds = sorted(set(Ro) & set(Rn))
    if not seeds:
        print(f"\n{old} vs {new}: no scenes in both (finished?)")
        return
    so = sum(bool(Ro[s]["success"]) for s in seeds)
    sn = sum(bool(Rn[s]["success"]) for s in seeds)
    only_o = [s for s in seeds if Ro[s]["success"] and not Rn[s]["success"]]
    only_n = [s for s in seeds if Rn[s]["success"] and not Ro[s]["success"]]
    print(f"\n{old} (3.8.1) against {new} (4.0.1), {len(seeds)} scenes in both")
    print(f"   success {so} against {sn}; only under 3.8.1: {len(only_o)} {only_o[:8]}; "
          f"only under 4.0.1: {len(only_n)} {only_n[:8]}")
    for k in ("cut", "in_crate"):
        print(f"   {k}: {sum(bool(Ro[s][k]) for s in seeds)} against {sum(bool(Rn[s][k]) for s in seeds)}")
    same_len = sum(Ro[s].get("frames") == Rn[s].get("frames") for s in seeds)
    print(f"   same number of frames (the same plan, played out the same length): {same_len} of {len(seeds)}")
    dcut = [first_cut_t(Tn[s]) - first_cut_t(To[s]) for s in seeds if s in To and s in Tn
            and first_cut_t(To[s]) is not None and first_cut_t(Tn[s]) is not None]
    if dcut:
        print(f"   cut time, 4.0.1 minus 3.8.1: median {np.median(dcut):+.2f} s, largest {max(dcut, key=abs):+.2f} s "
              f"({len(dcut)} scenes cut in both)")
    line = []
    for t in LOOKS:
        d = []
        for s in seeds:
            if s not in To or s not in Tn:
                continue
            a, b = trh.frame_at(To[s], t), trh.frame_at(Tn[s], t)
            if a is not None and b is not None:
                d.append(float(np.linalg.norm(np.subtract(a["jaw_mm"], b["jaw_mm"]))))
        if d:
            line.append(f"{t:.0f} s: {np.median(d):.2f} / {np.percentile(d, 90):.1f} / {max(d):.1f}")
    print("   jaw cut point, distance between the versions at the same moment (median / 90 % / largest, mm):")
    print("      " + "   ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("OLD", "NEW"))
    a = ap.parse_args()
    for old, new in (a.pair or [("teacher", "teacher_mj4"), ("train_teacher", "train_teacher_mj4")]):
        compare(old, new)


if __name__ == "__main__":
    main()
