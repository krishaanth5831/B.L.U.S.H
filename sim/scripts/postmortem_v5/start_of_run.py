#!/usr/bin/env python3
"""Post-mortem 2026-11-20: the measures that were written down BEFORE the picture tests ran (FINDINGS section 13),
for any traced runs. Reads eval rows and per-frame traces; no simulation.

    python start_of_run.py                                   the two picture tests against their baselines
    python start_of_run.py --group "label" run [run ...]     any runs, pooled under one label (repeatable)

A run is a folder in outputs/diag/runs/ (trace.jsonl + the eval rows act_*.jsonl), or the path of an eval .jsonl
(rows only: the trace columns stay empty).

Columns, in the order the criteria were written:
    travel      how far the jaw's cut point has moved from where it started after 10 / 30 / 60 / 100 commands (median, mm).
                The teacher: about 50 / 156 / 263 / 269 on held-out scenes.
    turn left   seen from above around the shoulder's axis: how many degrees the stalk is further round than the jaw
                at 3.33 s, when the second plan is made (+ = the arm has not turned far enough; runs not yet cut)
    outside     runs whose stalk is outside the fixed finger when the jaw first closes (where_at_shut.cls)
    success     with the scene-by-scene comparison against the baseline on the same scenes: won / lost. With nothing
                changed about 3 scenes in 50 (6 in 100) flip each way.
"""
import argparse
import glob
import json
import pathlib

import numpy as np

import turn_reach_height as trh
import where_at_shut as was

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"
EVAL = ROOT / "outputs" / "eval"
LOOK = (10, 30, 60, 100)   # commands after the start; trace line k (from 0) is the state after command k + 1


def load(run):
    """-> ({seed: row}, {seed: [trace lines]})"""
    p = pathlib.Path(run)
    if p.suffix == ".jsonl":
        R = [json.loads(ln) for ln in open(p) if ln.strip()]
        return {r["seed"]: r for r in R if not r.get("summary") and not r.get("hold")}, {}
    return trh.rows(run), trh.traces(run)


def measure(row, fr):
    out = dict(success=bool(row["success"]), cut=bool(row["cut"]), crate=bool(row.get("in_crate")), travel=None, turn=None, where=None,
               t_close=None)
    a = row.get("aim_close")
    if a:
        out["where"], out["t_close"] = was.cls(a["off_grip_mm"]), a["t"]
    if fr:
        j0 = np.array(fr[0]["jaw_mm"])   # ⚠️ the state after the first command, 0.03 s in: the jaw has moved under 4 mm by then
        out["travel"] = [float(np.linalg.norm(np.array(fr[k - 1]["jaw_mm"]) - j0)) if len(fr) >= k else np.nan for k in LOOK]
        f = trh.frame_at(fr, 3.33)
        if f is not None and not f["cut"] and f.get("stalk_mm") is not None:
            out["turn"] = trh.split(f["jaw_mm"], f["stalk_mm"])[0]
    return out


def med(a):
    a = [x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(a)) if a else float("nan")


def table(label, runs, base=None):
    """runs, base: lists of run names. Prints one line for runs (pooled); with base, the paired comparison too."""
    M = []
    for run in runs:
        R, T = load(run)
        M.append({s: measure(r, T.get(s)) for s, r in R.items()})
    allm = [m for d in M for m in d.values()]
    if not allm:
        print(f"{label}: no finished run yet")
        return
    n = len(allm)
    tr = [m["travel"] for m in allm if m["travel"]]
    travel = " / ".join(f"{med([t[i] for t in tr]):.0f}" for i in range(len(LOOK))) if tr else "no trace"
    closed = [m for m in allm if m["where"]]
    outside = sum(m["where"] == was.CLASSES[0] for m in closed)
    blade = sum(m["where"] == was.CLASSES[1] for m in closed)
    line = f"{label}\n      runs {n}: success {sum(m['success'] for m in allm)}, cut {sum(m['cut'] for m in allm)}, in crate {sum(m['crate'] for m in allm)}"
    if tr:
        line += (f" | travel after {'/'.join(map(str, LOOK))} commands: {travel} mm | turn left at 3.33 s: {med([m['turn'] for m in allm]):+.2f} deg "
                 f"(n {sum(m['turn'] is not None for m in allm)})")
    if closed:   # eval rows from before 2026-11-20 do not record where the stalk was at the first closing
        line += (f" | first closing at {med([m['t_close'] for m in closed]):.2f} s: outside the fixed finger {outside}/{len(closed)} "
                 f"({outside / len(closed):.0%}), between the fingers at blade depth {blade}/{len(closed)}")
    print(line)
    if base:
        for b in base:
            Rb, Tb = load(b)
            B = {s: measure(r, Tb.get(s)) for s, r in Rb.items()}
            for d in M:
                both = sorted(set(d) & set(B))
                if not both:
                    continue
                won = sum(d[s]["success"] and not B[s]["success"] for s in both)
                lost = sum(B[s]["success"] and not d[s]["success"] for s in both)
                o_new = sum(d[s]["where"] == was.CLASSES[0] for s in both)
                o_old = sum(B[s]["where"] == was.CLASSES[0] for s in both)
                where = f"; outside the fixed finger {o_new} against {o_old}" if any(B[s]["where"] for s in both) else ""
                print(f"      against {'the 100-scene test' if str(b).endswith('.jsonl') else b} on the same {len(both)} scenes: success "
                      f"{sum(d[s]['success'] for s in both)} against {sum(B[s]['success'] for s in both)} (won {won}, lost {lost}); cut "
                      f"{sum(d[s]['cut'] for s in both)} against {sum(B[s]['cut'] for s in both)}; in crate {sum(d[s]['crate'] for s in both)} against "
                      f"{sum(B[s]['crate'] for s in both)}{where}")


# ---- by the teacher's opening move (advisor, 2026-11-20): does a scene whose teacher took the detour do worse? ----
FAST_MM = 80.0   # jaw travel after 10 commands above this = the teacher's fast opening move (the detour over the
                 # via-pose): the valley between the two humps of the 736 recorded episodes, FINDINGS 13c


def teacher_opening():
    """{seed: True for the fast opening move} from the teacher's own traced runs (50 training + 50 held-out scenes)."""
    out = {}
    for name in ("train_teacher", "teacher"):
        for s, fr in trh.traces(name).items():
            if len(fr) >= 10:
                out[s] = float(np.linalg.norm(np.array(fr[9]["jaw_mm"]) - np.array(fr[0]["jaw_mm"]))) > FAST_MM
    return out


def by_opening(label, runs):
    fast = teacher_opening()
    print(label)
    for kind, want in (("the teacher opened with the straight move", False), ("the teacher opened with the detour", True)):
        A = []
        for run in runs:
            R, T = load(run)
            for s, r in R.items():
                if fast.get(s) is not want:
                    continue
                fr = T.get(s) or []
                early = [f for f in fr if f["t"] <= 3.34]
                m = measure(r, fr)
                m.update(fruit=bool(r.get("fruit_off")), moved=r.get("neighbours_moved_mm") or 0.0,
                         touch=any(f.get("arm_contacts") for f in early) if early else None)
                A.append(m)
        if not A:
            continue
        tr = [m["travel"] for m in A if m["travel"]]
        closed = [m for m in A if m["where"]]
        touch = [m["touch"] for m in A if m["touch"] is not None]
        print(f"      {kind:44s} runs {len(A):3d}: success {sum(m['success'] for m in A):3d} ({np.mean([m['success'] for m in A]):.0%}), cut "
              f"{sum(m['cut'] for m in A):3d}, travel after 10/30 commands {med([t[0] for t in tr]):.0f} / {med([t[1] for t in tr]):.0f} mm, turn left at 3.33 s "
              f"{med([m['turn'] for m in A]):+.2f} deg, outside the fixed finger {sum(m['where'] == was.CLASSES[0] for m in closed)}/{len(closed)}, "
              f"fruit knocked off in {sum(m['fruit'] for m in A)}, arm touched something before 3.33 s in {sum(touch)}/{len(touch)}, "
              f"neighbours moved {med([m['moved'] for m in A]):.0f} mm (median)")


def newest_eval(pred):
    """The newest finished 100-scene policy-only test in outputs/eval whose summary row passes pred."""
    best = None
    for f in sorted(glob.glob(str(EVAL / "act_teacher_v5_100000_*.jsonl")), key=lambda p: pathlib.Path(p).stat().st_mtime):
        s = [r for r in (json.loads(ln) for ln in open(f) if ln.strip()) if r.get("summary")]
        if s and s[-1].get("scenes") == 100 and not s[-1].get("hybrid") and pred(s[-1].get("infer") or ""):
            best = f
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar=("LABEL", "RUN"))
    ap.add_argument("--by-opening", action="store_true", help="split each --group by the teacher's opening move in that scene")
    a = ap.parse_args()
    if a.group:
        for g in a.group:
            (by_opening if a.by_opening else table)(g[0], g[1:])
        return
    print("TRAINING SCENES (the 50 clean ones)")
    table("   teacher (train_teacher)", ["train_teacher"])
    table("   policy as trained, raw pictures (train_clean)", ["train_clean"])
    table("   first plan from the dataset's recorded start frame (train_recorded)", ["train_recorded"], ["train_clean"])
    print("\nHELD-OUT SCENES")
    table("   teacher, first 50 scenes", ["teacher"])
    table("   policy as trained, first 50 scenes traced three times (base_a, base_b, base_c)", ["base_a", "base_b", "base_c"])
    base100 = newest_eval(lambda tag: not tag)
    if base100:
        table("   policy as trained, the 100-scene test (rows only)", [base100])
    for name, lab in (("codec1", "planning pictures through the encoder as a 1-frame clip (codec1)"),
                      ("codec1_unstretch", "the same, and the stretch divided out (codec1_unstretch)")):
        if (RUNS / name / "trace.jsonl").exists() and trh.rows(name):
            table(f"   {lab}", [name], ([base100] if base100 else []) + ["base_a", "base_b", "base_c"])


if __name__ == "__main__":
    main()
