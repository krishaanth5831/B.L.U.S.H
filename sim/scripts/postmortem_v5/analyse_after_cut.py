#!/usr/bin/env python3
"""Post-mortem 2026-11-19: after the cut, when is the truss lost? Teacher vs policy, from the per-frame traces.

    python analyse_after_cut.py                                        the teacher and the policy as trained
    python analyse_after_cut.py --group "label" run [run ...]           any traced runs in outputs/diag/runs/
                                [--group ...]                           (numpy only)

Per run with a cut: follow the distance between the cut end of the stalk and the jaw (trace field stalk_to_jaw_mm).
"Held" = within HELD_MM. The run is sorted by where it stops being held: at the cut itself, on the way to the crate,
over the crate (a normal release), or never (still held at the end).

2026-11-20: --group, for the traced re-runs of the corrected-inference variants (fix_unstretch, fix_mcdropout,
fix_mcdropout2 against base_a + base_b, the same 50 held-out scenes), and what the jaw looked like at the cut.
"""
import argparse
import collections
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"
HELD_MM = 25.0     # stalk end within this of the jaw point = still in the jaw (pads are ~20 mm long; a guess, unverified)
BEFORE = 5         # frames before the cut in which a pad contact counts as "pads on the truss before the cut" (a choice)
OPEN, SHUT = 20.0, 10.0   # jaw reading above OPEN = open, below SHUT = shut (the teacher holds 28 open, 1-5 shut; read
                          # off the traces)
PRESS_N = 0.2      # the weaker pad presses the stalk at least this hard = both pads hold it (truss.CAPTURE_N)
DEFAULT = (("TEACHER (100 runs)", ("teacher", "train_teacher")),
           ("POLICY (150 runs: held-out twice + training scenes)", ("base_a", "base_b", "train_clean")))


def rows(name):
    f = sorted(glob.glob(str(RUNS / name / "act_*.jsonl")))[-1]
    return {r["seed"]: r for r in (json.loads(ln) for ln in open(f) if ln.strip()) if not r.get("summary") and not r.get("hold")}


def traces(name):
    out = collections.defaultdict(list)
    with open(RUNS / name / "trace.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            out[r["seed"]].append(r)
    return out


def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def group(label, names):
    kinds, held_s, detail = collections.Counter(), [], []
    speed, at_cut = [], collections.defaultdict(list)
    n_runs = n_ok = n_crate = 0
    for n in names:
        R, T = rows(n), traces(n)
        for s, r in R.items():
            n_runs += 1
            n_ok += bool(r["success"])
            n_crate += bool(r["in_crate"])
            tr = T[s]
            kc = next((i for i, x in enumerate(tr) if x["cut"]), None)
            if kc is None:
                kinds["no cut"] += 1
                continue
            after = tr[kc:]
            lost = next((i for i, x in enumerate(after) if x["stalk_to_jaw_mm"] is not None and x["stalk_to_jaw_mm"] > HELD_MM), None)
            t_cut = after[0]["t"]
            if lost is None:
                k0 = "held to the end"
            else:
                dt = after[lost]["t"] - t_cut
                over = after[lost]["over_crate"]
                k0 = ("lost within 0.5 s of the cut" if dt < 0.5 else "let go over the crate" if over
                      else "lost on the way to the crate")
                held_s.append(dt)
            k = k0 + " -> " + ("in crate" if r["in_crate"] else "NOT in crate")
            kinds[k] += 1
            # what the jaw looked like at the cut: how far the stalk's nearest point is from the jaw's cut point in
            # the last frame before the cut, and whether a spring pad touched the truss in the BEFORE frames before it
            pre = tr[max(0, kc - BEFORE):kc]
            if pre:
                # which closing of the jaw made the cut: the first one, a later one (the policy went round again),
                # or none (the jaw had been shut for a second already and something dragged the stalk into the blade)
                cmds = [x["cmd_gripper"] for x in tr[:kc + 1]]
                closings = sum(1 for u, v in zip(cmds, cmds[1:]) if u >= OPEN and v < OPEN)
                shut_before = tr[max(0, kc - 30)]["gripper"] < SHUT
                how = "jaw already shut" if shut_before else "first closing" if closings <= 1 else "a later closing"
                # the weaker pad's force on the stalk before and just after the cut (traces from 2026-11-20 on)
                pf = [x.get("pad_force_n") for x in pre]
                pf_after = [x.get("pad_force_n") for x in after[1:4]]
                at_cut[k0].append((pre[-1]["stalk_to_jaw_mm"], max(x["pad_contacts"] for x in pre), t_cut,
                                   pre[-1]["gripper"], how,
                                   max(pf) if pf and pf[0] is not None else None,
                                   max(pf_after) if pf_after and pf_after[0] is not None else None))
            # how fast the jaw moves while it carries (first 3 s after the cut), mm/s
            seg = [x for x in after if x["t"] - t_cut <= 3.0]
            if len(seg) > 5:
                p = np.array([x["jaw_mm"] for x in seg])
                v = np.linalg.norm(np.diff(p, axis=0), axis=1) * 30
                speed.append((float(np.percentile(v, 95)), float(np.abs(np.diff(v)).max() * 30)))
            pads = [x["pad_contacts"] for x in after[:15]]
            detail.append((n, s, r["target"], r.get("n_fruit"), k, round(t_cut, 2), max(pads) if pads else None))
    n_cut = n_runs - kinds["no cut"]
    print(f"\n{label}: {n_runs} runs, {n_ok} successes, {n_cut} cuts, {n_crate} in the crate")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {v:3d}  {k}")
    for k0, v in sorted(at_cut.items(), key=lambda kv: -len(kv[1])):
        a = np.array([x[:4] for x in v], float)
        how = collections.Counter(x[4] for x in v)
        print(f"   at the cut, '{k0}' ({len(v)}): stalk to jaw point {q(a[:, 0], 50):.1f} mm (median; 90 %: {q(a[:, 0], 90):.1f}); "
              f"a pad on the truss in the {BEFORE} frames before the cut in {int((a[:, 1] > 0).sum())}; cut at {q(a[:, 2], 50):.2f} s; "
              f"made by: " + ", ".join(f"{k} {n}" for k, n in how.most_common()))
        pf = [(x[5], x[6]) for x in v if x[5] is not None]
        if pf:
            b = np.array([(x, y if y is not None else np.nan) for x, y in pf], float)
            print(f"      weaker pad's force on the stalk before the cut: {q(b[:, 0], 50):.1f} N (median); both pads pressing "
                  f"(>= {PRESS_N} N) in {int((b[:, 0] >= PRESS_N).sum())} of {len(b)}; still pressing 0.1 s after the cut in "
                  f"{int((np.nan_to_num(b[:, 1]) >= PRESS_N).sum())}")
    if speed:
        sp = np.array(speed)
        print(f"   jaw speed in the 3 s after the cut: 95 % of frames below {q(sp[:, 0], 50):.0f} mm/s (median run; worst run "
              f"{sp[:, 0].max():.0f}); largest frame-to-frame change of speed {q(sp[:, 1], 50):.0f} mm/s2 (median run; 90 %: {q(sp[:, 1], 90):.0f})")
    if label.startswith("TEACHER"):
        for d in detail:
            if "NOT" in d[4]:
                print("     ", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar=("LABEL", "RUN"),
                    help="a label, then the traced runs (folders in outputs/diag/runs/) to pool under it")
    a = ap.parse_args()
    for label, names in ([(g[0], g[1:]) for g in a.group] if a.group else DEFAULT):
        group(label, names)


if __name__ == "__main__":
    main()
