#!/usr/bin/env python3
"""Post-mortem 2026-11-19: after the cut, when is the truss lost? Teacher vs policy, from the per-frame traces.

    python analyse_after_cut.py        (numpy only)

Per run with a cut: follow the distance between the cut end of the stalk and the jaw (trace field stalk_to_jaw_mm).
"Held" = within HELD_MM. The run is sorted by where it stops being held: at the cut itself, on the way to the crate,
over the crate (a normal release), or never (still held at the end).
"""
import collections
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs" / "diag" / "runs"
HELD_MM = 25.0     # stalk end within this of the jaw point = still in the jaw (pads are ~20 mm long; a guess, unverified)


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


def main():
    for names, label in ((("teacher", "train_teacher"), "TEACHER (100 runs)"),
                         (("base_a", "base_b", "train_clean"), "POLICY (150 runs: held-out twice + training scenes)")):
        kinds, held_s, detail = collections.Counter(), [], []
        speed = []
        for n in names:
            R, T = rows(n), traces(n)
            for s, r in R.items():
                tr = T[s]
                kc = next((i for i, x in enumerate(tr) if x["cut"]), None)
                if kc is None:
                    kinds["no cut"] += 1
                    continue
                after = tr[kc:]
                lost = next((i for i, x in enumerate(after) if x["stalk_to_jaw_mm"] is not None and x["stalk_to_jaw_mm"] > HELD_MM), None)
                t_cut = after[0]["t"]
                if lost is None:
                    k = "held to the end"
                else:
                    dt = after[lost]["t"] - t_cut
                    over = after[lost]["over_crate"]
                    k = ("lost within 0.5 s of the cut" if dt < 0.5 else "let go over the crate" if over
                         else "lost on the way to the crate")
                    held_s.append(dt)
                k += " -> " + ("in crate" if r["in_crate"] else "NOT in crate")
                kinds[k] += 1
                # how fast the jaw moves while it carries (first 3 s after the cut), mm/s
                seg = [x for x in after if x["t"] - t_cut <= 3.0]
                if len(seg) > 5:
                    p = np.array([x["jaw_mm"] for x in seg])
                    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * 30
                    speed.append((float(np.percentile(v, 95)), float(np.abs(np.diff(v)).max() * 30)))
                pads = [x["pad_contacts"] for x in after[:15]]
                detail.append((n, s, r["target"], r.get("n_fruit"), k, round(t_cut, 2), max(pads) if pads else None))
        print(f"\n{label}")
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"   {v:3d}  {k}")
        if speed:
            sp = np.array(speed)
            print(f"   jaw speed in the 3 s after the cut: 95 % of frames below {q(sp[:, 0], 50):.0f} mm/s (median run; worst run "
                  f"{sp[:, 0].max():.0f}); largest frame-to-frame change of speed {q(sp[:, 1], 50):.0f} mm/s2 (median run; 90 %: {q(sp[:, 1], 90):.0f})")
        if label.startswith("TEACHER"):
            for d in detail:
                if "NOT" in d[4]:
                    print("     ", d)


if __name__ == "__main__":
    main()
