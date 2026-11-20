#!/usr/bin/env python3
"""Scene-by-scene comparison of the corrected-inference rollouts of the 100k checkpoint with its baseline test
(same 100 held-out scenes + 20 nothing-ripe scenes). Policy-only and hybrid, finished or still running.

    python3 outputs/notes/2026-11-19_v5_postmortem/paired_vs_baseline.py        (numpy only, from the repo root)

⚠️ The baselines ran the policy on the CPU, the variants on the GPU. ⚠️ Noise floor: the same test run twice flips
about 3 scenes in 50 (FINDINGS 12c), so about 6 in 100 with nothing changed.
"""
import glob
import json
import os

import numpy as np

E = "outputs/eval"


def rows(f):
    R = [json.loads(ln) for ln in open(f) if ln.strip()]
    return [r for r in R if not r.get("summary")], [r for r in R if r.get("summary")]


def nfruit(r):
    v = r.get("fruit_off") or 0
    return len(v) if isinstance(v, (list, tuple)) else int(v)


def kind_of(r, s):
    hyb = (s[-1].get("hybrid") if s else r[0].get("hybrid")) if (s or r) else None
    return "hybrid" if hyb else "alone"


files = sorted(glob.glob(f"{E}/act_teacher_v5_100000_*.jsonl"), key=os.path.getmtime)
base = {}
for f in files:                                     # newest finished 100-scene test without a changed inference
    r, s = rows(f)
    if not s or s[-1].get("scenes") != 100 or s[-1].get("infer"):
        continue
    if s[-1].get("hybrid") and (s[-1].get("teacher_drives") or s[-1].get("trigger") != "failing"):
        continue
    base["hybrid" if s[-1].get("hybrid") else "alone"] = (f, r)

variants = {}
for f in files:
    r, s = rows(f)
    if not r:
        continue
    k = kind_of(r, s)
    if k not in base or os.path.getmtime(f) <= os.path.getmtime(base[k][0]):
        continue
    if s and not s[-1].get("infer"):
        continue
    tag = s[-1]["infer"] if s else "(running) " + os.path.basename(f)[-14:-6]
    variants[(k, tag)] = r

for k in ("alone", "hybrid"):
    if k not in base:
        continue
    B = {r["seed"]: r for r in base[k][1] if not r.get("hold")}
    BH = [r for r in base[k][1] if r.get("hold")]
    print(f"\n=== {k.upper()}: baseline {os.path.basename(base[k][0])}: {sum(r['success'] for r in B.values())}/{len(B)}"
          f", cut {sum(r['cut'] for r in B.values())}, in crate {sum(r['in_crate'] for r in B.values())}"
          + (f", holds {sum(r['success'] for r in BH)}/{len(BH)}" if BH else ""))
    got = {}
    for (kk, tag), r in variants.items():
        if kk != k:
            continue
        R = [x for x in r if not x.get("hold") and x["seed"] in B]
        H = [x for x in r if x.get("hold")]
        if len(R) < 5:
            continue
        got[tag] = {x["seed"]: x for x in R}
        n = len(R)
        b = [B[x["seed"]] for x in R]
        up = sum(x["success"] and not y["success"] for x, y in zip(R, b))
        down = sum((not x["success"]) and y["success"] for x, y in zip(R, b))
        cm, bm = [x["closest_mm"] for x in R], [y["closest_mm"] for y in b]
        cuts, bcuts = [x for x in R if x["cut"]], [y for y in b if y["cut"]]
        line = (f"\n{tag}: {n} scenes\n"
                f"   success {sum(x['success'] for x in R)} (baseline on the same scenes {sum(y['success'] for y in b)}); "
                f"won {up}, lost {down}; clean {sum(bool(x.get('success_strict')) for x in R)} "
                f"({sum(bool(y.get('success_strict')) for y in b)})\n"
                f"   cut {len(cuts)} ({len(bcuts)}); in crate {sum(x['in_crate'] for x in R)} ({sum(y['in_crate'] for y in b)}); "
                f"of the cuts delivered {sum(x['in_crate'] for x in cuts)}/{len(cuts)} ({sum(y['in_crate'] for y in bcuts)}/{len(bcuts)}); "
                f"wrong truss cut {sum(bool(x.get('wrong_cut')) for x in R)} ({sum(bool(y.get('wrong_cut')) for y in b)})\n"
                f"   closest median {np.median(cm):.1f} mm ({np.median(bm):.1f}); closer in {sum(a < c for a, c in zip(cm, bm))}/{n}; "
                f"within 6 mm {sum(a <= 6 for a in cm)} ({sum(c <= 6 for c in bm)}); within 9 mm {sum(a <= 9 for a in cm)} ({sum(c <= 9 for c in bm)})\n"
                f"   fruit knocked off: runs {sum(nfruit(x) > 0 for x in R)} ({sum(nfruit(y) > 0 for y in b)})")
        if H:
            line += f"; holds {sum(x['success'] for x in H)}/{len(H)}"
        if k == "hybrid":
            line += (f"\n   teacher stepped in {sum(bool(x.get('rescued')) for x in R)} ({sum(bool(y.get('rescued')) for y in b)}); "
                     f"policy finished by itself {sum(x['success'] for x in R if not x.get('rescued'))}/{sum(not x.get('rescued') for x in R)} "
                     f"({sum(y['success'] for y in b if not y.get('rescued'))}/{sum(not y.get('rescued') for y in b)})")
        print(line)
    tags = list(got)
    for i in range(len(tags)):                      # variants against each other on the scenes both have run
        for j in range(i + 1, len(tags)):
            a, c = got[tags[i]], got[tags[j]]
            common = sorted(set(a) & set(c))
            if len(common) < 5:
                continue
            print(f"   [{tags[i]}] vs [{tags[j]}] on {len(common)} common scenes: success {sum(a[s]['success'] for s in common)} vs "
                  f"{sum(c[s]['success'] for s in common)}; only first {sum(a[s]['success'] and not c[s]['success'] for s in common)}, "
                  f"only second {sum(c[s]['success'] and not a[s]['success'] for s in common)}; cut {sum(a[s]['cut'] for s in common)} vs "
                  f"{sum(c[s]['cut'] for s in common)}; in crate {sum(a[s]['in_crate'] for s in common)} vs {sum(c[s]['in_crate'] for s in common)}")
