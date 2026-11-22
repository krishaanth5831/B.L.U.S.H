#!/usr/bin/env python3
"""Teacher v5 via search, the pure-Python steps (2026-11-18):
    via_pick.py cands via/cases_*.jsonl            # pool the RRT way-points, cluster (max-abs NORM_TOL)
                                                   #   -> via/cands.json
    via_pick.py pick via/score_*.jsonl             # greedy: the via that unblocks the most NEW scenes, then the next
A scene counts as unblocked when one of its blocked cases clears (the teacher needs one candidate)."""
import json
import sys

import numpy as np

NORM_TOL = 12.0          # way-points this close (largest joint difference, norm units) are one candidate (guess)
KEEP = 6                 # at most this many new vias (VIA_NORM has 4)

if sys.argv[1] == "cands":
    pts = []
    for f in sys.argv[2:]:
        for line in open(f):
            r = json.loads(line)
            pts += r["via"]
    P = np.array(pts)
    print(len(P), "way-points from", sum(1 for f in sys.argv[2:] for _ in open(f)), "cases")
    cands, counts = [], []
    left = np.ones(len(P), bool)
    while left.any():                    # greedy: the way-point with the most neighbours is a candidate
        idx = np.flatnonzero(left)
        D = np.abs(P[idx][:, None, :] - P[idx][None, :, :]).max(2)
        nb = (D < NORM_TOL).sum(1)
        c = idx[int(np.argmax(nb))]
        members = idx[D[int(np.argmax(nb))] < NORM_TOL]
        cands.append(P[members].mean(0).round(1).tolist())
        counts.append(int(len(members)))
        left[members] = False
    json.dump(cands, open("via/cands.json", "w"))
    print(len(cands), "candidates; way-points per candidate:", counts[:20])
else:
    rows = [json.loads(x) for f in sys.argv[2:] for x in open(f) if x.strip()]
    cands = json.load(open("via/cands.json"))
    seeds = sorted({r["seed"] for r in rows})
    clears = {j: {r["seed"] for r in rows if j in r["ok"]} for j in range(len(cands))}
    any_clear = set().union(*clears.values()) if clears else set()
    print(f"{len(rows)} blocked cases in {len(seeds)} scenes; {len(any_clear)} scenes unblocked by some candidate")
    chosen, got = [], set()
    for _ in range(KEEP):
        j = max(clears, key=lambda k: (len(clears[k] - got), -k))
        gain = clears[j] - got
        if not gain:
            break
        chosen.append(j)
        got |= gain
        print(f"  via {j} {cands[j]}: +{len(gain)} scenes (total {len(got)} of {len(seeds)})")
    json.dump([cands[j] for j in chosen], open("via/chosen.json", "w"))
