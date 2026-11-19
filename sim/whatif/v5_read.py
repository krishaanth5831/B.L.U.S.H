#!/usr/bin/env python3
"""Teacher v5 dry-run results (2026-11-18): python v5_read.py TAG [TAG ...]  (rows of blush_eval_v5_<TAG>_dry_*.jsonl)
Per run: kept, kept on the first plan (what a finisher gets), dry runs used, what the kept plans used (phase, mode,
u, release alternate, retreat), failure classes, by the target's fruit count (census_world for the 154 held-out
seeds, else the row's own n_fruit), and the logged strain."""
import collections
import glob
import json
import pathlib
import sys

T = pathlib.Path.home() / ".cache/blush_sim/truss"
S = pathlib.Path(__file__).resolve().parents[2] / "outputs/notes/2026-11-16_17_step3_night"   # untracked

# the failure classes of the v4 census (step3_night/census_v4_read.py), copied
def cls(r):
    """A failure class from a generate_truss / release_run row."""
    if r is None:
        return "missing"
    if r.get("kept") is True:
        return "kept"
    why = (r.get("reason") or "").lower()
    for key, name in (("no ready", "no target"), ("no target", "no target"), ("not (all) in the crate", "not in crate"),
                      ("fruit", "fruit off"), ("neighbour", "neighbour moved"), ("u 0.", "cut not at u 0.5"),
                      ("no release pose", "no release pose"), ("crate blocked", "carry blocked"),
                      ("crate -> start", "carry blocked"), ("slide hits", "slide hits"),
                      ("standoff blocked", "path blocked"), ("no ik", "IK reach"), ("start pose", "start pose"),
                      ("dry run", "dry run other")):
        if key in why:
            return name
    if "no plan left" in why:
        return "no plan left"
    return "other: " + why[:50]


def rows(tags):
    out = {}
    for tag in tags:
        for f in sorted(glob.glob(str(T / f"blush_eval_v5_{tag}_dry_*.jsonl"))):
            for line in open(f):
                if line.startswith("{"):
                    r = json.loads(line)
                    if "seed" in r:
                        out[r["seed"]] = r
    return out


R = rows(sys.argv[1:])
cw = {}
for p in [S / "census_world/new.jsonl"] + sorted((S.parent / "2026-11-18_teacher_v5/census").glob("*.jsonl")):
    if p.exists():
        for line in open(p):
            if line.startswith("{"):
                r = json.loads(line)
                cw[r["seed"]] = r
n = len(R)
kept = {s for s, r in R.items() if r.get("kept") is True}
first = {s for s in kept if (R[s].get("tries") or [{}])[0].get("ok")}
print(f"{n} seeds: kept {len(kept)} ({100 * len(kept) / max(n, 1):.0f} %), on the first plan {len(first)}, "
      f"after a retry {len(kept) - len(first)}")
c = collections.Counter()
for s in kept:
    m = R[s]["plan"]
    c[(m.get("phase"), m.get("neighbours"))] += 1
print("kept plans by phase / mode:", dict(sorted(c.items(), key=lambda kv: -kv[1])))
print("kept u:", dict(collections.Counter(R[s]["plan"]["cut_u"] for s in kept)))
print("kept retreat:", dict(collections.Counter(str(R[s]["plan"]["retreat_option"]) for s in kept)))
print("dry runs per kept seed:", dict(collections.Counter(len(R[s].get("tries", [])) for s in kept)))
fails = collections.Counter()
for s, r in R.items():
    if s not in kept:
        why = r.get("reason", "")
        last = why.split(" | ")[-1] if " | " in why else why
        fails[cls(dict(reason=last))] += 1
print("failures (last reason):", dict(fails.most_common()))
try_fail = collections.Counter()
for s, r in R.items():
    for t in r.get("tries", []):
        if not t["ok"]:
            try_fail[cls(dict(reason="dry run: " + t["why"]))] += 1
print("failed dry runs, all tries:", dict(try_fail.most_common()))
by_n = collections.defaultdict(lambda: [0, 0])
for s, r in R.items():
    k = cw[s]["n"] if s in cw else (r.get("plan") or {}).get("n_fruit")
    by_n[k][0] += s in kept
    by_n[k][1] += 1
print("by fruit count (kept / seeds):", {k: f"{a}/{b}" for k, (a, b) in sorted(by_n.items(), key=lambda kv: str(kv[0]))})
st = [t["strain"] for r in R.values() for t in r.get("tries", []) if t.get("ok") and "strain" in t]
if st:
    for key in ("hard_n", "sat_frames", "sat_run", "cmd_step", "cmd_acc"):
        v = sorted(x[key] for x in st)
        print(f"strain {key} on kept runs: median {v[len(v) // 2]}, p90 {v[int(len(v) * 0.9)]}, max {v[-1]}")
print("still failing:", " ".join(str(s)[-3:] for s in sorted(set(R) - kept)))
