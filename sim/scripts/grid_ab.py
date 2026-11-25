#!/usr/bin/env python3
"""Compare base_grid runs arm by arm on the seeds every arm has (won/lost vs the first arm, failure types).

    python sim/scripts/grid_ab.py A=outputs/v6/base_grid_r13_A_*.jsonl B=outputs/v6/base_grid_r13_B_*.jsonl

Each argument is NAME=glob[,glob...]; later files win when a seed repeats. The first arm is the reference.
--swing: also what each arm would score at other limits for a neighbour's swing (swing_table).
⚠️ A "no plan" record lists every reason any candidate hit, per mode, SORTED ALPHABETICALLY (truss_teacher joins
sorted(set(tried))), so its first reason is not the main one. This scores a no-plan scene by the FURTHEST stage any
candidate reached (STAGES order); the why string is cut at 400 chars, so a late stage can be missed (rare).
"""
import collections
import glob
import json
import re
import sys

STAGES = ["start pose not clear", "start -> standoff blocked", "no cut pose", "no IK along the slide", "slide hits",
          "retreat: the truss in hand hits", "retreat -> crate blocked", "no release pose"]   # plan order


def load(globs):
    out = {}
    for g in globs.split(","):
        for f in sorted(glob.glob(g)):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    if "seed" in r:
                        out[r["seed"]] = r
    return out


def bucket(r):
    w = str(r.get("why"))
    if r.get("ok"):
        return "ok"
    if not r.get("plan", True) or w.startswith("no plan"):
        best = max((i for i, st in enumerate(STAGES) if st in w), default=None)
        return "no plan: " + (STAGES[best] if best is not None else "?")
    m = re.search(r" in (\w+)", w)
    leg = m.group(1) if m else "?"
    if "fruit came off" in w:
        return f"fruit off ({leg})"
    if "moved a neighbour" in w:
        return f"neighbour moved ({leg})"
    if "not (all) in" in w:
        return "not in crate"
    return w[:40]


def swing_table(arms, seeds, limits=(15, 20, 30, 40, 60, None)):
    """--swing: scenes that would pass if a neighbour's fruit may swing up to each limit (mm; None = no limit), from
    the attempts' nb_mm + why_noswing (base_grid's second reading). A scene passes at a limit if any attempt that ran
    passes there; the retries were chosen under the judge's own limit, so this is a close estimate, not a re-run."""
    for k, v in arms.items():
        have = [s for s in seeds if all("nb_mm" in at for at in v[s].get("attempts", []))]
        if not have:
            continue
        row = []
        for lim in limits:
            n = sum(any(at["why_noswing"] == "ok" and (lim is None or at["nb_mm"] <= lim)
                        for at in v[s].get("attempts", [])) for s in have)
            row.append(f"{'none' if lim is None else lim}: {n} ({100 * n / len(have):.0f} %)")
        print(f"{k}: neighbour swing limit (mm) -> scenes ok of {len(have)}:  " + "   ".join(row))


def main():
    swing = "--swing" in sys.argv
    sys.argv = [x for x in sys.argv if x != "--swing"]
    arms = {}
    for a in sys.argv[1:]:
        name, _, globs = a.partition("=")
        arms[name] = load(globs)
    seeds = sorted(set.intersection(*[set(v) for v in arms.values()]))
    print(f"seeds in every arm: {len(seeds)}  ({', '.join(f'{k} {len(v)}' for k, v in arms.items())})")
    names = list(arms)
    for k in names:
        v = arms[k]
        ok = sum(bool(v[s].get("ok")) for s in seeds)
        first = sum(1 for s in seeds if v[s].get("ok") and v[s].get("tries", 1) == 1)
        print(f"{k}: ok {ok}/{len(seeds)} = {100 * ok / len(seeds):.1f} %  (first plan {first})")
        for b, n in collections.Counter(bucket(v[s]) for s in seeds).most_common():
            if b != "ok":
                print(f"     {n:3d}  {b}")
    if swing:
        swing_table(arms, seeds)
    ref = names[0]
    for k in names[1:]:
        won = [s for s in seeds if arms[k][s].get("ok") and not arms[ref][s].get("ok")]
        lost = [s for s in seeds if not arms[k][s].get("ok") and arms[ref][s].get("ok")]
        print(f"{k} vs {ref}: won {len(won)} lost {len(lost)} net {len(won) - len(lost):+d}")
        print(f"     won  {won}\n     lost {lost}")
    if len(names) > 1:
        print("ok in at least one arm:", sum(any(arms[k][s].get("ok") for k in names) for s in seeds),
              "| failed in every arm:", sum(not any(arms[k][s].get("ok") for k in names) for s in seeds))


if __name__ == "__main__":
    main()
