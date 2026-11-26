#!/usr/bin/env python3
"""Seed plans for dataset v6's new episodes: which seeds each recording part tries, so that the old 800 episodes and
the new ones TOGETHER are spread evenly over where the truss hangs (K7 2026-11-26: "more varied episodes", "another
600 episodes on top of the previous 800"; post-mortem step 6: "about 50 episodes per position and height").

    python scripts/v6_seed_plan.py --census ../outputs/v6/census_v6/cen_*.jsonl --old ../outputs/v6/cams/v5_*.npz \\
        --out ../outputs/v6/seed_plan_v6            # from sim/, env lerobot-mj4; writes p00.jsonl ... + plan.json
    python scripts/v6_seed_plan.py ... --dry        # only the table

Reads no simulator and writes no dataset. The inputs:
  --census  base_grid.py --build-only records of fresh seeds (seed, target, cut_mm), built in the dataset's world
            with the scene generator the recording will use (--world crate23, layout rig base).
  --old     where the old dataset's cut points are: camera_score.py collect files of its own scenes (p_cut).
A cut episode's cell = generate_truss.cell_of(cut point): 4 parts along the row x 3 heights. The new episodes go
one at a time to the cell that has the fewest so far (old + new), as long as the census still has MIN_SUPPLY seeds
per episode for it: so the cells end level where the census allows it, and a rare cell gets what its seeds reach
instead of leaving a part short. Each cell's episodes are split over the parts as evenly as it goes; a part's file
lists, per cell, up to OVERSUPPLY seeds per episode of its quota (the teacher fails some), the cells taking turns,
and the part's nothing-ripe seeds. generate_truss.py --seeds reads it (read_seed_plan): a full cell's seeds are
skipped without building a scene. No seed is listed twice, in a part or across parts.
⚠️ A part can still end short when the teacher fails more than the list has room for; --topup plans one more part
for what is missing (the recorder's attempt logs say what was kept):
    python scripts/v6_seed_plan.py --census ... --old ... --topup ~/.cache/blush_sim/truss/<name>_new_p*.jsonl \\
        --plan ../outputs/v6/seed_plan_v6/plan.json --used "../outputs/v6/seed_plan_v6/p*.jsonl" --first-part 24 \\
        --out ../outputs/v6/seed_plan_v6        # writes p24.jsonl (+ plan_topup_p24.json), then record part 24
"""
import argparse
import collections
import glob
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import generate_truss as gt  # noqa: E402

CELLS = [f"x{i}z{j}" for j in range(len(gt.CELL_Z_EDGES) + 1) for i in range(gt.BALANCE_BINS)]
OVERSUPPLY = 2.5        # seeds listed per episode wanted, where the census has them. Dataset v5's log: the teacher
#                         kept 736 of 1,044 cut scenes it tried = 70 %; per slot 17-83 %, so 2.5 leaves room for a
#                         cell at 40 %
MIN_SUPPLY = 1.8        # a cell is planned only as far as the census has this many seeds per episode (1 / 1.8 = 56 %
#                         kept). Dataset v5's recording kept 60-83 % per cell, the lowest x2z2 60 % and x3z0 61 %
#                         (its attempt logs; the refused scenes' cells from the logged cut_x and the cue pixel
#                         against the kept ones, 96 % right on those; 2026-11-27), so 1.6 was too tight for two cells
# the share of built scenes the teacher kept in dataset v5's recording, per cell (same source as MIN_SUPPLY's: 734
# kept, 308 refused). Only used to say how many episodes the lists should give (expected_kept), not to plan
KEEP_RATE = {"x0z0": 0.83, "x1z0": 0.75, "x2z0": 0.66, "x3z0": 0.61, "x0z1": 0.80, "x1z1": 0.83, "x2z1": 0.65,
             "x3z1": 0.68, "x0z2": 0.68, "x1z2": 0.70, "x2z2": 0.60, "x3z2": 0.67}


def census(files):
    """{cell: [seed, ...]} in seed order, and the seeds with no ripe truss."""
    pool, none, seen = collections.defaultdict(list), [], set()
    for f in sorted(files):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:                  # a line cut short by a stopped job
                continue
            if "seed" not in r or r["seed"] in seen:
                continue
            seen.add(r["seed"])
            if r.get("cut_mm") is None:
                none.append(r["seed"])
            else:
                pool[gt.cell_of(np.asarray(r["cut_mm"], float) / 1000.0)].append(r["seed"])
    return pool, none, len(seen)


def old_cells(files):
    c = collections.Counter()
    for f in files:
        for p in np.load(f, allow_pickle=True)["p_cut"]:
            c[gt.cell_of(p)] += 1
    return c


def level(old, n, cap):
    """New episodes per cell: one at a time to the cell with the fewest so far (old + new) that is still under its
    cap. -> {cell: count}; fewer than n in all when every cell is at its cap."""
    want = dict.fromkeys(CELLS, 0)
    for _ in range(n):
        free = [c for c in CELLS if want[c] < cap[c]]
        if not free:
            break
        want[min(free, key=lambda c: (old[c] + want[c], c))] += 1
    return want


def expected_kept(q, n, p):
    """The mean of min(q, Binomial(n, p)): the episodes n listed seeds give toward a quota of q when the teacher
    keeps a scene with chance p."""
    return sum(min(q, k) * math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1))


def kept_so_far(logs):
    """generate_truss.py attempt logs of the parts already recorded: kept episodes per cell ('hold' for holds)."""
    c = collections.Counter()
    for f in logs:
        for line in open(f):
            r = json.loads(line) if line.strip() else {}
            if r.get("kept") and "seed" in r:           # the log's last line is a summary with its own "kept"
                c["hold" if r.get("hold") else r.get("cell", "?")] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--part-size", type=int, default=25, help="as dataset v5's parts (record_parts.sh)")
    ap.add_argument("--hold-share", type=float, default=gt.HOLD_SHARE, help="nothing-ripe episodes (v5: 64 of 800)")
    ap.add_argument("--oversupply", type=float, default=OVERSUPPLY)
    ap.add_argument("--min-supply", type=float, default=MIN_SUPPLY)
    ap.add_argument("--used", nargs="*", default=[], help="seed plans already written (p*.jsonl): their seeds are "
                                                         "left out of this plan")
    ap.add_argument("--topup", nargs="*", default=[], help="attempt logs of recorded parts: plan only what they "
                                                          "left missing of the full plan, as extra parts")
    ap.add_argument("--plan", help="with --topup: the plan.json those parts were recorded from")
    ap.add_argument("--first-part", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    files = [f for pat in a.census for f in glob.glob(pat)]
    pool, none, n_seeds = census(files)
    used = set()
    for pat in a.used:
        for f in glob.glob(pat):
            used |= {json.loads(x)["seed"] for x in open(f) if x.strip() and '"seed"' in x}
    pool = {c: [s for s in v if s not in used] for c, v in pool.items()}
    none = [s for s in none if s not in used]
    old = old_cells([f for pat in a.old for f in glob.glob(pat)])
    n_hold = round(a.hold_share * a.episodes)
    n_cut = a.episodes - n_hold
    # per cell: level old + new, as far as the census seeds reach (level)
    cap = {c: int(len(pool.get(c, [])) / a.min_supply) for c in CELLS}
    if a.topup:                    # what the recorded parts left missing of the plan they were recorded from
        full = json.load(open(a.plan))["want"] if a.plan else level(old, n_cut, dict.fromkeys(CELLS, n_cut)) | {
            "hold": n_hold}
        got = kept_so_far([f for pat in a.topup for f in glob.glob(pat)])
        want = {c: min(max(0, full.get(c, 0) - got[c]), cap[c]) for c in CELLS}
        n_cut, n_hold = sum(want.values()), max(0, full.get("hold", n_hold) - got["hold"])
        print(f"top-up: {sum(got.values())} episodes kept so far, {n_cut + n_hold} planned now")
    else:
        want = level(old, n_cut, cap)
        if sum(want.values()) < n_cut:
            print(f"⚠️ the census has seeds for only {sum(want.values())} of the {n_cut} cut episodes at "
                  f"{a.min_supply} seeds per episode: more census seeds are needed")
            n_cut = sum(want.values())
    capped = [c for c in CELLS if want[c] == cap[c] and not a.topup]
    want["hold"] = n_hold
    n_parts = math.ceil((n_cut + n_hold) / a.part_size)
    # deal the wanted episodes to the parts: each cell's episodes spread evenly over the parts
    labels = sorted(((k + 0.5) / want[c], c) for c in want for k in range(want[c]))
    parts = [collections.Counter() for _ in range(n_parts)]
    for i, (_, c) in enumerate(labels):
        parts[i % n_parts][c] += 1
    print(f"census: {n_seeds} seeds, {sum(len(v) for v in pool.values())} with a ripe truss, {len(none)} with none"
          f"{f' ({len(used)} seeds already planned are left out)' if used else ''}")
    print(f"plan: {n_cut} cut + {n_hold} nothing-ripe episodes in {n_parts} parts of up to {a.part_size}; per cell up "
          f"to {a.oversupply} seeds listed per episode, and no more episodes than the census has {a.min_supply} "
          f"seeds for")
    print("\n| Where the cut point hangs | Old 800 | New | Together | Census seeds | Seeds per new episode |\n"
          "|---|---|---|---|---|---|")
    xs = np.linspace(*gt.ts.STEM_X, gt.BALANCE_BINS + 1)
    zs = ("below 0.20 m", "0.20-0.26 m", "above 0.26 m")
    for c in CELLS:
        i, j = int(c[1]), int(c[3])
        ratio = len(pool.get(c, [])) / want[c] if want[c] else float("inf")
        print(f"| {c}: row {xs[i]:.2f}-{xs[i + 1]:.2f} m, {zs[j]} | {old[c]} | {want[c]} | {old[c] + want[c]} | "
              f"{len(pool.get(c, []))} | {'-' if not want[c] else f'{ratio:.1f}'} |")
    print(f"| all | {sum(old.values())} | {n_cut} | {sum(old.values()) + n_cut} | "
          f"{sum(len(v) for v in pool.values())} | |")
    if capped:
        print(f"\n⚠️ held back by the census ({a.min_supply} seeds per episode): {capped}. Their share went to the "
              f"other cells; more census seeds would level them further")
    write = not (a.dry or a.out is None)       # --dry deals the seeds too (for the expected count), writes nothing
    if write:
        a.out.mkdir(parents=True, exist_ok=True)
    # the nothing-ripe seeds first, out of the pools (any scene can be one): scenes with no ripe truss, then from the
    # far end of the cells with the most seeds to spare, so no seed is both a hold and a cut episode
    pool = {c: list(v) for c, v in pool.items()}
    n_hold_seeds = sum(q["hold"] + 2 for q in parts if q["hold"])            # two spare per part
    hold_seeds = list(none[:n_hold_seeds])
    while len(hold_seeds) < n_hold_seeds:
        spare = {c: len(pool.get(c, [])) - math.ceil(want[c] * a.oversupply) for c in CELLS if pool.get(c)}
        c = max(spare, key=lambda c: (spare[c], c)) if spare else None
        if c is None or spare[c] <= 0:
            break
        hold_seeds.append(pool[c].pop())
    hold_seeds = iter(hold_seeds)
    # per cell: the seeds a part gets = its share of the cell's list, the list being min(OVERSUPPLY, what the census
    # has) seeds per episode, dealt in running totals so the last part is not the one left without
    per_ep = {c: min(a.oversupply, len(pool.get(c, [])) / want[c]) for c in CELLS if want[c]}
    given, asked = collections.Counter(), collections.Counter()
    taken, summary, expect = set(), [], collections.Counter()
    for k, q in enumerate(parts):
        rows_by = {}
        for c in CELLS:
            if q[c]:
                asked[c] += q[c]
                n = min(len(pool[c]), round(asked[c] * per_ep[c])) - given[c]
                rows_by[c] = [dict(seed=s, cell=c, hold=False) for s in pool[c][given[c]:given[c] + n]]
                given[c] += n
                assert not taken & {r["seed"] for r in rows_by[c]}
                taken |= {r["seed"] for r in rows_by[c]}
                expect[c] += expected_kept(q[c], n, KEEP_RATE[c])
        rows = [r for grp in zip_longest_lists(list(rows_by.values())) for r in grp]
        holds = []
        while len(holds) < q["hold"] + 2 and q["hold"]:
            s = next(hold_seeds, None)
            if s is None:
                break
            assert s not in taken
            taken.add(s)
            holds.append(dict(seed=s, hold=True))
        step = max(1, len(rows) // (len(holds) + 1))
        for i, h in enumerate(holds):                      # the holds spread through the list
            rows.insert(min(len(rows), (i + 1) * step + i), h)
        name = f"p{a.first_part + k:02d}.jsonl"
        if write:
            with open(a.out / name, "w") as out:
                out.write(json.dumps(dict(quota={c: int(n) for c, n in q.items() if n}, part=a.first_part + k,
                                          made_by="sim/scripts/v6_seed_plan.py",
                                          episodes=int(sum(q.values())))) + "\n")
                for r in rows:
                    out.write(json.dumps(r) + "\n")
        summary.append(dict(part=a.first_part + k, file=name, episodes=int(sum(q.values())), seeds=len(rows),
                            quota={c: int(n) for c, n in q.items() if n}))
    # what the lists should give: a cell of a part ends short when the teacher refuses more of its seeds than the
    # list has spare (a part's seeds are its own: the parts record side by side)
    short = {c: want[c] - expect[c] for c in CELLS if want[c]}
    worst = sorted(short, key=short.get, reverse=True)[:3]
    print(f"\n{len(taken)} seeds listed. At the keep rates of dataset v5's recording ({min(KEEP_RATE.values()):.0%}-"
          f"{max(KEEP_RATE.values()):.0%} per cell) the lists should give about {sum(expect.values()):.0f} of the "
          f"{n_cut} cut episodes, so about {sum(short.values()):.0f} short (most in "
          f"{', '.join(f'{c} {short[c]:.1f}' for c in worst)}): --topup plans those as one more part")
    if not write:
        return
    # a top-up never replaces the plan it tops up (--plan reads that one again for a second top-up)
    plan_name = f"plan_topup_p{a.first_part:02d}.json" if a.topup else "plan.json"
    json.dump(dict(episodes=n_cut + n_hold, cut=n_cut, hold=n_hold, old=dict(old), want={c: int(v) for c, v in
                   want.items()}, pool={c: len(v) for c, v in pool.items()}, oversupply=a.oversupply,
                   min_supply=a.min_supply, held_back=capped, census=sorted(files), parts=summary,
                   expected_cut_kept=round(sum(expect.values()), 1)),
              open(a.out / plan_name, "w"), indent=1)
    print(f"wrote {len(parts)} seed plans to {a.out} (p{a.first_part:02d}.jsonl ...) and {plan_name}")


def zip_longest_lists(groups):
    """Rows of several lists taking turns: [[a1, b1, c1], [a2, b2], ...]."""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.append([g[i] for g in groups if i < len(g)])
    return out


if __name__ == "__main__":
    main()
