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
A cut episode's cell = generate_truss.cell_of(cut point): 4 parts along the row x 3 heights. Per cell the plan asks
for what brings old + new to the same total (cells already above it get none), split over the parts as evenly as it
goes; each part's file lists, per cell, OVERSUPPLY times the seeds its quota needs (the teacher fails some), the
cells taking turns, and the part's nothing-ripe seeds. generate_truss.py --seeds reads it (read_seed_plan): a full
cell's seeds are skipped without building a scene.
⚠️ A part can end short when a cell's seeds run out; --topup LOGS... plans one more part for what is missing.
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
OVERSUPPLY = 2.5        # seeds listed per episode wanted. Dataset v5's log: the teacher kept 736 of 1,044 cut scenes
#                         it tried = 70 %; per slot 17-83 %, so 2.5 leaves room for a cell at 40 %


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


def split(total, weights):
    """Whole numbers summing to total, proportional to weights (largest remainders)."""
    w = np.asarray(weights, float)
    raw = total * w / w.sum()
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[:total - out.sum()]:
        out[i] += 1
    return out


def kept_so_far(logs):
    """generate_truss.py attempt logs of the parts already recorded: kept episodes per cell ('hold' for holds)."""
    c = collections.Counter()
    for f in logs:
        for line in open(f):
            r = json.loads(line) if line.strip() else {}
            if r.get("kept"):
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
    ap.add_argument("--used", nargs="*", default=[], help="seed plans already written (p*.jsonl): their seeds are "
                                                         "left out of this plan")
    ap.add_argument("--topup", nargs="*", default=[], help="attempt logs of recorded parts: plan only what they "
                                                          "left missing of the full plan, as extra parts")
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
    # per cell: what brings old + new to one level (water filling: cells above the level get none)
    lvl = (sum(old.values()) + n_cut) / len(CELLS)
    for _ in range(len(CELLS)):
        low = [c for c in CELLS if old[c] < lvl]
        new_lvl = (sum(old[c] for c in low) + n_cut) / len(low)
        if abs(new_lvl - lvl) < 1e-9:
            break
        lvl = new_lvl
    want = dict(zip(CELLS, split(n_cut, [max(0.0, lvl - old[c]) for c in CELLS])))
    want["hold"] = n_hold
    if a.topup:
        got = kept_so_far([f for pat in a.topup for f in glob.glob(pat)])
        want = {c: max(0, want[c] - got[c]) for c in want}
        n_cut, n_hold = sum(v for c, v in want.items() if c != "hold"), want["hold"]
        print(f"top-up: {sum(got.values())} episodes kept so far, {n_cut + n_hold} missing")
    n_parts = math.ceil((n_cut + n_hold) / a.part_size)
    # deal the wanted episodes to the parts: each cell's episodes spread evenly over the parts
    labels = sorted(((k + 0.5) / want[c], c) for c in want for k in range(want[c]))
    parts = [collections.Counter() for _ in range(n_parts)]
    for i, (_, c) in enumerate(labels):
        parts[i % n_parts][c] += 1
    print(f"census: {n_seeds} seeds, {sum(len(v) for v in pool.values())} with a ripe truss, {len(none)} with none"
          f"{f' ({len(used)} seeds already planned are left out)' if used else ''}")
    print(f"plan: {n_cut} cut + {n_hold} nothing-ripe episodes in {n_parts} parts of up to {a.part_size}; a seed list "
          f"of {a.oversupply} x the quota per cell")
    print("\n| Where the cut point hangs | Old 800 | New | Together | Census seeds | Seeds per new episode |\n"
          "|---|---|---|---|---|---|")
    xs = np.linspace(*gt.ts.STEM_X, gt.BALANCE_BINS + 1)
    zs = ("below 0.20 m", "0.20-0.26 m", "above 0.26 m")
    short = {}
    for c in CELLS:
        i, j = int(c[1]), int(c[3])
        ratio = len(pool.get(c, [])) / want[c] if want[c] else float("inf")
        if want[c] and ratio < a.oversupply:
            short[c] = ratio
        print(f"| {c}: row {xs[i]:.2f}-{xs[i + 1]:.2f} m, {zs[j]} | {old[c]} | {want[c]} | {old[c] + want[c]} | "
              f"{len(pool.get(c, []))} | {'-' if not want[c] else f'{ratio:.1f}'} |")
    print(f"| all | {sum(old.values())} | {n_cut} | {sum(old.values()) + n_cut} | "
          f"{sum(len(v) for v in pool.values())} | |")
    if short:
        print(f"\n⚠️ fewer than {a.oversupply} census seeds per wanted episode in {short}: those cells may end short "
              f"(more census seeds, or accept it)")
    if a.dry or a.out is None:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    take = {c: iter(v) for c, v in pool.items()}
    hold_seeds = iter(none + [s for c in CELLS for s in pool.get(c, [])[::-1]])   # holds: any seed; from the far end
    taken, summary = set(), []
    for k, q in enumerate(parts):
        rows_by = {}
        for c in CELLS:
            if q[c]:
                rows_by[c] = [dict(seed=s, cell=c, hold=False) for s in
                              (next(take[c], None) for _ in range(math.ceil(q[c] * a.oversupply) + 1))
                              if s is not None]
                taken |= {r["seed"] for r in rows_by[c]}
        rows = [r for grp in zip_longest_lists(list(rows_by.values())) for r in grp]
        holds = []
        while len(holds) < q["hold"] + 2 and q["hold"]:
            s = next(hold_seeds, None)
            if s is None:
                break
            if s not in taken:
                taken.add(s)
                holds.append(dict(seed=s, hold=True))
        step = max(1, len(rows) // (len(holds) + 1))
        for i, h in enumerate(holds):                      # the holds spread through the list
            rows.insert(min(len(rows), (i + 1) * step + i), h)
        f = a.out / f"p{a.first_part + k:02d}.jsonl"
        with open(f, "w") as out:
            out.write(json.dumps(dict(quota={c: int(n) for c, n in q.items() if n}, part=a.first_part + k,
                                      made_by="sim/scripts/v6_seed_plan.py", episodes=int(sum(q.values())))) + "\n")
            for r in rows:
                out.write(json.dumps(r) + "\n")
        summary.append(dict(part=a.first_part + k, file=f.name, episodes=int(sum(q.values())), seeds=len(rows),
                            quota={c: int(n) for c, n in q.items() if n}))
    json.dump(dict(episodes=n_cut + n_hold, cut=n_cut, hold=n_hold, old=dict(old), want={c: int(v) for c, v in
                   want.items()}, pool={c: len(v) for c, v in pool.items()}, oversupply=a.oversupply,
                   census=sorted(files), parts=summary), open(a.out / "plan.json", "w"), indent=1)
    print(f"\nwrote {len(parts)} seed plans to {a.out} (p{a.first_part:02d}.jsonl ...) and plan.json; "
          f"{len(taken)} seeds listed")


def zip_longest_lists(groups):
    """Rows of several lists taking turns: [[a1, b1, c1], [a2, b2], ...]."""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.append([g[i] for g in groups if i < len(g)])
    return out


if __name__ == "__main__":
    main()
