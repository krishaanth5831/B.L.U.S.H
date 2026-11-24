#!/usr/bin/env python3
"""Tables of one or more catch maps (sim/catch_map.py logs): per axis and offset, of the scenes tested there, how many
the jaw cut, still held 0.5 s after the cut, and fully succeeded, and at the cut how often both pads pressed.

    python3 sim/scripts/catch_map_table.py outputs/v6/catch_map_jaw_v5_part*.jsonl
    python3 sim/scripts/catch_map_table.py v5=outputs/v6/catch_map_jaw_v5_part*.jsonl v6=outputs/v6/catch_map_jaw_v6_part*.jsonl

Only scenes with every offset of an axis logged count (a stopped run's half-done last scene is left out). Plain python3.
"""
import collections
import glob
import json
import sys

PAD_MIN_N = 0.5                 # a pad "presses" at the cut above this (N); a grazing contact is a few tenths


def load(paths):
    rows = [json.loads(line) for p in paths for f in glob.glob(p) for line in open(f) if line.strip()]
    by = collections.defaultdict(dict)                 # (seed, axis) -> {offset: row}
    base = {r["seed"]: r for r in rows if r["axis"] == "none"}
    for r in rows:
        if r["axis"] != "none" and base.get(r["seed"], {}).get("ok"):
            by[(r["seed"], r["axis"])][r["offset_mm"]] = r
    return base, by


def table(name, paths):
    base, by = load(paths)
    full = collections.defaultdict(list)               # axis -> [(seed, {offset: row})] with every offset logged
    n_off = {}
    for (seed, axis), offs in by.items():
        n_off[axis] = max(n_off.get(axis, 0), len(offs))
    for (seed, axis), offs in by.items():
        if len(offs) == n_off[axis]:
            full[axis].append((seed, offs))
    print(f"\n**{name}**: {sum(1 for r in base.values() if r['ok'])} scenes whose first plan works aimed exactly")
    for axis in ("side", "depth", "height"):
        scenes = full.get(axis, [])
        if not scenes:
            continue
        print(f"\n{axis} ({len(scenes)} scenes) | offset (mm) | cut | held 0.5 s | success | both pads pressing at the "
              f"cut | fruit off")
        print("|---|---|---|---|---|---|")
        offsets = sorted({o for _, offs in scenes for o in offs} | {0})
        for o in offsets:
            rs = [base[s] if o == 0 else offs[o] for s, offs in scenes]
            n = len(rs)
            cut = sum(bool(r.get("cut")) for r in rs)
            held = sum(r.get("held") is True for r in rs)
            ok = sum(bool(r.get("ok")) for r in rs)
            both = sum(1 for r in rs if r.get("cut") and min(r.get("pad_fixed_n") or 0, r.get("pad_moving_n") or 0)
                       > PAD_MIN_N)
            off = sum((r.get("fruit_off") or 0) > 0 for r in rs)
            print(f"| {o:+d} | {cut}/{n} | {held}/{n} | {ok}/{n} | {both}/{cut} | {off}/{n} |")


def main():
    groups = collections.OrderedDict()
    for a in sys.argv[1:]:
        name, _, path = a.rpartition("=")
        groups.setdefault(name or "catch map", []).append(path)
    for name, paths in groups.items():
        table(name, paths)


if __name__ == "__main__":
    main()
