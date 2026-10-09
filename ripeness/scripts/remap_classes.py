"""remap_classes.py: rewrite YOLO label files IN PLACE to {0: unripe, 1: turning, 2: ripe}.

    python ripeness/scripts/remap_classes.py <labels_dir> <dataset_name>

Boxes whose source class maps to None are dropped. Run it once per dataset: running it
twice remaps already-remapped ids.
"""
import pathlib
import sys

# Source class name -> our class id. None = drop that box.
MAPS = {
    "laboro": {"l_green": 0, "l_half_ripened": 1, "l_fully_ripened": 2,
               "b_green": None, "b_half_ripened": None, "b_fully_ripened": None},   # b_ = normal-size, not cherry
    "mendeley": {"Green": 0, "Half_ripened": 1, "Fully_ripened": 2},
}
# Source id -> name, copied from each dataset's own yaml/json.
# ⚠️ Both orders are UNVERIFIED. Check them against the dataset files before running,
#    because a wrong order silently swaps ripe and unripe.
SOURCE_NAMES = {
    "laboro": ["b_fully_ripened", "b_half_ripened", "b_green",
               "l_fully_ripened", "l_half_ripened", "l_green"],
    "mendeley": ["Fully_ripened", "Half_ripened", "Green"],
}

labels_dir, name = pathlib.Path(sys.argv[1]), sys.argv[2]
kept = dropped = 0
for f in labels_dir.rglob("*.txt"):
    out = []
    for line in f.read_text().splitlines():
        cid, *box = line.split()
        new = MAPS[name][SOURCE_NAMES[name][int(cid)]]
        if new is None:
            dropped += 1
            continue
        out.append(" ".join([str(new), *box]))
        kept += 1
    f.write_text("\n".join(out) + ("\n" if out else ""))
print(f"{name}: kept {kept} boxes, dropped {dropped}")
