"""build_split.py: build the YOLO image lists + dataset yamls from data/splits.yaml.

    python ripeness/scripts/build_split.py

Writes ripeness/data/yolo/ (gitignored):
    train.txt / val.txt / test.txt     own + public (test = own frozen sessions only)
    train_public.txt / val_public.txt  public only, for the pretrain stage
    ripeness.yaml / public.yaml        point Ultralytics at those lists

Nothing is copied. The lists hold absolute image paths, and Ultralytics finds each label
by swapping the last /images/ in the path for /labels/.
"""
import hashlib
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "ripeness/data"
OUT = DATA / "yolo"
NAMES = {0: "unripe", 1: "turning", 2: "ripe"}   # must match LABELLING.md
IMG_EXT = {".jpg", ".jpeg", ".png"}


def images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT and "/images/" in p.as_posix())


def is_val(path, fraction):
    # Deterministic per-file split: the same file lands on the same side on every rebuild.
    return int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 1000 < fraction * 1000


splits = yaml.safe_load((DATA / "splits.yaml").read_text())
test_s, val_s = set(splits.get("test") or []), set(splits.get("val") or [])
assert not test_s & val_s, f"session in both val and test: {test_s & val_s}"

lists = {k: [] for k in ("train", "val", "test", "train_public", "val_public")}

own_root = DATA / "own/images"
sessions = sorted(p.name for p in own_root.iterdir() if p.is_dir()) if own_root.exists() else []
missing = (test_s | val_s) - set(sessions)
assert not missing, f"splits.yaml names sessions that don't exist: {missing}"
for s in sessions:
    imgs = images(own_root / s)
    unlabelled = [p for p in imgs if not (DATA / "own/labels" / s / (p.stem + ".txt")).exists()]
    if unlabelled:
        print(f"⚠️ {s}: {len(unlabelled)} frames have no label file. They train as background (no fruit).")
    lists["test" if s in test_s else "val" if s in val_s else "train"] += imgs

for name, opts in (splits.get("public") or {}).items():
    folder = DATA / "public" / name
    if not folder.exists():
        print(f"skip public set {name}: {folder.relative_to(ROOT)} not downloaded")
        continue
    for p in images(folder):
        side = "val" if is_val(p, opts.get("val_fraction", 0.1)) else "train"
        lists[side].append(p)
        lists[f"{side}_public"].append(p)

OUT.mkdir(parents=True, exist_ok=True)
for k, v in lists.items():
    (OUT / f"{k}.txt").write_text("".join(f"{p}\n" for p in v))
    print(f"{k:>13}: {len(v)} images")

for fname, tr, va, te in (("ripeness.yaml", "train", "val", "test"),
                          ("public.yaml", "train_public", "val_public", None)):
    cfg = {"path": str(OUT), "train": f"{tr}.txt", "val": f"{va}.txt", "names": NAMES}
    if te:
        cfg["test"] = f"{te}.txt"
    (OUT / fname).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {OUT.relative_to(ROOT)}/ripeness.yaml and public.yaml")
