"""prepare_negatives.py: Open Images photos of red / yellow / green things that are NOT tomatoes,
with the look-alikes boxed as class 3 `not_tomato`, for the ripeness detector.

    python ripeness/scripts/prepare_negatives.py

Writes ripeness/data/public/negatives_oi/
    images/{train,val,test}/<id>.jpg   labels/{train,val,test}/<id>.txt  `3 cx cy w h` per look-alike, or empty
    manifest.csv                       image id, Open Images split, the distractor class it was drawn for, our split
    test_no_piles.txt                  test image ids without group-of boxes (see eval_negatives.py)
    _meta/                             the Open Images annotation CSVs it read

Why: Laboro has a tomato in nearly every frame, so "red/green/yellow and round" was enough to score
well, and the pretrain boxed red balls, peppers etc. as tomatoes. These images teach "this colour,
but not a tomato". The look-alikes get Open Images' own boxes as class 3 `not_tomato`, so live_view shows
what the model thinks a thing is (pretrain_v3). Unboxed parts of the image stay background.
pretrain_v2 was trained on the same images with EMPTY labels (commit 852f5d7).

`test` is never trained or tuned on (build_split.py skips it). eval_negatives.py reports the
false-positive rate (tomato classes on these images) and `not_tomato` recall on it.

⚠️ Open Images labels aren't exhaustive. Any image with a Tomato box or a positive Tomato image label
   is dropped, but a salad or market shot can still hide an unlabelled tomato, which would teach
   "tomato = background". Look at the contact sheet (contact.jpg) before training.
⚠️ Licence: annotations CC BY 4.0, images listed by Open Images as CC BY 2.0 (attribution).
"""
import concurrent.futures as cf
import hashlib
import pathlib
import urllib.request

import cv2
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DST = ROOT / "ripeness/data/public/negatives_oi"
META = DST / "_meta"
GCS = "https://storage.googleapis.com/openimages"
CSVS = {"classes.csv": "v7/oidv7-class-descriptions-boxable.csv",
        "validation-bbox.csv": "v5/validation-annotations-bbox.csv",
        "test-bbox.csv": "v5/test-annotations-bbox.csv",
        "validation-labels.csv": "v5/validation-annotations-human-imagelabels-boxable.csv",
        "test-labels.csv": "v5/test-annotations-human-imagelabels-boxable.csv"}
# Images with any of these are dropped too: meals hide unlabelled tomato slices (first contact sheet,
# 2026-10-12: a cucumber salad, a breakfast plate and a pizza, all with tomato, none labelled Tomato).
EXCLUDE = ["Tomato", "Food", "Salad", "Pizza", "Fast food", "Sandwich", "Submarine sandwich", "Hamburger",
           "Hot dog", "Pasta", "Taco", "Burrito"]
# Dropped by eye after sampling: a market stall and a cucumber crate, both with real tomatoes in view.
# Dropped after the draw, so the rest of the sample stays what was reviewed.
REVIEWED_DROP = {"89ac39a9dc0d169c", "4df9dc09efc766c3"}
PER_CLASS = 30    # images drawn per distractor class. ~25 classes -> ~600 after de-duplication
# v4 (2026-10-13): a second draw for train/val only, because `not_tomato` recall was low (0.28 on test
# with v3). Extra images whose hash lands in `test` are discarded, so the 312 held-out images never change.
EXTRA_PER_CLASS = 150   # many draws are lost to the group-of drop; 45 only took train from 299 to 379
SEED = 0
# Our split by a hash of the image id: stable across reruns. Train negatives end up ~40 % of Laboro's
# 643 train images. More than Ultralytics' usual 0-10 % background advice, because false positives
# are the problem being fixed. Watch `unripe` recall (green apples / cucumbers vs green tomatoes).
SPLIT = (("train", 45), ("val", 55), ("test", 100))   # cumulative % of the hash

NOT_TOMATO = 3   # class id, LABELLING.md
# Look-alikes whose Open Images boxes become `not_tomato`. Not boxed: Christmas tree, Toy, Traffic light,
# whose boxes cover the whole tree / toy / signal rather than a fruit-sized part. Those images stay background.
UNBOXED = {"Christmas tree", "Toy", "Traffic light"}

# Things a claw camera might see that share a tomato's colour or shape.
DISTRACTORS = [
    # red
    "Apple", "Bell pepper", "Strawberry", "Pomegranate", "Radish", "Cricket ball", "Balloon",
    "Rose", "Christmas tree", "Lantern", "Traffic light", "Toy",
    # yellow / orange
    "Tennis ball", "Banana", "Peach", "Mango", "Grapefruit", "Orange (fruit)", "Lemon (plant)", "Pumpkin", "Football",
    # green (and trusses that look like cherry-tomato clusters)
    "Cucumber", "Zucchini", "Grape",
]


def fetch_csvs():
    META.mkdir(parents=True, exist_ok=True)
    for local, remote in CSVS.items():
        if not (META / local).exists():
            print("download", remote)
            urllib.request.urlretrieve(f"{GCS}/{remote}", META / local)


def our_split(image_id):
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100
    return next(name for name, upto in SPLIT if h < upto)


def pick():
    classes = pd.read_csv(META / "classes.csv", header=None, names=["mid", "name"])
    mid = dict(zip(classes.name, classes.mid))
    missing = [d for d in DISTRACTORS + EXCLUDE if d not in mid]
    assert not missing, f"not Open Images boxable classes: {missing}"
    rng = np.random.default_rng(SEED)
    rows, taken = [], set()
    for oi_split in ("validation", "test"):
        bbox = pd.read_csv(META / f"{oi_split}-bbox.csv", usecols=["ImageID", "LabelName"])
        labels = pd.read_csv(META / f"{oi_split}-labels.csv")
        bad = {mid[e] for e in EXCLUDE}
        dropped = set(bbox[bbox.LabelName.isin(bad)].ImageID) | \
                 set(labels[labels.LabelName.isin(bad) & (labels.Confidence == 1)].ImageID)
        for d in DISTRACTORS:
            ids = sorted(set(bbox[bbox.LabelName == mid[d]].ImageID) - dropped - taken)
            # Half the quota from each Open Images split, so neither runs dry for the rarer classes.
            n = min(len(ids), PER_CLASS // 2 if oi_split == "validation" else PER_CLASS - sum(r[2] == d for r in rows))
            for i in rng.choice(ids, size=n, replace=False):
                rows.append((i, oi_split, d, our_split(i)))
                taken.add(i)
    # Second draw, same rng stream, so the first draw (and its test set) is unchanged.
    for oi_split in ("validation", "test"):
        bbox = pd.read_csv(META / f"{oi_split}-bbox.csv", usecols=["ImageID", "LabelName"])
        labels = pd.read_csv(META / f"{oi_split}-labels.csv")
        dropped = set(bbox[bbox.LabelName.isin(bad)].ImageID) | \
                  set(labels[labels.LabelName.isin(bad) & (labels.Confidence == 1)].ImageID)
        for d in DISTRACTORS:
            ids = sorted(set(bbox[bbox.LabelName == mid[d]].ImageID) - dropped - taken)
            for i in rng.choice(ids, size=min(len(ids), EXTRA_PER_CLASS // 2), replace=False):
                taken.add(i)
                if our_split(i) != "test":
                    rows.append((i, oi_split, d, our_split(i)))
    df = pd.DataFrame(rows, columns=["image_id", "oi_split", "distractor", "split"])
    return df[~df.image_id.isin(REVIEWED_DROP)]


def not_tomato_boxes(image_ids):
    """image id -> YOLO lines for every boxed look-alike in it, not just the class it was drawn for."""
    classes = pd.read_csv(META / "classes.csv", header=None, names=["mid", "name"])
    mid = dict(zip(classes.name, classes.mid))
    boxed = {mid[d] for d in DISTRACTORS if d not in UNBOXED}
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf"]
    b = pd.concat([pd.read_csv(META / f"{s}-bbox.csv", usecols=cols) for s in ("validation", "test")])
    # ⚠️ IsGroupOf boxes cover a whole pile with one box. Dropped: they'd teach pile-sized `not_tomato`
    #    boxes. The fruit in such a pile is left as background, as in v2.
    b = b[b.ImageID.isin(image_ids) & b.LabelName.isin(boxed)]
    grouped = set(b[b.IsGroupOf == 1].ImageID)
    b = b[b.IsGroupOf == 0]
    out = {}
    for r in b.itertuples():   # Open Images boxes are already normalised 0-1
        out.setdefault(r.ImageID, []).append(
            f"{NOT_TOMATO} {(r.XMin + r.XMax) / 2:.6f} {(r.YMin + r.YMax) / 2:.6f} {r.XMax - r.XMin:.6f} {r.YMax - r.YMin:.6f}")
    return out, grouped


def drop_conflicting(df, boxes, grouped):
    """Train/val only: drop images whose look-alikes would be background in one image and `not_tomato`
    in the next. That's any image with a group-of box (the pile goes unboxed), and any image drawn for a
    boxed class that ended up with no box. Test is left as drawn, so every version is scored on the same set."""
    unboxed_draw = ~df.distractor.isin(UNBOXED) & ~df.image_id.isin(boxes.keys())
    bad = (df.split != "test") & (df.image_id.isin(grouped) | unboxed_draw)
    print(f"dropped {bad.sum()} train/val images with group-of or missing look-alike boxes")
    return df[~bad]


def download(row, lines):
    out = DST / "images" / row.split / f"{row.image_id}.jpg"
    if not out.exists():
        urllib.request.urlretrieve(f"https://s3.amazonaws.com/open-images-dataset/{row.oi_split}/{row.image_id}.jpg", out)
    (DST / "labels" / row.split / f"{row.image_id}.txt").write_text("".join(f"{l}\n" for l in lines))
    return out


def contact_sheet(paths, n=240, cell=120, cols=16):
    # One look at ~120 thumbnails catches the unlabelled-tomato images the CSV filter can't.
    rng = np.random.default_rng(SEED)
    paths = [paths[i] for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False)]
    rows = -(-len(paths) // cols)
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for k, p in enumerate(paths):
        im = cv2.imread(str(p))
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        y, x = (k // cols) * cell, (k % cols) * cell
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(sheet, p.stem[:6], (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(DST / "contact.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


fetch_csvs()
df = pick()
boxes, grouped = not_tomato_boxes(set(df.image_id))
df = drop_conflicting(df, boxes, grouped)
for s, _ in SPLIT:
    (DST / "images" / s).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / s).mkdir(parents=True, exist_ok=True)
with cf.ThreadPoolExecutor(16) as ex:
    paths = list(ex.map(lambda r: download(r, boxes.get(r.image_id, [])), df.itertuples()))
df.to_csv(DST / "manifest.csv", index=False)
# Test keeps its group-of images, whose piles are unlabelled, so a correct `not_tomato` box on one fruit of a
# pile scores as a false positive there. eval_negatives.py also reports `not_tomato` on this pile-free subset.
(DST / "test_no_piles.txt").write_text("".join(f"{i}\n" for i in sorted(df.image_id[(df.split == "test") & ~df.image_id.isin(grouped)])))
contact_sheet(paths)
print(df.groupby(["distractor", "split"]).size().unstack(fill_value=0).to_string())
print(df.split.value_counts().to_string())
kept = df.image_id[df.image_id.isin(boxes.keys())]
print(f"`not_tomato` boxes: {sum(len(boxes[i]) for i in kept)} in {len(kept)} of {len(df)} images (the rest are background)")
print(f"wrote {DST.relative_to(ROOT)}/  -> look at contact.jpg, then add `negatives_oi` to splits.yaml")
