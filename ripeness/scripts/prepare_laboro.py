"""prepare_laboro.py: Laboro Tomato (COCO JSON) -> YOLO layout with our 3 classes.

    python ripeness/scripts/prepare_laboro.py                 # cherry (l_) + regular (b_) tomatoes
    python ripeness/scripts/prepare_laboro.py --cherry-only   # l_ only, as the vault note first planned

Reads  ripeness/data/public/laboro/laboro_tomato/{train,test,annotations}  (unzipped laboro_tomato.zip)
Writes ripeness/data/public/laboro_yolo/{images,labels}/{train,val}/       (Laboro's test -> our val)

Laboro's own train/test split is kept rather than re-split by hash, because its photos come in
near-duplicate runs and a random split would leak them across.

⚠️ Don't use ultralytics' convert_coco for this: it defaults to cls91to80=True, which treats the
   ids as COCO-91 and scrambles the classes.
⚠️ CC BY-NC-SA 4.0: anything trained on this is demo-only, never a product.
"""
import json
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "ripeness/data/public/laboro/laboro_tomato"
DST = ROOT / "ripeness/data/public/laboro_yolo"
MAX_SIDE = 1280   # originals are 3120x4160 / 3024x4032. Training runs at 640, so 1280 keeps 2x headroom
                  # for mosaic/scale augmentation while cutting JPEG decode time ~10x.

# Laboro category name -> our class id (LABELLING.md). Its stages are by % red: green <30,
# half_ripened 30-89, fully_ripened >=90, so `ripe` here is the strict end of ours (the safe side).
MAP = {"l_green": 0, "l_half_ripened": 1, "l_fully_ripened": 2,
       "b_green": 0, "b_half_ripened": 1, "b_fully_ripened": 2}
if "--cherry-only" in sys.argv:
    MAP = {k: (v if k.startswith("l_") else None) for k, v in MAP.items()}

for split_in, split_out in (("train", "train"), ("test", "val")):
    coco = json.loads((SRC / "annotations" / f"{split_in}.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    (DST / "images" / split_out).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split_out).mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for im in coco["images"]:
        img = cv2.imread(str(SRC / split_in / im["file_name"]))   # IMREAD_COLOR applies EXIF rotation
        W, H = im["width"], im["height"]
        # ⚠️ Boxes are in the JSON's width/height frame. If the decoded image disagrees, EXIF rotation
        #    and the annotations don't line up, and the labels would be silently wrong.
        assert img.shape[1] == W and img.shape[0] == H, f"{im['file_name']}: decoded {img.shape[1]}x{img.shape[0]} != json {W}x{H}"
        lines = []
        for a in anns.get(im["id"], []):
            cls = MAP[cats[a["category_id"]]]
            if cls is None:
                continue
            x, y, w, h = a["bbox"]   # COCO: top-left x, y, width, height in pixels
            lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        s = MAX_SIDE / max(W, H)
        if s < 1:
            img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        stem = pathlib.Path(im["file_name"]).stem
        cv2.imwrite(str(DST / "images" / split_out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (DST / "labels" / split_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    print(f"{split_in} -> {split_out}: {n_img} images, {n_box} boxes")
