"""convert_coco.py: COCO JSON annotations -> YOLO .txt labels (boxes only).

    python ripeness/scripts/convert_coco.py <coco_annotations_dir> <save_dir>

Laboro ships COCO JSON. The output class ids are still the SOURCE ids, so run
remap_classes.py on the result before training.
⚠️ Ultralytics writes save_dir/labels/<json_name>/ and appends a number to save_dir if it
   already exists. Check where the labels actually landed.
"""
import sys

from ultralytics.data.converter import convert_coco

convert_coco(labels_dir=sys.argv[1], save_dir=sys.argv[2],
             use_segments=False,   # boxes only; the gate doesn't need masks
             cls91to80=False)      # ⚠️ default True remaps ids as if COCO-91 and scrambles any other dataset
