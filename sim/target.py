"""The start-frame target in the sim: YOLO selects one of the ripe fruits, the sim's truth checks it.

Same rule as the real rig (BLUSH Sim Teleop Plan, "same overlay rule as real"): on the first
birdseye frame of a scene, the birdseye gate picks one ripe box, and that box stays fixed for the
whole episode. Each scene has 2-3 ripe fruits (scene.N_RIPE), so the box is what says which one
to pick. The sim knows every fruit's true box, so:
- YOLO's pick lands on a ripe fruit -> that fruit is the target, YOLO's box is drawn   (source "yolo")
- YOLO picks nothing, or a fruit that isn't ripe -> the most visible ripe fruit is the target and
  its true box is drawn (source "true_box", reason "yolo_missed" / "yolo_wrong"), so no episode is lost
- every ripe fruit is hidden at the start frame -> box around one's outline     (source "true_box_hidden")
How often the first case happens is the recall score (eval_yolo_recall.py): per render tier, it
says how real the sim looks to the gate.
"""
import pathlib
import sys

import numpy as np

import scene

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ripeness.gate.birdseye import iou, select_target  # noqa: E402

# ⚠️ unverified: the real t_ripe_birdseye comes from a measured birdseye test session (Roadmap M2.5).
# 0.25 = live_view.py's default draw threshold, leaning to recall as M2.5 asks.
T_RIPE_BIRDSEYE = 0.25
# A YOLO box counts as a fruit if it overlaps the fruit's true box this much, or its centre is inside it.
# Loose on purpose: a 20 px fruit's YOLO box is often a few px off the pixel-tight true box.
MATCH_IOU = 0.3
# Reach mask: the image area where a reachable fruit can hang. The real one gets drawn by hand on
# the locked mast's image (M2.5); here it's projected from the sim's own reach rule (scene.REACH_M).
FRUIT_DEPTH_M = 0.10             # fruit hangs up to this far from the stem towards the aisle (vines_xml)


def matches(det_box, true_box):
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    inside = true_box[0] - 1 <= cx <= true_box[2] + 1 and true_box[1] - 1 <= cy <= true_box[3] + 1
    return inside or iou(det_box, true_box) >= MATCH_IOU


def reach_mask(cams, d):
    """Convex polygon (birdseye pixels) of the vine area within scene.REACH_M of the shoulder."""
    import cv2
    sh = scene.shoulder_world()
    y_stem = scene.TARGET_ROW - np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]) * scene.KEYHOLE_OFF
    toward = -np.sign(scene.TARGET_ROW - scene.ARM_BASE[1])
    g = np.linspace(0, 1, 41)
    pts = np.array([(x, y, z) for x in scene.HOUSE_L * g for z in scene.CROP_Z * g
                    for y in (y_stem, y_stem + toward * FRUIT_DEPTH_M)])
    pts = pts[np.linalg.norm(pts - sh, axis=1) < scene.REACH_M]
    uv = cams.project(d, "birdseye", pts)
    uv = uv[~np.isnan(uv).any(1)].astype(np.float32)
    hull = cv2.convexHull(uv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in hull]


def fruit_boxes(cams, d, m, info, ids=None):
    """Every fruit's visible box in the birdseye frame: [(box or None, pixel count), ...] in fruit order."""
    ids = cams.ids(d, "birdseye") if ids is None else ids
    return [cams.true_box(d, "birdseye", m.geom(f["geom"]).id, ids) for f in info["fruit"]]


def matched_fruit(det_box, boxes):
    """Index of the fruit a detection sits on (best overlap), or None."""
    hits = [(iou(det_box, b), k) for k, (b, _) in enumerate(boxes) if b is not None and matches(det_box, b)]
    return max(hits)[1] if hits else None


def start_frame(cams, d, m, info, dets, t_ripe=T_RIPE_BIRDSEYE, ids=None, mask=None):
    """Pick the scene's target and its box, plus what happened (logged per scene for the recall score)."""
    mask = reach_mask(cams, d) if mask is None else mask
    boxes = fruit_boxes(cams, d, m, info, ids)
    ripe = info["ripe"]
    pick = select_target(dets, t_ripe, mask)
    hit = None if pick is None else matched_fruit(pick["box"], boxes)
    name = lambda k: None if k is None else info["fruit"][k]["tomato"]   # noqa: E731
    out = dict(ripe=[name(k) for k in ripe], n_dets=len(dets), t_ripe=t_ripe,
               yolo_box=None if pick is None else [round(v, 1) for v in pick["box"]],
               yolo_conf=None if pick is None else round(pick["conf"], 3), yolo_fruit=name(hit))
    if hit is not None and hit in ripe:
        k, source, reason, box = hit, "yolo", "", out["yolo_box"]
    else:
        reason = "yolo_missed" if pick is None else "yolo_wrong"
        seen = [k for k in ripe if boxes[k][0] is not None]
        if seen:
            k = max(seen, key=lambda j: boxes[j][1])          # the most visible ripe fruit
            source, box = "true_box", list(boxes[k][0])
        elif ripe:
            k = ripe[0]
            gid = m.geom(info["fruit"][k]["geom"]).id
            sb = cams.sphere_box(d, "birdseye", d.geom_xpos[gid], m.geom_size[gid][0])
            source, box = ("true_box_hidden", list(sb)) if sb is not None else ("none", None)
            if sb is None:
                reason = "target_out_of_view"
        else:
            k, source, box, reason = None, "none", None, "no_ripe_fruit"
    tb = None if k is None or boxes[k][0] is None else [int(v) for v in boxes[k][0]]
    out.update(target=k, target_name=name(k), source=source, reason=reason, box=box, true_box=tb,
               visible_px=0 if k is None else int(boxes[k][1]),
               yolo_iou=None if pick is None or tb is None else round(iou(pick["box"], tb), 3))
    return out
