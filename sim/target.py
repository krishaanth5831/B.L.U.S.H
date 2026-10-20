"""The start-frame target box in the sim: YOLO selects, the sim's true box checks it and fills in.

Same rule as the real rig (BLUSH Sim Teleop Plan, "same overlay rule as real"): on the first
birdseye frame of a scene, the birdseye gate picks one ripe box, and that box stays fixed for the
whole episode. The sim knows where the ripe fruit really is, so:
- YOLO's pick matches the true target -> use YOLO's box               (source "yolo")
- YOLO picks nothing, or a different fruit -> use the true box instead   (source "true_box", reason
  "yolo_missed" / "yolo_wrong"), so no episode is lost
- the target is hidden behind something -> box around its outline       (source "true_box_hidden")
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
# A YOLO box counts as the target if it overlaps the true box this much, or its centre is inside it.
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


def true_target(cams, d, m, info, ids=None):
    """The target's box in the birdseye frame: visible pixels, else its outline if hidden."""
    if info["target"] is None:
        return None, 0, False
    t = info["fruit"][info["target"]]
    gid = m.geom(t["geom"]).id
    box, px = cams.true_box(d, "birdseye", gid, ids)
    if box is not None:
        return box, px, True
    return cams.sphere_box(d, "birdseye", d.geom_xpos[gid], m.geom_size[gid][0]), 0, False


def choose(dets, true_box, visible, mask, t_ripe=T_RIPE_BIRDSEYE):
    """The box to draw for this scene, plus what happened (logged per scene for the recall score)."""
    pick = select_target(dets, t_ripe, mask)
    out = dict(true_box=None if true_box is None else [int(v) for v in true_box], visible=bool(visible),
               yolo_box=None if pick is None else [round(v, 1) for v in pick["box"]],
               yolo_conf=None if pick is None else round(pick["conf"], 3), t_ripe=t_ripe,
               n_dets=len(dets),
               yolo_iou=None if pick is None or true_box is None else round(iou(pick["box"], true_box), 3))
    if pick is not None and true_box is not None and matches(pick["box"], true_box):
        out.update(source="yolo", box=out["yolo_box"], reason="")
    elif true_box is None:
        # The target is out of view: draw nothing rather than a wrong fruit.
        out.update(source="none", box=None, reason="target_out_of_view")
    else:
        out.update(source="true_box" if visible else "true_box_hidden", box=out["true_box"],
                   reason="yolo_missed" if pick is None else "yolo_wrong")
    return out
