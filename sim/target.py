"""Box matching + the reach mask for the start-frame target (truss_target.py picks the target truss with them).

    matches(det_box, true_box)     a YOLO box counts as on a true box (IoU >= MATCH_IOU or centre inside)
    reach_mask(cams, d)            birdseye polygon of the vine area within scene.REACH_M of the shoulder

Same rule as the real rig (BLUSH Sim Teleop Plan, "same overlay rule as real"): the birdseye's first frame picks
the target, inside the reach mask. Importing this module puts the repo root on sys.path (ripeness/ here, harvest/
in the plugin and view_scene.py rely on that). The single-fruit target picking (start_frame) was removed 2026-11-08.
"""
import pathlib
import sys

import numpy as np

import scene

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ripeness.gate.birdseye import iou  # noqa: E402  (truss_target imports it from here too)

# A YOLO box counts as a fruit if it overlaps the fruit's true box this much, or its centre is inside it.
# Loose on purpose: a 20 px fruit's YOLO box is often a few px off the pixel-tight true box.
MATCH_IOU = 0.3
# Reach mask: the image area where a reachable fruit can hang. The real one gets drawn by hand on
# the locked mast's image (M2.5); here it's projected from the sim's own reach rule (scene.REACH_M).
FRUIT_DEPTH_M = 0.10             # fruit hangs up to this far from the stem toward the aisle (guess)


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
