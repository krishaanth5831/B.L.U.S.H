"""The YOLO view drawn into the camera frames: the one module that draws it, sim and real.

    from harvest.yolo_overlay import draw_overlay
    frame = draw_overlay(frame, dets, names)                                # claw: every detection
    frame = draw_overlay(frame, dets, names, target=(x0, y0, x1, y1))       # birdseye: + the target

`dets` = [{"box": (x0, y0, x1, y1), "cls": int, "conf": float}, ...], raw YOLO output (any conf);
`names` = the model's class names. Returns a copy; frames are 640x480 uint8.

What gets drawn (K7, 2026-10-20: the recorded frames carry the full YOLO view):
- every detection with conf >= DISPLAY_CONF: a red box + "<class> <conf>%", e.g. "ripe 82%", on
  both cameras, every frame
- birdseye only: the target, a green box. It's picked on the scene's first frame (highest `ripe`
  confidence in reach, ripeness/gate/birdseye.py) and then held for the whole episode, so it never
  jumps between ripe fruits. No red box is drawn under it; it carries the live label of the
  detection on it, if any.
⚠️ Sim recordings, real recordings and the gated rollout must all draw with this module, the same
weights and the same thresholds, or the policy trains on different pictures than it sees on the arm.
⚠️ Frames are RGB, like lerobot's camera and dataset frames. Red (255, 0, 0) drawn into a BGR OpenCV
frame comes out blue: convert BGR frames to RGB first.
⚠️ π0.5 sees these frames at 224x168 (resize_with_pad from 640x480, lerobot 5aa74557). The boxes
survive that; the label text (~3 px tall there) won't be readable to the policy.
"""
import numpy as np

DET_COLOUR = (255, 0, 0)        # RGB red: every detection
DET_THICKNESS = 2
TARGET_COLOUR = (0, 255, 0)     # green: the target
TARGET_THICKNESS = 5            # ~1.75 px at π0.5's 224 px width: still a solid stripe. 2 px would be a smear
GAP = 3                         # the target box sits this far outside the fruit, so its line never covers the fruit,
MIN_INNER = 24                  # and is at least this wide inside, so a fruit a few pixels wide still gets an obvious box
DISPLAY_CONF = 0.25             # lowest confidence drawn (ripeness/scripts/live_view.py's default)
DISPLAY_NMS_IOU = 0.7           # class-agnostic NMS for drawing only, one verdict per object (ultralytics' default IoU)
ON_TARGET_IOU = 0.3             # a detection this much on the target (or centred inside it) is the target
# ⚠️ Boxes covering >= 90 % of the frame both ways are skipped: the sim claw camera looking at the bright
# table got a 620x480 "not_tomato 71%" box (2026-10-20). A cherry tomato never fills that much of
# either camera's frame (the claw camera sits behind the fingers).
FILL_FRAC = 0.9
FONT, FONT_SCALE = 0, 0.4       # cv2.FONT_HERSHEY_SIMPLEX


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _on(det_box, box):
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3] or _iou(det_box, box) >= ON_TARGET_IOU


def _fills_frame(box, shape):
    h, w = shape[:2]
    return box[2] - box[0] >= FILL_FRAC * w and box[3] - box[1] >= FILL_FRAC * h


def display_dets(dets, shape):
    """The detections that get drawn: conf >= DISPLAY_CONF, not frame-filling, then greedy
    class-agnostic NMS."""
    keep = []
    for det in sorted((x for x in dets if x["conf"] >= DISPLAY_CONF and not _fills_frame(x["box"], shape)),
                      key=lambda x: -x["conf"]):
        if all(_iou(det["box"], k["box"]) < DISPLAY_NMS_IOU for k in keep):
            keep.append(det)
    return keep


def label(det, names):
    return f"{names[int(det['cls'])]} {round(100 * det['conf'])}%"


def _text(out, s, x, y, colour):
    import cv2
    (tw, th), _ = cv2.getTextSize(s, FONT, FONT_SCALE, 1)
    x = int(min(max(x, 0), out.shape[1] - tw - 1))
    y = int(y - 3 if y - 3 - th >= 0 else y + th + 3)        # above the box, or inside it at the top edge
    cv2.putText(out, s, (x, y), FONT, FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)   # dark outline for contrast
    cv2.putText(out, s, (x, y), FONT, FONT_SCALE, colour, 1, cv2.LINE_AA)


def target_rect(box):
    """The drawn target rectangle (outside the fruit, MIN_INNER wide at least) as two corners."""
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw = max((x1 - x0) / 2 + GAP, MIN_INNER / 2) + TARGET_THICKNESS / 2
    hh = max((y1 - y0) / 2 + GAP, MIN_INNER / 2) + TARGET_THICKNESS / 2
    return (int(round(cx - hw)), int(round(cy - hh))), (int(round(cx + hw)), int(round(cy + hh)))


def draw_overlay(frame, dets, names, target=None, label_target=True):
    """label_target=False for a target that YOLO didn't pick (the sim's true-box fallback): its box
    then carries no class or confidence, since YOLO never called it ripe."""
    import cv2
    out = np.ascontiguousarray(frame).copy()
    shown = display_dets(dets, out.shape)
    on_target = [x for x in shown if target is not None and _on(x["box"], target)]
    for det in shown:
        if det in on_target:
            continue                  # never a red box under the green one
        x0, y0, x1, y1 = (int(round(v)) for v in det["box"])
        cv2.rectangle(out, (x0, y0), (x1, y1), DET_COLOUR, DET_THICKNESS, lineType=cv2.LINE_8)
        _text(out, label(det, names), x0, y0, DET_COLOUR)
    if target is not None:
        p0, p1 = target_rect(target)
        cv2.rectangle(out, p0, p1, TARGET_COLOUR, TARGET_THICKNESS, lineType=cv2.LINE_8)
        if on_target and label_target:
            _text(out, label(on_target[0], names), p0[0], p0[1] - TARGET_THICKNESS // 2, TARGET_COLOUR)
    return out
