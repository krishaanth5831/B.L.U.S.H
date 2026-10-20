"""The one function that draws the target box on the birdseye frame.

    from harvest.target_box import draw_target_box
    frame = draw_target_box(frame, (x0, y0, x1, y1))     # returns a copy; frame is 640x480 uint8

Training data (the sim, and the real overlay script once it's written) and the gated rollout must
draw the box identically, so they all import this (Throwaway Pilot step 8, "one function").

Style (chosen 2026-10-20; the user left it to Claude):
- Magenta (255, 0, 255). It's the same in RGB and BGR, so a lerobot RGB frame and an OpenCV BGR
  frame get the same box (no channel-order bug possible), and no plant, tomato or ply is magenta.
- 5 px line, no fill. π0.5 resizes 640x480 to 224x168 + padding (resize_with_pad, lerobot
  5aa74557), so the line ends up ~1.75 px there: still a solid stripe. 2 px would be 0.7 px, a smear.
- Drawn OUTSIDE the fruit with a 3 px gap, so the line never covers the fruit's own pixels (the
  claw gate and the policy still see its colour), and at least 24 px inside, so a fruit a few
  pixels wide still gets an obvious box.
⚠️ Changing any of this means re-drawing every training episode (sim + real) and retraining.
"""
import numpy as np

COLOUR = (255, 0, 255)
THICKNESS = 5
GAP = 3
MIN_INNER = 24


def draw_target_box(frame, box):
    import cv2
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw = max((x1 - x0) / 2 + GAP, MIN_INNER / 2) + THICKNESS / 2
    hh = max((y1 - y0) / 2 + GAP, MIN_INNER / 2) + THICKNESS / 2
    out = np.ascontiguousarray(frame).copy()
    cv2.rectangle(out, (int(round(cx - hw)), int(round(cy - hh))), (int(round(cx + hw)), int(round(cy + hh))),
                  COLOUR, THICKNESS, lineType=cv2.LINE_8)
    return out
