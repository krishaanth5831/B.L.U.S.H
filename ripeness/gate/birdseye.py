"""birdseye.py: the birdseye selection rule. Detections in, one target box (or None) out.

    from ripeness.gate.birdseye import select_target
    pick = select_target(dets, t_ripe=0.25, reach_mask=polygon)   # None = nothing to pick, skip

`dets` is a list of {"box": (x0, y0, x1, y1), "cls": int, "conf": float}, i.e. YOLO output at a
low conf. No ultralytics import here, so the sim (lerobot env) and the real gate share this rule.

Rule (BLUSH Roadmap M2.5, step 7): `ripe` >= t_ripe, box centre inside the reach mask, most
confident; on a tie, the least overlapped by other boxes (least hidden).
⚠️ t_ripe and the reach mask come from a measured birdseye test session (M2.5). Until then the
callers pass guesses and must say so. They never go in gate.yaml, which only holds measured values.
"""
RIPE = 2   # class id, LABELLING.md


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(pt, poly):
    """Point in polygon (ray casting). poly: [(x, y), ...]."""
    x, y = pt
    hit = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def select_target(dets, t_ripe, reach_mask=None):
    cands = []
    for k, det in enumerate(dets):
        if int(det["cls"]) != RIPE or det["conf"] < t_ripe:
            continue
        x0, y0, x1, y1 = det["box"]
        if reach_mask is not None and not inside(((x0 + x1) / 2, (y0 + y1) / 2), reach_mask):
            continue
        overlap = max((iou(det["box"], o["box"]) for j, o in enumerate(dets) if j != k), default=0.0)
        cands.append((round(det["conf"], 3), -overlap, k))
    if not cands:
        return None
    return dets[max(cands)[2]]
