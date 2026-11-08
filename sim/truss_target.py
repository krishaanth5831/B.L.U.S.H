"""The start-frame target for whole-truss harvesting: which ready truss, its box and its cut point (K7, 2026-11-03).

    boxes = truss_boxes(cams, d, m, info)             per truss: visible box (birdseye px) or None, pixel count
    labels = yolo_labels(cams, d, m, info)            YOLO training rows: truss_ready / truss_not_ready / cut_point
    out = start_frame(cams, d, m, info, dets)         the target truss, its box and cut point, and why

Same rule as target.py for single fruit, moved to trusses: on the first birdseye frame the truss YOLO's most
confident `truss_ready` box inside the reach mask is the target if it sits on a ready truss (source "yolo");
otherwise the most visible ready truss's true box is drawn (source "true_box", reason yolo_missed / yolo_wrong),
so no episode is lost. The cut point is the truss YOLO's `cut_point` box inside the target box (source "yolo"),
else the true cut point projected into the image. Box and cut point are held for the whole episode and go to
the policy as numbers (observation.environment_state, ENV_NAMES). Rig "base": the target is still picked on the
birdseye, but the numbers (and the drawn box) are where the BASE camera sees that truss (view_target, sim truth).

The true boxes come from the segmentation render: a truss's box is the union of its fruit's and its stalk's
visible pixels, the cut point box a square around the projected middle of its cut zone.
"""
import numpy as np

import truss_scene as ts
from target import iou, matches, reach_mask

CLASSES = {0: "truss_ready", 1: "truss_not_ready", 2: "cut_point"}
READY, NOT_READY, CUT = 0, 1, 2
# Cut-point box: a square this big (px) around the projected cut point. The stalk is 3-5 mm, ~3-5 px in the
# birdseye at 0.6 m; the square gives YOLO something to find (design choice).
CUT_BOX_PX = 14
# A truss counts as seen when at least this many pixels show (fruit + stalk); fewer and it gets no label.
MIN_PX = 40
T_READY_BIRDSEYE = 0.25          # ⚠️ unverified, like target.T_RIPE_BIRDSEYE: set from a measured test split
ENV_NAMES = ["box_cx", "box_cy", "box_w", "box_h", "cut_u", "cut_v"]
W, H = 640, 480                  # every sim camera's frame (lens.W, H): birdseye, and base for rig "base" (view_target)


def _union(boxes):
    b = [x for x in boxes if x is not None]
    if not b:
        return None
    b = np.array(b)
    return (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))


def truss_boxes(cams, d, m, info, ids=None, cam="birdseye"):
    ids = cams.ids(d, cam) if ids is None else ids
    out = []
    for t in info["trusses"]:
        parts = [cams.true_box(d, cam, m.geom(f["geom"]).id, ids) for f in t["fruit"]]
        parts += [cams.true_box(d, cam, m.geom(g).id, ids) for g in t["geoms"]]
        px = sum(n for _, n in parts)
        out.append((_union([b for b, _ in parts]) if px >= MIN_PX else None, px))
    return out


def outline_box(cams, d, m, t, cam="birdseye"):
    """Box around a truss's fruit outlines and stalk whether or not something hides it (cf. target.py)."""
    bs = [cams.sphere_box(d, cam, d.xpos[m.body(f["tomato"]).id], f["radius"]) for f in t["fruit"]]
    b = m.body(t["body"]).id
    pts = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ (q - t["knuckle"])
           for q in (t["knuckle"], t["peduncle_end"], t["rachis_end"])]
    uv = cams.project(d, cam, np.array(pts))
    bs += [(u, v, u, v) for u, v in uv if not np.isnan(u)]
    u = _union(bs)
    return None if u is None else [max(0, u[0]), max(0, u[1]), min(W - 1, u[2]), min(H - 1, u[3])]


def cut_pixel(cams, d, m, t, cam="birdseye"):
    """The middle of the truss's cut zone (world, where it hangs now) in this camera's pixels, or None."""
    p, _ = ts.cut_target_now(t, d, m)
    uv = cams.project(d, cam, p)[0]
    return None if np.isnan(uv).any() else (float(uv[0]), float(uv[1]))


def yolo_labels(cams, d, m, info, ids=None):
    """[(cls, (x0, y0, x1, y1)), ...] for one birdseye frame."""
    rows = []
    for t, (box, _) in zip(info["trusses"], truss_boxes(cams, d, m, info, ids)):
        if box is None:
            continue
        rows.append((READY if t["ready"] else NOT_READY, box))
        c = cut_pixel(cams, d, m, t)
        if c is not None and box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]:
            h = CUT_BOX_PX / 2
            rows.append((CUT, (max(0, c[0] - h), max(0, c[1] - h), min(W - 1, c[0] + h), min(H - 1, c[1] + h))))
    return rows


def view_target(cams, d, m, t, cam):
    """The chosen target truss as another camera sees it (rig "base": the target is picked on the birdseye, the
    policy's camera gets its box + cut point from the sim's truth): (visible box or the outline box when hidden,
    cut pixel or None). Both None when the truss is out of that camera's view."""
    ids = cams.ids(d, cam)
    parts = [cams.true_box(d, cam, m.geom(f["geom"]).id, ids) for f in t["fruit"]]
    parts += [cams.true_box(d, cam, m.geom(g).id, ids) for g in t["geoms"]]
    box = _union([b for b, _ in parts]) if sum(n for _, n in parts) >= MIN_PX else None
    if box is None:
        box = outline_box(cams, d, m, t, cam)
    return (None if box is None else [int(v) for v in box]), cut_pixel(cams, d, m, t, cam)


def env_vector(box, cut):
    """[cx, cy, w, h, u, v] / image size; zeros where missing."""
    v = np.zeros(6, np.float32)
    if box is not None:
        x0, y0, x1, y1 = box
        v[:4] = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
    if cut is not None:
        v[4:] = [cut[0] / W, cut[1] / H]
    return v


def start_frame(cams, d, m, info, dets, t_ready=T_READY_BIRDSEYE, ids=None, mask=None):
    mask = reach_mask(cams, d) if mask is None else mask
    boxes = truss_boxes(cams, d, m, info, ids)
    ready = info["ready"]
    from ripeness.gate.birdseye import inside
    cands = [x for x in dets if int(x["cls"]) == READY and x["conf"] >= t_ready
             and inside(((x["box"][0] + x["box"][2]) / 2, (x["box"][1] + x["box"][3]) / 2), mask)]
    pick = max(cands, key=lambda x: x["conf"]) if cands else None
    hit = None
    if pick is not None:
        hits = [(iou(pick["box"], b), k) for k, (b, _) in enumerate(boxes) if b is not None and matches(pick["box"], b)]
        hit = max(hits)[1] if hits else None
    name = lambda k: None if k is None else info["trusses"][k]["name"]   # noqa: E731
    out = dict(ready=[name(k) for k in ready], n_dets=len(dets), t_ready=t_ready,
               yolo_box=None if pick is None else [round(v, 1) for v in pick["box"]],
               yolo_conf=None if pick is None else round(pick["conf"], 3), yolo_truss=name(hit))
    if hit is not None and hit in ready:
        k, source, reason, box = hit, "yolo", "", out["yolo_box"]
    else:
        reason = "yolo_missed" if pick is None else "yolo_wrong"
        seen = [k for k in ready if boxes[k][0] is not None]
        if seen:
            k = max(seen, key=lambda j: boxes[j][1])
            source, box = "true_box", list(boxes[k][0])
        elif ready:
            k = ready[0]                                  # hidden (behind the arm): box around its outline
            box = outline_box(cams, d, m, info["trusses"][k])
            source = "true_box_hidden" if box is not None else "none"
            if box is None:
                k, reason = None, "target_out_of_view"
        else:
            k, source, box, reason = None, "none", None, "no_ready_truss"
    cut, cut_source = None, "none"
    if k is not None:
        cuts = [x for x in dets if int(x["cls"]) == CUT and box[0] <= (x["box"][0] + x["box"][2]) / 2 <= box[2]
                and box[1] <= (x["box"][1] + x["box"][3]) / 2 <= box[3]]
        if source == "yolo" and cuts:
            c = max(cuts, key=lambda x: x["conf"])["box"]
            cut, cut_source = ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2), "yolo"
        else:
            cut = cut_pixel(cams, d, m, info["trusses"][k])
            cut_source = "true" if cut is not None else "none"
    tb = None if k is None or boxes[k][0] is None else [int(v) for v in boxes[k][0]]
    out.update(target=k, target_name=name(k), source=source, reason=reason, box=box, true_box=tb,
               yolo_iou=None if (pick is None or tb is None) else round(iou(pick["box"], tb), 2),
               cut=None if cut is None else [round(c, 1) for c in cut], cut_source=cut_source, mask=mask)
    return out
