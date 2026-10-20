#!/usr/bin/env python3
"""The recall score: how often the ripeness YOLO picks the right fruit on the sim's birdseye frame.

  python eval_yolo_recall.py                  60 layouts, a table per target size, writes nothing
  python eval_yolo_recall.py --show           + a window per layout: YOLO boxes, true box, reach mask
  python eval_yolo_recall.py --n 200 --t 0.4  more layouts, another selection threshold

For each layout it renders the start frame (arm folded out of the way, like the real rest pose),
runs the YOLO through the same rule the plugin uses (sim/target.py) and scores it against the
sim's true target box:
  picked right     the rule selects the ripe target (the plugin then uses YOLO's box)
  picked wrong     it selects another fruit (the plugin falls back to the true box)
  missed           it selects nothing (true box again)
  ripe seen        some `ripe` box, any confidence, lies on the target (is the threshold the problem?)
"picked right" is the photorealism score of a render tier. On today's flat-colour tier it is
expected to be low: the YOLO was trained on photos. Run in the lerobot env (the YOLO runs in the
ripeness env through yolo_worker.py).
"""
import argparse
import os

import numpy as np

SIZE_BUCKETS = ((0, 10), (10, 20), (20, 40), (40, 999))   # target width in px at 640x480
CLASS_COLOURS = {0: (0, 200, 0), 1: (255, 165, 0), 2: (255, 0, 0), 3: (160, 160, 160)}   # RGB, like live_view.py
REST_NORM = (0, -100, 100, 60, 0, 0)   # folded arm (normalised units): out of the birdseye's way


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="layouts (seeds 0..n-1)")
    ap.add_argument("--t", type=float, default=None, help="t_ripe for the selection (default target.T_RIPE_BIRDSEYE)")
    ap.add_argument("--show", action="store_true", help="window per layout; any key = next, q = stop showing")
    a = ap.parse_args()
    if not a.show:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    import arm_map
    import scene
    import target as tg
    from cameras import SimCameras
    from harvest.target_box import draw_target_box
    from lens import LENSES
    from ripeness.gate.birdseye import RIPE
    from yolo_client import YoloClient

    t_ripe = tg.T_RIPE_BIRDSEYE if a.t is None else a.t
    yolo = YoloClient()
    print(f"weights {yolo.weights.name}, t_ripe {t_ripe} (⚠️ unverified), match: IoU >= {tg.MATCH_IOU} or centre inside")
    print(f"birdseye lens: {LENSES['birdseye'].describe()}")
    fm = arm_map.FollowerMap()
    rows, show = [], a.show
    for seed in range(a.n):
        m, d, info = scene.build(seed)
        if info["target"] is None:
            continue
        q, _ = scene.arm_ids(m)
        d.qpos[q] = fm.norm_to_rad(REST_NORM)
        mujoco.mj_forward(m, d)
        cams = SimCameras(m)
        rgb = cams.rgb(d, "birdseye")
        true_box, px, visible = tg.true_target(cams, d, m, info)
        mask = tg.reach_mask(cams, d)
        dets = yolo(rgb)
        ch = tg.choose(dets, true_box, visible, mask, t_ripe)
        ripe_seen = true_box is not None and any(int(x["cls"]) == RIPE and tg.matches(x["box"], true_box) for x in dets)
        any_seen = true_box is not None and any(tg.matches(x["box"], true_box) for x in dets)
        width = 0 if true_box is None else true_box[2] - true_box[0] + 1
        rows.append(dict(seed=seed, width=width, visible=visible, source=ch["source"], reason=ch["reason"],
                         ripe_seen=ripe_seen, any_seen=any_seen, in_mask=true_box is not None and tg.select_target(
                             [{"box": true_box, "cls": RIPE, "conf": 1.0}], 0, mask) is not None))
        if show:
            import cv2
            img = rgb.copy()
            cv2.polylines(img, [np.int32(mask)], True, (255, 255, 255), 1)
            for x in dets:
                b = [int(v) for v in x["box"]]
                cv2.rectangle(img, b[:2], b[2:], CLASS_COLOURS.get(int(x["cls"]), (255, 255, 255)), 1)
                cv2.putText(img, f"{yolo.names[int(x['cls'])][:4]} {x['conf']:.2f}", (b[0], max(b[1] - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLOURS.get(int(x["cls"]), (255, 255, 255)), 1)
            drawn = draw_target_box(img, ch["box"]) if ch["box"] is not None else img
            cv2.putText(drawn, f"seed {seed}: {ch['source']} {ch['reason']}  target {width}px", (8, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("birdseye: YOLO boxes (thin) | drawn target box (magenta) | reach mask (white)", drawn[..., ::-1])
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                show = False
        cams.close()
    yolo.close()

    def line(name, rs):
        n = len(rs)
        if not n:
            return f"{name:>12s} {0:4d}"
        c = lambda f: sum(f(r) for r in rs)
        right = c(lambda r: r["source"] == "yolo")
        wrong = c(lambda r: r["reason"] == "yolo_wrong")
        miss = c(lambda r: r["reason"] == "yolo_missed")
        return (f"{name:>12s} {n:4d}  {right / n:6.0%}  {wrong / n:6.0%}  {miss / n:6.0%}  "
                f"{c(lambda r: r['ripe_seen']) / n:8.0%}  {c(lambda r: r['any_seen']) / n:6.0%}  "
                f"{c(lambda r: r['in_mask']) / n:6.0%}  {c(lambda r: not r['visible']) / n:6.0%}")

    print(f"\n{'target px':>12s} {'n':>4s}  {'right':>6s}  {'wrong':>6s}  {'missed':>6s}  {'ripe any':>8s}  "
          f"{'any box':>6s}  {'in mask':>6s}  {'hidden':>6s}")
    for lo, hi in SIZE_BUCKETS:
        print(line(f"{lo}-{hi}", [r for r in rows if lo <= r["width"] < hi]))
    print(line("all", rows))
    w = np.array([r["width"] for r in rows if r["width"]])
    print(f"\ntarget width px: median {np.median(w):.0f}, range {w.min()}-{w.max()}")
    print(f"recall score (picked right): {np.mean([r['source'] == 'yolo' for r in rows]):.0%} "
          f"of {len(rows)} layouts. The plugin fills every miss with the true box, so no episode is lost.")


if __name__ == "__main__":
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    main()
