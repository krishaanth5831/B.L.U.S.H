#!/usr/bin/env python3
"""The recall score: how often the ripeness YOLO picks a ripe fruit on the sim's birdseye start frame.

  python eval_yolo_recall.py                  100 layouts, a table per fruit size, writes nothing
  python eval_yolo_recall.py --show           + a window per layout: the recorded YOLO view + reach mask
  python eval_yolo_recall.py --n 200 --t 0.4  more layouts, another selection threshold

For each layout it renders the start frame (arm in scene.START_NORM, tucked out of the way),
runs the YOLO through the same rule the plugin uses (sim/target.py) and scores it against the
sim's true boxes of the 2-3 ripe fruits:
  picked right     the rule selects one of the ripe fruits (the plugin then draws YOLO's box)
  picked wrong     it selects a fruit that isn't ripe (the plugin falls back to a true box)
  missed           it selects nothing (true box again)
  ripe any         some `ripe` box, any confidence, lies on a ripe fruit (is the threshold the problem?)
  any box          some box of any class lies on a ripe fruit (does YOLO see the fruit at all?)
"picked right" is the photorealism score of a render tier. On today's flat-colour tier it is
expected to be low: the YOLO was trained on photos. Run in the lerobot env (the YOLO runs in the
ripeness env through yolo_worker.py).
"""
import argparse
import os

import numpy as np

SIZE_BUCKETS = ((0, 10), (10, 20), (20, 40), (40, 999))   # width of the most visible ripe fruit, px at 640x480



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="layouts (seeds 0..n-1)")
    ap.add_argument("--t", type=float, default=None, help="t_ripe for the selection (default target.T_RIPE_BIRDSEYE)")
    ap.add_argument("--show", action="store_true", help="window per layout; any key = next, q = stop showing")
    a = ap.parse_args()
    if not a.show:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    import scene
    import target as tg
    from cameras import SimCameras
    from harvest import yolo_overlay as ov
    from lens import LENSES
    from ripeness.gate.birdseye import RIPE
    from yolo_client import YoloClient

    t_ripe = tg.T_RIPE_BIRDSEYE if a.t is None else a.t
    yolo = YoloClient()
    names = yolo.ready()
    print(f"weights {yolo.weights.name}, t_ripe {t_ripe} (⚠️ unverified), match: IoU >= {tg.MATCH_IOU} or centre inside")
    print(f"birdseye lens: {LENSES['birdseye'].describe()}")
    rows, show, n_ripe = [], a.show, []
    for seed in range(a.n):
        m, d, info = scene.build(seed)
        if not info["ripe"]:
            continue
        n_ripe.append(len(info["ripe"]))
        q, _ = scene.arm_ids(m)
        mujoco.mj_forward(m, d)                  # scene.build put the arm in scene.START_NORM
        cams = SimCameras(m)
        rgb = cams.rgb(d, "birdseye")
        ids = cams.ids(d, "birdseye")
        mask = tg.reach_mask(cams, d)
        dets = yolo(rgb)
        st = tg.start_frame(cams, d, m, info, dets, t_ripe, ids=ids, mask=mask)
        boxes = tg.fruit_boxes(cams, d, m, info, ids)
        ripe_boxes = [boxes[k][0] for k in info["ripe"] if boxes[k][0] is not None]
        on_ripe = lambda x: any(tg.matches(x["box"], b) for b in ripe_boxes)   # noqa: E731
        width = max((b[2] - b[0] + 1 for b in ripe_boxes), default=0)
        rows.append(dict(seed=seed, width=width, hidden=not ripe_boxes, source=st["source"], reason=st["reason"],
                         ripe_seen=any(int(x["cls"]) == RIPE and on_ripe(x) for x in dets),
                         any_seen=any(on_ripe(x) for x in dets),
                         in_mask=bool(ripe_boxes) and all(tg.select_target(
                             [{"box": b, "cls": RIPE, "conf": 1.0}], 0, mask) is not None for b in ripe_boxes)))
        if show:
            import cv2
            img = ov.draw_overlay(rgb, dets, names, target=st["box"], label_target=st["source"] == "yolo")
            cv2.polylines(img, [np.int32(mask)], True, (255, 255, 255), 1)
            cv2.putText(img, f"seed {seed}: {st['source']} {st['reason']}  {len(info['ripe'])} ripe", (8, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("birdseye start frame: red = YOLO, green = target, white = reach mask (key = next, q = stop)",
                       img[..., ::-1])
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                show = False
        cams.close()
    yolo.close()

    def line(name, rs):
        n = len(rs)
        if not n:
            return f"{name:>12s} {0:4d}"
        c = lambda f: sum(f(r) for r in rs)   # noqa: E731
        return (f"{name:>12s} {n:4d}  {c(lambda r: r['source'] == 'yolo') / n:6.0%}  "
                f"{c(lambda r: r['reason'] == 'yolo_wrong') / n:6.0%}  {c(lambda r: r['reason'] == 'yolo_missed') / n:6.0%}  "
                f"{c(lambda r: r['ripe_seen']) / n:8.0%}  {c(lambda r: r['any_seen']) / n:7.0%}  "
                f"{c(lambda r: r['in_mask']) / n:7.0%}  {c(lambda r: r['hidden']) / n:6.0%}")

    print(f"\n{'ripe px':>12s} {'n':>4s}  {'right':>6s}  {'wrong':>6s}  {'missed':>6s}  {'ripe any':>8s}  "
          f"{'any box':>7s}  {'in mask':>7s}  {'hidden':>6s}")
    for lo, hi in SIZE_BUCKETS:
        print(line(f"{lo}-{hi}", [r for r in rows if lo <= r["width"] < hi]))
    print(line("all", rows))
    w = np.array([r["width"] for r in rows if r["width"]])
    print(f"\nripe fruit per layout: {dict(zip(*np.unique(n_ripe, return_counts=True)))}; "
          f"most visible ripe fruit: median {np.median(w):.0f} px wide, range {w.min()}-{w.max()}")
    print(f"recall score (picked a ripe fruit): {np.mean([r['source'] == 'yolo' for r in rows]):.0%} "
          f"of {len(rows)} layouts. The plugin fills every miss with a true box, so no episode is lost.")


if __name__ == "__main__":
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    main()
