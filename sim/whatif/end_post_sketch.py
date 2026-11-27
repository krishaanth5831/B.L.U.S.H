#!/usr/bin/env python3
"""Seen from above: the hand's recorded paths and the plank rig's end post. READ-ONLY, one picture.

    python whatif/end_post_sketch.py ../hardware/greenhouse/pipe_rail_rig/renders/end_post_check.png     # from sim/

The plank rig's end frames stand 225 mm from each arm, the simulator's 370 mm. `camera_sweep_logs.py arm` counted
how often the recorded motions come near the place of the rig's row post (rig README, "Checked 2026-11-27"); this
draws why. For every logged episode of dataset v6 (the 800 old ones, the 600 new ones) the fixed finger's tip is
followed through every frame: the logged joint readings posed on the arm, as the sweep does it. One cut episode in
--every is drawn, orange when the sweep found the arm within 4 cm of the post's corner line in that episode (its
--episodes-json files), blue otherwise. Over ALL the episodes it prints where the tip is at the first frame, how
far it goes past the end frame's inside face, and how high it gets, also under the row's string rail.

⚠️ Needs the state logs (~/.cache/blush_sim/states) and the sweep's json files in outputs/v6/endpost/ (untracked).
⚠️ A tip is one point of the hand: the fingers and the pad reach a few cm around it. The counts are the sweep's.
"""
import argparse
import json
import os
import sys
import types

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import camera_sweep_logs as csl  # noqa: E402
import rebuild_v5_check as rc  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"      # as camera_post_sketch.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"                        # reference palette slots 1-3
T = 0.044                # the rig's posts are 44 x 44 mm battens (pipe_rail_rig.fs `T`)
FRAME_M = 0.225          # plank rig as drawn: the end frame's inside face to the arm (rig README, cart centred)
LONGER_M = 0.375         # the same in a house 300 mm longer ("Inside length" 1200 mm, rig README way 2)
ROW_PITCH = 0.40         # rows A and B are 400 mm apart and the arm stands halfway (pipe_rail_rig.fs rowPitch)
CORNER = "0.145:0.378:0.52"     # the sweep's line on the row-B post's corner nearest the arm. At 4 cm the nine
#                                 row-B lines together list the same episodes as this one (380 old, 264 new)
NEAR_M = 0.04            # the sweep's "under 4 cm" (camera_score.CLEAR_MIN_M)
RAIL_Z = 0.52            # underside of the rig's string rails above the arm base (pipe_rail_rig.fs cropAbove 520)
RAIL_SIDE_M = 0.03       # "under the rail": this far to either side of the 44 mm rail (guess: about a hand's width)
EPS = os.path.join(os.path.dirname(HERE), "..", "outputs", "v6", "endpost", "{}arm_eps.json")


def tip_paths(m, d, qidx, fm, tip, scenes):
    """Per scene: (seed, nothing-ripe?, the tip's place (N, 3) at every logged frame)."""
    import mujoco
    out = []
    for x in scenes:
        st = np.asarray(np.load(x["npz"], allow_pickle=True)["state"], float)
        p = np.empty((len(st), 3))
        for k, row in enumerate(st):
            d.qpos[qidx] = fm.norm_to_rad(row)
            mujoco.mj_kinematics(m, d)
            p[k] = d.xpos[tip]
        out.append((x["seed"], x["hold"], p))
    return out


def box_xy(m, d, name):
    """(x0, y0, width, depth) of a box geom, seen from above."""
    g = m.geom(name).id
    c, h = d.geom_xpos[g], m.geom_size[g]
    return c[0] - h[0], c[1] - h[1], 2 * h[0], 2 * h[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("png")
    ap.add_argument("--every", type=int, default=8, help="draw one cut episode in this many (default 8)")
    a = ap.parse_args()
    import mujoco

    import arm_map
    import scene
    import truss_scene as ts
    ts.set_world("crate23")
    old = rc.v5_scenes()
    new = csl.logged(types.SimpleNamespace(sessions="v6new"))       # ⚠️ changes rc.V5_SESSIONS: list the old first
    m, d, _ = ts.build(old[1]["seed"], base_mount=True, jaw="v5")    # any scene: only the arm is posed
    mujoco.mj_forward(m, d)
    qidx, _ = scene.arm_ids(m)
    fm = arm_map.FollowerMap()
    base = np.array(scene.ARM_BASE[:2])
    near = {}
    for tag in ("", "new_"):
        near.update({int(s): v for s, v in json.load(open(EPS.format(tag)))[CORNER].items()})
    P = tip_paths(m, d, qidx, fm, m.body("arm/finray_fixed_cap").id, old + new)

    # ---- numbers over every episode ----
    face = base[0] - FRAME_M
    allp = np.vstack([p for _, _, p in P])
    k = int(allp[:, 0].argmin())
    start = np.array([p[0] for _, _, p in P])
    print(f"{len(P)} logged episodes, {len(allp)} frames; fixed finger's tip (world m):")
    print(f"  at the first frame: x {start[:, 0].min():.3f}-{start[:, 0].max():.3f}, y {start[:, 1].min():.3f}-"
          f"{start[:, 1].max():.3f}, z {start[:, 2].min():.3f}-{start[:, 2].max():.3f}; the plank rig's end frame "
          f"has its inside face at x {face:.3f}")
    print(f"  furthest to the arm's left: x {allp[k, 0]:.3f} at y {allp[k, 1]:.3f}, z {allp[k, 2]:.3f} = "
          f"{1000 * (face - allp[k, 0]):.0f} mm past that face")
    k = int(allp[:, 2].argmax())
    rail = np.abs(allp[:, 1] - (base[1] + ROW_PITCH / 2)) < T / 2 + RAIL_SIDE_M
    print(f"  highest: z {allp[k, 2]:.3f} at x {allp[k, 0]:.3f}, y {allp[k, 1]:.3f}; under the row's string rail "
          f"(within {1000 * RAIL_SIDE_M:.0f} mm of its sides; the rig's rail has its underside at z "
          f"{RAIL_Z:.2f}): z {allp[rail, 2].max():.3f}")
    past = np.array([p[:, 0].min() < face for _, _, p in P])
    hold = np.array([h for _, h, _ in P])
    print(f"  tip past that face at some frame: {past.sum()} of {len(P)} episodes ({(past & ~hold).sum()} of "
          f"{(~hold).sum()} with a cut, {(past & hold).sum()} of {hold.sum()} nothing-ripe)")

    # ---- the picture: mm from the arm base, the arm looking up the page ----
    def mm(x, y):
        return (np.asarray(x) - base[0]) * 1000, (np.asarray(y) - base[1]) * 1000

    fig, ax = plt.subplots(figsize=(12.5, 8.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    box = dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color="#e9e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-790, 420)
    ax.set_ylim(-285, 400)
    ax.set_xlabel("mm from the arm base:   to the arm's left  <      > to its right", color=INK2, fontsize=10)
    ax.set_ylabel("mm toward the row the arm faces", color=INK2, fontsize=10)

    cuts = [(s, p) for s, h, p in P if not h][::a.every]
    n_near = 0
    for want_near in (False, True):                                  # the near ones on top
        for s, p in cuts:
            is_near = near.get(s, 1.0) < NEAR_M
            if is_near != want_near:
                continue
            n_near += is_near
            ax.plot(*mm(p[:, 0], p[:, 1]), color=ORANGE if is_near else BLUE, lw=0.7, alpha=0.5 if is_near else 0.4,
                    zorder=3 if is_near else 2)
    # the row of plants and the crate
    sx = mm(np.array(ts.STEM_X), [0, 0])[0]
    ry = 1000 * (ROW_PITCH / 2)
    ax.plot(sx, [ry - 20, ry - 20], color=AQUA, lw=7, alpha=0.35, solid_capstyle="round", zorder=1)
    ax.text(sx.mean() + 40, ry + 4, "row of plants (the trusses hang here)", color=INK, fontsize=10.5, ha="center",
            bbox=box, zorder=10)
    cx, cy, cw, cd = box_xy(m, d, "crate_floor")
    (cx, cy), cw, cd = mm(cx, cy), cw * 1000, cd * 1000
    ax.add_patch(Rectangle((cx, cy), cw, cd, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=1.2, zorder=1))
    ax.text(cx + cw - 30, cy + cd - 30, "crate", color=INK, fontsize=10.5, ha="right", va="top", bbox=box, zorder=10)
    # the arm and the hand at the start
    ax.add_patch(Circle((0, 0), 45, facecolor="#e9e8e4", edgecolor=INK, lw=1.5, zorder=6))
    ax.text(0, 0, "arm", color=INK, fontsize=10.5, ha="center", va="center", zorder=7)
    hx, hy = mm(np.median(start[:, 0]), np.median(start[:, 1]))
    ax.add_patch(Circle((hx, hy), 9, facecolor=INK, edgecolor=SURFACE, lw=2, zorder=8))
    ax.annotate("the hand's tip at the start: it points to the\narm's left, into the end frame's opening", (hx, hy),
                xytext=(-205, -150), color=INK, fontsize=10.5, ha="left", va="top", bbox=box, zorder=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    # the simulator's end frame (model geoms) and the dataset's camera post
    for g in ("ef0_post0", "ef0_post1"):
        x0, y0, w, h = box_xy(m, d, g)
        (x0, y0) = mm(x0, y0)
        ax.add_patch(Rectangle((x0, y0), w * 1000, h * 1000, facecolor=INK2, edgecolor=INK2, lw=1, zorder=6))
    cam = scene.fixed_cam("high")[0]
    ax.add_patch(Circle(mm(cam[0], cam[1]), 8, facecolor=MUTED, edgecolor=SURFACE, lw=1.5, zorder=6))
    ax.annotate("the dataset's camera post\n(it rides on the cart)", mm(cam[0], cam[1]), xytext=(-775, 40), color=INK2,
                fontsize=10, ha="left", va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    # the rig's row posts: as drawn, and in a house 300 mm longer
    for rowy in (ry, -ry):
        ax.add_patch(Rectangle((-LONGER_M * 1000 - T * 1000, rowy - T * 500), T * 1000, T * 1000, facecolor="none",
                               edgecolor=INK2, lw=1.4, ls=(0, (4, 3)), zorder=5))
        ax.add_patch(Rectangle((-FRAME_M * 1000 - T * 1000, rowy - T * 500), T * 1000, T * 1000, facecolor=INK,
                               edgecolor=SURFACE, lw=2, zorder=9))
    ax.plot([-FRAME_M * 1000] * 2, [-ry + T * 500, ry - T * 500], color=INK, lw=1.2, ls=(0, (2, 4)), zorder=5)
    ax.annotate("END POST AS DRAWN\n225 mm from the arm:\nthe hand's swing passes here",
                (-FRAME_M * 1000 - 22, ry + 22), xytext=(-330, 318), color=INK, fontsize=11, fontweight="bold",
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
    ax.text(-FRAME_M * 1000 - 8, -95, "open between\nthe two posts", color=INK2, fontsize=9.5, ha="right", va="center")
    ax.annotate("in a house 300 mm longer the post\nstands here (375 mm from the arm):\nwhere the simulator has its "
                "end\nframe (370 mm, the grey plates)", (-LONGER_M * 1000 - T * 1000, ry), xytext=(-775, 300),
                color=INK2, fontsize=10, ha="left", va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    fig.suptitle("Seen from above: the hand's paths in dataset v6 and the plank rig's end post", color=INK,
                 fontsize=15.5, x=0.012, ha="left", fontweight="bold", y=0.985)
    fig.text(0.013, 0.94, f"Lines: the fixed finger's tip through every frame of {len(cuts)} recorded episodes "
             f"(1 in {a.every} of the {(~hold).sum()} with a cut).\nOrange ({n_near}): the sweep found the arm "
             f"within 40 mm of the end post as drawn.  Blue ({len(cuts) - n_near}): it stayed further away.\n"
             "A tip is one point of the hand: the fingers reach a few cm around it.", color=INK2, fontsize=10,
             ha="left", va="top", linespacing=1.35)
    fig.tight_layout(rect=(0, 0.02, 1, 0.885))
    fig.savefig(a.png, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
