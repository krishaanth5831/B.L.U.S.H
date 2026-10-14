#!/usr/bin/env python3
"""Open a window with the assembled fin-ray gripper and a jaw-angle slider.

  python view_gripper.py            3D view; drag to orbit, slider = jaw angle q (deg)
  python view_gripper.py --stock    also ghost the stock fingers, to compare outlines

Grey = STS3215 gripper servo, blue = PLA stubs, orange = TPU fin rays.
Needs: numpy shapely trimesh manifold3d matplotlib (an interactive backend, e.g. TkAgg or QtAgg)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import finray_gripper as fg

LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])   # fixed light for simple shading

def collection(mesh, rgb, alpha=1.0):
    shade = 0.45 + 0.55 * np.clip(mesh.face_normals @ LIGHT, 0, 1)
    colors = np.c_[np.outer(shade, rgb), np.full(len(shade), alpha)]
    return Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colors, linewidths=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", action="store_true", help="ghost the stock fingers")
    a = ap.parse_args()
    parts = fg.build()
    servo = fg.load_stock("sts3215_03a_v1.stl", fg.SERVO_VIS)
    q0_inv = np.linalg.inv(fg.moving_T(0.0))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0.0, 0.1, 1.0, 0.9], projection="3d")
    for m, c, al in ((servo, (0.6, 0.6, 0.6), 1.0), (parts["fixed"]["stub"], (0.25, 0.45, 0.8), 1.0),
                     (parts["fixed"]["finger"], (1.0, 0.55, 0.1), 1.0)):
        ax.add_collection3d(collection(m, c, al))
    if a.stock:
        ax.add_collection3d(collection(parts["fixed"]["stock"], (0.3, 0.3, 0.3), 0.12))
    ax.set_xlim(-60, 110); ax.set_ylim(-85, 85); ax.set_zlim(-130, 40)
    ax.set_box_aspect((1, 1, 1)); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (mm)")
    ax.view_init(elev=15, azim=-80)
    moving = []

    def draw(q):
        for c in moving: c.remove()
        moving.clear()
        Tq = fg.moving_T(q) @ q0_inv
        items = [(parts["moving"]["stub"], (0.25, 0.45, 0.8), 1.0), (parts["moving"]["finger"], (1.0, 0.55, 0.1), 1.0)]
        if a.stock:
            items.append((parts["moving"]["stock"], (0.3, 0.3, 0.3), 0.12))
        for m, c, al in items:
            mm = m.copy(); mm.apply_transform(Tq)
            moving.append(ax.add_collection3d(collection(mm, c, al)))
        ax.set_title(f"SO-101 fin-ray gripper, jaw q = {q:.0f} deg  (closed limit {fg.Q_LIMITS_DEG[0]:.0f}, open {fg.Q_LIMITS_DEG[1]:.0f})")
        fig.canvas.draw_idle()

    s = Slider(fig.add_axes([0.15, 0.03, 0.7, 0.03]), "jaw q (deg)", *fg.Q_LIMITS_DEG, valinit=0.0)
    s.on_changed(draw)
    draw(0.0)
    plt.show()

if __name__ == "__main__":
    main()
