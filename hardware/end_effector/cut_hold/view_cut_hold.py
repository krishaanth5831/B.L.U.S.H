#!/usr/bin/env python3
"""Open a window with the v6 cut-and-hold jaw and a jaw-angle slider.

  python view_cut_hold.py      3D view; drag to orbit, slider = jaw angle q (deg); -10 = the closed stop

Grey = STS3215 gripper servo, blue = PLA jaws, dark = blades, green = pads (at rest, before a stalk pushes them back).
Needs: numpy shapely trimesh manifold3d matplotlib networkx (an interactive backend, e.g. TkAgg or QtAgg)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cut_hold_gripper as ch

fg = ch.fg
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def collection(mesh, rgb, alpha=1.0):
    shade = 0.45 + 0.55 * np.clip(mesh.face_normals @ LIGHT, 0, 1)
    colors = np.c_[np.outer(shade, rgb), np.full(len(shade), alpha)]
    return Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colors, linewidths=0)


def main():
    parts = ch.build()
    servo = fg.load_stock("sts3215_03a_v1.stl", fg.SERVO_VIS)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.12)
    static = [(servo, (0.6, 0.6, 0.6), 0.35), (parts["fixed"]["jaw"], (0.2, 0.45, 0.8), 1.0),
              (parts["fixed"]["blade"], (0.15, 0.15, 0.15), 1.0), (parts["fixed"]["pad"], (0.2, 0.7, 0.3), 1.0)]
    for m, c, a in static:
        ax.add_collection3d(collection(m, c, a))
    moving = [(parts["moving"]["jaw"], (0.2, 0.45, 0.8)), (parts["moving"]["blade"], (0.15, 0.15, 0.15)),
              (parts["moving"]["pad"], (0.2, 0.7, 0.3))]
    live = []

    def draw(q):
        for c in live:
            c.remove()
        live.clear()
        T = fg.moving_T(q) @ np.linalg.inv(fg.moving_T(0.0))
        for m, c in moving:
            mm = m.copy()
            mm.apply_transform(T)
            col = collection(mm, c)
            ax.add_collection3d(col)
            live.append(col)
        fig.canvas.draw_idle()

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-120, 0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (across the jaw)")
    ax.set_ylabel("y (along the stalk)")
    ax.set_zlabel("z (along the fingers)")
    s = Slider(plt.axes([0.15, 0.04, 0.7, 0.03]), "jaw q (deg)", ch.Q_CLOSED, 60.0, valinit=25.0)
    s.on_changed(draw)
    draw(25.0)
    plt.show()


if __name__ == "__main__":
    main()
