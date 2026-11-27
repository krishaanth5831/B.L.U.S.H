"""Sketch of where rig trio's camera post stands, from above and from the side (to scale, simulator positions).

    python whatif/camera_post_sketch.py ../outputs/v6/cams/where_the_camera_post_stands.png     # from sim/

A drawing for people, not a measurement: the numbers are copied from scene.py and truss_scene.py (comments below).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # reference palette slots 1-3
ARM = (0.37, 0.20)                       # scene.ARM_BASE
POST = (0.02, 0.30)                      # arm base + CART_CAMS offset (-0.35, +0.10)
HIGH_Z, LOW_Z = 0.42, 0.05               # lens heights (scene.CART_CAMS)
AIM_H, AIM_L = (0.30, 0.37, 0.22), (0.34, 0.37, 0.26)
OLD = (0.09, 0.26, 0.10)                 # arm base + BASE_CAM_OFFSET
ROW_X, ROW_Y = (0.20, 0.48), 0.38        # truss_scene.STEM_X; stems hang at y 0.38
CRATE = (0.5025, 0.03, 0.23, 0.28)       # crate23: centre (0.6175, 0.17), 23 x 28 cm
BAND_Z = (0.13, 0.31)                    # cut points of the trusses (notes section 24)
FRAME_X = ARM[0] - 0.225                 # plank rig as designed: end frames 225 mm from the arm

fig, (ax, az) = plt.subplots(1, 2, figsize=(15, 6.3), dpi=120, gridspec_kw=dict(width_ratios=[1.25, 1]))
fig.patch.set_facecolor(SURFACE)
for a in (ax, az):
    a.set_facecolor(SURFACE)
    a.set_aspect("equal")
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(colors=MUTED, labelsize=9, length=0)
    a.grid(True, color="#e9e8e4", linewidth=0.8)
    a.set_axisbelow(True)

# ---------------- seen from above ----------------
ax.set_title("Seen from above", color=INK, fontsize=15, loc="left", pad=12, fontweight="bold")
ax.set_xlim(-0.17, 0.80); ax.set_ylim(-0.12, 0.52)
ax.set_xlabel("along the row, m", color=INK2, fontsize=10); ax.set_ylabel("toward the row, m", color=INK2, fontsize=10)
# what the high camera takes in (72 deg wide lens), as a faint wedge
ang = np.degrees(np.arctan2(AIM_H[1] - POST[1], AIM_H[0] - POST[0]))
ax.add_patch(Wedge(POST, 0.60, ang - 36, ang + 36, color=ORANGE, alpha=0.07, lw=0))
ax.plot([POST[0], AIM_H[0]], [POST[1], AIM_H[1]], color=ORANGE, lw=1.6)
ax.plot([POST[0], AIM_L[0]], [POST[1], AIM_L[1]], color=ORANGE, lw=1.6, ls=(0, (4, 3)))
# the row of plants
ax.plot(ROW_X, [ROW_Y, ROW_Y], color=AQUA, lw=7, alpha=0.35, solid_capstyle="round")
for x in np.linspace(ROW_X[0], ROW_X[1], 5):
    ax.add_patch(Circle((x, ROW_Y), 0.008, color=AQUA))
ax.text(0.40, 0.408, "row of plants (the trusses hang here)", color=INK, fontsize=11, ha="center")
# the crate, the arm
ax.add_patch(Rectangle(CRATE[:2], CRATE[2], CRATE[3], facecolor=BLUE, alpha=0.25, edgecolor=BLUE, lw=1.5))
ax.text(CRATE[0] + CRATE[2] / 2, CRATE[1] + CRATE[3] / 2, "crate", color=INK, fontsize=11, ha="center", va="center")
ax.add_patch(Circle(ARM, 0.045, facecolor="#e9e8e4", edgecolor=INK, lw=1.5))
ax.text(ARM[0], ARM[1], "arm", color=INK, fontsize=11, ha="center", va="center")
# the old camera
ax.add_patch(Rectangle((OLD[0] - 0.012, OLD[1] - 0.012), 0.024, 0.024, facecolor=MUTED, edgecolor=MUTED))
ax.text(OLD[0], OLD[1] - 0.022, "old base\ncamera", color=INK2, fontsize=9.5, ha="center", va="top")
# the new post
ax.add_patch(Circle(POST, 0.016, facecolor=ORANGE, edgecolor=SURFACE, lw=2, zorder=5))
ax.annotate("NEW: one post, both cameras\nhigh lens 42 cm up, low lens 5 cm up", POST, xytext=(-0.16, 0.455),
            color=INK, fontsize=11, ha="left", fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
# distances from the arm to the post
yb = 0.02
ax.annotate("", (POST[0], yb), (ARM[0], yb), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
ax.text((FRAME_X + ARM[0]) / 2 + 0.004, yb + 0.012, "35 cm to the arm's left", color=INK2, fontsize=10, ha="center")
ax.plot([POST[0], POST[0]], [yb, POST[1] - 0.02], color=MUTED, lw=0.8, ls=":")
ax.plot([ARM[0], ARM[0]], [yb, ARM[1] - 0.05], color=MUTED, lw=0.8, ls=":")
xb = -0.03
ax.annotate("", (xb, ARM[1]), (xb, POST[1]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
ax.text(xb - 0.012, (ARM[1] + POST[1]) / 2, "10 cm toward\nthe row", color=INK2, fontsize=10, ha="right", va="center")
ax.plot([xb, ARM[0] - 0.05], [ARM[1], ARM[1]], color=MUTED, lw=0.8, ls=":")
ax.plot([xb, POST[0] - 0.02], [POST[1], POST[1]], color=MUTED, lw=0.8, ls=":")
# the plank rig's end frame
ax.plot([FRAME_X, FRAME_X], [-0.10, 0.36], color=INK2, lw=1.4, ls=(0, (6, 4)))
ax.text(FRAME_X + 0.012, -0.105, "end frame of the plank rig as designed (22.5 cm from the arm):\nthe new post "
        "stands 12.5 cm outside it", color=INK2, fontsize=10, ha="left", va="bottom")

# ---------------- seen from the side ----------------
az.set_title("Seen from the side (heights)", color=INK, fontsize=15, loc="left", pad=12, fontweight="bold")
az.set_xlim(-0.17, 0.62); az.set_ylim(-0.06, 0.52)
az.set_xlabel("along the row, m", color=INK2, fontsize=10)
az.set_ylabel("height above the deck, m", color=INK2, fontsize=10)
az.add_patch(Rectangle((ROW_X[0], BAND_Z[0]), ROW_X[1] - ROW_X[0], BAND_Z[1] - BAND_Z[0], facecolor=AQUA,
                       alpha=0.22, edgecolor=AQUA, lw=1.2))
az.text((ROW_X[0] + ROW_X[1]) / 2, BAND_Z[1] + 0.012, "where the trusses are cut\n(13 to 31 cm up)", color=INK,
        fontsize=10.5, ha="center", va="bottom")
az.plot([POST[0], POST[0]], [0, HIGH_Z + 0.02], color=ORANGE, lw=4, solid_capstyle="butt")
for z, aim, ls in ((HIGH_Z, AIM_H, "-"), (LOW_Z, AIM_L, (0, (4, 3)))):
    az.plot([POST[0], aim[0]], [z, aim[2]], color=ORANGE, lw=1.6, ls=ls)
    az.add_patch(Circle((POST[0], z), 0.013, facecolor=ORANGE, edgecolor=SURFACE, lw=2, zorder=5))
az.text(POST[0] + 0.02, HIGH_Z + 0.03, "HIGH camera, 42 cm:\nlooks down over the arm", color=INK, fontsize=10.5,
        ha="left", va="bottom")
az.text(POST[0] - 0.015, LOW_Z + 0.03, "LOW camera,\n5 cm: looks\nup along\nthe row", color=INK, fontsize=10.5,
        ha="right", va="bottom")
az.add_patch(Rectangle((OLD[0] - 0.012, OLD[2] - 0.012), 0.024, 0.024, facecolor=MUTED, edgecolor=MUTED))
az.annotate("old base camera", (OLD[0], OLD[2]), xytext=(0.13, 0.035), color=INK2, fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
az.add_patch(Rectangle((ARM[0] - 0.045, 0), 0.09, 0.05, facecolor="#e9e8e4", edgecolor=INK, lw=1.5))
az.text(ARM[0], -0.012, "arm base", color=INK, fontsize=10.5, ha="center", va="top")
az.plot([-0.17, 0.62], [0, 0], color=INK2, lw=1)
fig.suptitle("Where the new camera post stands (simulator positions, to scale)", color=INK, fontsize=17, x=0.012,
             ha="left", fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(sys.argv[1], facecolor=SURFACE)
print("wrote", sys.argv[1])
