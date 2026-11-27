"""Rig trio's camera post on the plank rig's cart: where it stands, what carries it, what it clears, what it sees.

    cd hardware/greenhouse/pipe_rail_rig && python cart_camera_post.py renders/cart_camera_post.png

(env lerobot-mj4.) Prints the numbers README.md quotes and draws the sketch. A geometry check on paper:
nothing here is built, and nothing here is in the Onshape model or its cut plan yet.

The lens positions and aims are the simulator's (sim/scene.py CART_CAMS, read at run time): dataset v6's pictures were
rendered from them, so on the real cart they are not a choice. Everything about HOW the post is carried (the boom,
the rod) is a proposal of 2026-11-27, drawn to fit the cart as pipe_rail_rig.fs v3 builds it.
"""
import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "sim"))
import scene  # noqa: E402
from lens import LENSES  # noqa: E402

# --- the plank rig, copied from pipe_rail_rig.fs v3 (2026-11-19) at its default parameters; all mm ---
L, P, GAP = 900.0, 400.0, 450.0      # inside length, row pitch, arm spacing (the feature's defaults)
T = 44.0                             # batten 44 x 44
PLANK_T, PLANK_W = 18.0, 119.0       # deck plank
TUBE_R, GAUGE, WHEEL_R = 9.5, 160.0, 25.0
STEM_OFF, STEM_R, FRUIT_R = 30.0, 4.0, 11.0
CROP_ABOVE = 520.0                   # string rail underside above the arm base
AISLE = P / 2
ARM_X = (L / 2 - GAP / 2, L / 2 + GAP / 2)          # arm 1 faces row B (+y), arm 2 faces row A (-y): armYaw [90, -90]
TUBE_Z = T + TUBE_R                                 # tube centre: the tubes rest on the 44 mm sleepers
WHEEL_Z = TUBE_Z + TUBE_R - 5 + WHEEL_R             # the groove sits 5 mm down over the tube
DECK_Z = WHEEL_Z + WHEEL_R + 8                      # deck underside
ARM_Z = DECK_Z + PLANK_T                            # deck top = the arm's base plane (no riser)
CROP_Z = ARM_Z + CROP_ABOVE
DECK_X = (ARM_X[0] - 140, ARM_X[1] + 140)
DECK_Y = (AISLE - PLANK_W, AISLE + PLANK_W)
WHEEL_X = (DECK_X[0] + 50, DECK_X[1] - 50)
WHEEL_Y = (AISLE - GAUGE / 2, AISLE + GAUGE / 2)
CLEAT_X = (ARM_X[0] - PLANK_W / 2, ARM_X[0] + PLANK_W / 2)      # cleat D-3 under arm 1, 18 mm under the deck
TUBE_X = (20.0, L + 300.0)
SLEEPER_X = (150.0, 450.0, 750.0, 1050.0)
STEM_DX = (-150.0, -75.0, 0.0, 75.0)                # stems in front of each arm
TRUSS_H = (160.0, 200.0, 280.0, 310.0)              # truss clip heights above the arm base (the .fs's two per stem)
FRUIT_DZ = (-55.0, -35.0)                           # fruit centres below their clip (the .fs's dz, least and most)
FRUIT_REACH = STEM_OFF + 35 + 4 + FRUIT_R           # a fruit's far side from the row line, toward the aisle

# --- the proposal (2026-11-27, not built): a boom under the deck's end, a threaded rod standing in its outer end ---
ROD_R = 6.0                          # M12 threaded rod; the simulator's post is 8 mm in radius (scene.BASE_POST_R)
NUT_R = 11.0                         # an M12 hex nut across its corners, 21.9 mm (ISO 4032), halved
BOOM = 44.0                          # the boom is one more length of the rig's 44 x 44 batten
BOOM_IN = (158.0, None)              # x of the boom's inner end (centre): just short of cleat D-3
BOOM_AT_WHEEL = (125.0, 228.0)       # a point the boom's centre line passes through, picked to clear the wheel bracket
BOOM_PAST_ROD = 40.0                 # wood left beyond the rod
HOLE_SIDE = 0.3                      # print clearance per side for any printed clamp on the rod (FDM rule: 0.3 mm)
# Sway of the bare rod, a beam-formula estimate (NOT measured): the rod clamped at the boom, the camera at its top.
E_STEEL = 200e3                      # MPa, textbook
ROD_KINDS = {"M12": (9.853, 0.728), "M16": (13.546, 1.332)}   # root diameter mm (ISO 724), kg per m (pitch diameter)
CAM_KG = 0.04                        # ⚠️ guess, unverified: one camera board with its clamp
BOOM_TWIST = 7.4                     # N/mm at the high camera from the boom twisting (44 x 44 spruce, G ~600 MPa,
#                                      rough)
PAN_AHEAD = 0.0388                   # m: the arm's pan axis stands this far toward the row from the simulator's arm
#                                      base origin (joint arm/shoulder_pan in the built model, read 2026-11-27)


def sim_to_rig(off, arm=0):
    """An offset from the arm base in the simulator's axes (m; x along the row, y toward the row the arm faces) to rig
    mm. Arm 2 is arm 1 turned half a turn (armYaw 90 and -90), so its x and y flip and nothing is mirrored."""
    s = 1.0 if arm == 0 else -1.0
    return np.array([ARM_X[arm] + s * 1000 * off[0], AISLE + s * 1000 * off[1], ARM_Z + 1000 * off[2]])


def cams():
    """{name: (lens offset from the arm base, unit view direction, housing centre offset)} from the simulator."""
    out = {}
    for name in scene.CART_CAMS:
        pos, aim = scene.fixed_cam(name)
        off = pos - np.asarray(scene.ARM_BASE, float)
        f = (aim - pos) / np.linalg.norm(aim - pos)
        out[name] = (off, f, off - f * (scene.MAST_LENS_GAP + scene.CAM_BOARD[2] / 2))
    return out


def box_points(lo, hi, step):
    """Points on the six faces of a box, `step` apart."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = []
    for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        gu = np.arange(lo[u], hi[u] + step / 2, step)
        gv = np.arange(lo[v], hi[v] + step / 2, step)
        uu, vv = np.meshgrid(gu, gv)
        for w in (lo[ax], hi[ax]):
            p = np.empty((uu.size, 3))
            p[:, ax], p[:, u], p[:, v] = w, uu.ravel(), vv.ravel()
            pts.append(p)
    return np.concatenate(pts)


def seen(name, boxes, step=0.5):
    """The 640 x 480 picture of camera `name` on arm 1's post: which pixels show one of the rig boxes (rig mm)."""
    pos, aim = scene.fixed_cam(name)
    x, y, f = scene._look_xyaxes(pos, aim)
    lens = LENSES[name]
    mask = np.zeros((lens.h, lens.w), np.uint8)
    base = np.asarray(scene.ARM_BASE, float)
    for lo, hi in boxes:
        w = base + (box_points(lo, hi, step) - np.array([ARM_X[0], AISLE, ARM_Z])) / 1000.0      # rig mm -> sim m
        c = w - pos
        cam = np.c_[c @ x, -(c @ y), c @ f]                    # OpenCV frame: x right, y down, z forward
        cam = cam[cam[:, 2] > 0.01]
        # ⚠️ the lens model folds back past ~81 deg off the axis (its one distortion term): keep to what a lens sees
        cam = cam[np.arctan2(np.hypot(cam[:, 0], cam[:, 1]), cam[:, 2]) < np.radians(75)]
        uv = np.round(lens.project(cam)).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < lens.w) & (uv[:, 1] >= 0) & (uv[:, 1] < lens.h)
        mask[uv[ok, 1], uv[ok, 0]] = 1
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))       # close the gaps between samples


def pixel(name, off):
    """Where a point (an offset from the arm base, m) lands in camera `name`'s picture."""
    pos, aim = scene.fixed_cam(name)
    x, y, f = scene._look_xyaxes(pos, aim)
    c = np.asarray(scene.ARM_BASE, float) + np.atleast_2d(off) - pos
    return LENSES[name].project(np.c_[c @ x, -(c @ y), c @ f])


def main(out):
    cam = cams()
    say = print
    # ---------------- where the post stands ----------------
    say("lens points and aims, from sim/scene.py (offsets from the arm base, cm; x = to the arm's right along the row, "
        "y = toward the row it faces, z = up from the deck top):")
    for name, (off, f, hc) in cam.items():
        say(f"  {name:5s} lens {np.round(off * 100, 1).tolist()}  turned {np.degrees(np.arctan2(f[1], f[0])):.1f} deg "
            f"from the aisle line toward the row, tilted {np.degrees(np.arcsin(f[2])):+.1f} deg  (housing centre "
            f"{np.round(hc * 100, 1).tolist()})")
    off0 = cam["high"][0]
    say(f"  by ruler from the pan axis (the upright axis the whole arm turns about): {-off0[0] * 100:.1f} cm to the "
        f"arm's left, {(off0[1] - PAN_AHEAD) * 100:.1f} cm toward the row")
    rod_off = np.mean([hc for _, _, hc in cam.values()], axis=0)      # the simulator's post stands under the housings
    rod = [sim_to_rig((rod_off[0], rod_off[1], 0), a)[:2] for a in (0, 1)]
    say(f"simulator's post (8 mm radius, under both housings): {np.round(rod_off[:2] * 100, 1).tolist()} cm from the "
        "arm base")
    for a in (0, 1):
        lens_rig = sim_to_rig(off0, a)
        end = 0.0 if a == 0 else L
        say(f"arm {a + 1} (at x {ARM_X[a]:.0f}): rod axis at rig x {rod[a][0]:.0f}, y {rod[a][1]:.0f} mm; lenses at x "
            f"{lens_rig[0]:.0f}, y {lens_rig[1]:.0f}, z {sim_to_rig(cam['low'][0], a)[2]:.0f} and {lens_rig[2]:.0f} "
            f"mm; the rod is {abs(rod[a][0] - end):.0f} mm outside the end frame's inside face "
            f"({abs(rod[a][0] - end) - T:.0f} mm beyond the 44 mm end post)")
    half = np.array([L, P]) - rod[0]
    assert np.allclose(rod[1], half), "arm 2's post is not arm 1's turned half a turn about the house centre"
    say(f"arm 2's post = arm 1's turned half a turn about the middle of the house (x -> {L:.0f} - x, y -> {P:.0f} - "
        "y): checked. ⚠️ Not a mirror image: a mirrored post would stand on that arm's right and its pictures would be "
        "flipped.")

    # ---------------- the boom ----------------
    r = rod[0]
    u = (np.array(BOOM_AT_WHEEL) - r) / np.linalg.norm(np.array(BOOM_AT_WHEEL) - r)      # from the rod toward the cart
    n = np.array([-u[1], u[0]])                                                          # across the boom
    ang = np.degrees(np.arctan2(abs(u[1]), abs(u[0])))
    yc = lambda x: r[1] + u[1] / u[0] * (x - r[0])                                       # noqa: E731  centre line
    inner = np.array([BOOM_IN[0], yc(BOOM_IN[0])])
    outer = r - u * BOOM_PAST_ROD
    corners = np.array([inner + n * BOOM / 2, inner - n * BOOM / 2, outer - n * BOOM / 2, outer + n * BOOM / 2])
    length = np.linalg.norm(inner - outer)
    half_w = BOOM / 2 / abs(u[0])                                                        # half the boom's width along y
    boom_z = (DECK_Z - BOOM, DECK_Z)
    say(f"boom: 44 x 44 batten, {length:.0f} mm long, under the deck (its top against the deck's underside, z "
        f"{boom_z[0]:.0f}-{boom_z[1]:.0f}), turned {ang:.1f} deg from the aisle line; inner end at x {inner[0]:.0f}, y "
        f"{inner[1]:.0f}; outer end at x {outer[0]:.0f}, y {outer[1]:.0f}; {DECK_X[0] - outer[0]:.0f} mm reaches past "
        f"the deck's end, {(inner[0] - DECK_X[0]) / abs(u[0]):.0f} mm lies under the deck")
    checks = []

    def check(what, gap, least):
        checks.append((what, gap, least))
        say(f"  {'ok ' if gap >= least else '⚠️ '} {what}: {gap:.0f} mm (wanted at least {least:.0f})")

    say("clearances, arm 1 (arm 2's are the same by the half turn):")
    plate_y = WHEEL_Y[1] - 15                                   # the wheel bracket's plate nearest the aisle centre
    check("boom side to the wheel bracket plate", plate_y - (yc(WHEEL_X[0] - 20) + half_w), 8)
    check("boom side to the wheel's axle bolt head", (WHEEL_Y[1] - 19) - (yc(WHEEL_X[0]) + half_w), 8)
    check("boom's inner end to cleat D-3 (the arm bolts through that cleat)", CLEAT_X[0] - corners[:2, 0].max(), 1)
    check("boom underside over the rail tubes", boom_z[0] - (TUBE_Z + TUBE_R), 6)
    check("boom underside over the sleepers and the closed end's floor tie", boom_z[0] - T, 10)
    check("fixing screws' line from the seam between the two deck strips (they must bite strip D-2)",
          yc(BOOM_IN[0] - 10) - AISLE, 12)
    in_frame = [yc(x) + s * half_w for x in (-T, 0.0) for s in (-1, 1)]
    check("boom to the row post where it passes through the end frame", (P - T / 2) - max(in_frame), 30)
    check("boom's far corner to the row post when the cart rolls it through", (P - T / 2) - corners[:, 1].max(), 30)
    check("rod to the row post when the cart rolls it through the end frame", (P - T / 2) - (r[1] + NUT_R), 30)
    check("rod to the stems (dowels) of the row it passes", (P - STEM_OFF - STEM_R) - (r[1] + ROD_R), 30)
    check("rod to the printed fruit of the row it passes", (P - FRUIT_REACH) - (r[1] + ROD_R), 10)
    hi_c, lo_c = sim_to_rig(cam["high"][2])[2], sim_to_rig(cam["low"][2])[2]
    reach = np.hypot(*scene.CAM_BOARD[:2]) / 2 * 1000           # a housing's corner from its centre, whatever its tilt
    fruit_lo, fruit_hi = ARM_Z + TRUSS_H[0] + FRUIT_DZ[0] - FRUIT_R, ARM_Z + TRUSS_H[-1] + FRUIT_DZ[1] + FRUIT_R
    check("high camera housing over the highest printed fruit (it passes over them)", (hi_c - reach) - fruit_hi, 30)
    check("low camera housing under the lowest printed fruit (it passes under them)", fruit_lo - (lo_c + reach), 20)
    check("high camera housing under the end frame's top tie", (CROP_Z + T) - (hi_c + reach), 50)
    check("low camera housing over the deck plane", (lo_c - reach) - ARM_Z, 5)
    say(f"rod: M12, from the boom's underside (nut sunk into the wood: nothing below z {boom_z[0]:.0f}) to z "
        f"{hi_c + 5:.0f}: {hi_c + 5 - boom_z[0]:.0f} mm, so one 1 m rod cut in half gives both posts; free above the "
        f"boom to the high camera: {hi_c - DECK_Z - 12:.0f} mm; hole for a printed clamp: "
        f"{2 * (ROD_R + HOLE_SIDE):.1f} mm")
    assert all(g >= least for _, g, least in checks), "a clearance is under what was wanted: see the ⚠️ lines"
    free = hi_c - DECK_Z - 12                                   # rod length that can bend, mm
    for kind, (d, kg_m) in ROD_KINDS.items():
        k = 3 * E_STEEL * (np.pi * d ** 4 / 64) / free ** 3      # N/mm at the top of a clamped rod
        k_all = 1 / (1 / k + 1 / BOOM_TWIST)
        mass = 0.236 * kg_m * free / 1000 + CAM_KG              # 33/140 of a clamped rod's mass acts at its tip
        hz = np.sqrt(k_all * 1000 / mass) / (2 * np.pi)
        say(f"sway estimate, {kind} rod (beam formula, not measured): {k:.1f} N/mm at the high camera, "
            f"{k_all:.1f} N/mm with the boom's twist; first sway near {hz:.0f} Hz")

    # ---------------- what the two cameras see of the rig's end frame ----------------
    frame = [((-T, P - T / 2, T), (0, P + T / 2, CROP_Z)),                # row B post of the closed end
             ((-T, -T / 2, T), (0, T / 2, CROP_Z)),                       # row A post of the closed end
             ((-T, T / 2, 0), (0, P - T / 2, T)),                         # floor tie
             ((-T, -T / 2, CROP_Z + T), (0, P + T / 2, CROP_Z + 2 * T))]  # top tie
    band = [(x, 0.17, z) for x in (-0.17, 0.11) for z in (0.13, 0.31)]    # where the trusses are cut (notes 24)
    masks = {}
    for name in cam:
        m = seen(name, frame)
        cols = np.flatnonzero(m.any(0))
        uv = pixel(name, np.array(band))
        masks[name] = (m, uv)
        hit = m[np.clip(np.round(uv[:, 1]).astype(int), 0, m.shape[0] - 1),
                np.clip(np.round(uv[:, 0]).astype(int), 0, m.shape[1] - 1)].any()
        say(f"{name} camera: the rig's end frame fills {100 * m.mean():.0f} % of the picture"
            + (f" (columns {cols.min()}-{cols.max()} of 640)" if cols.size else "")
            + f"; the truss band's corners land at columns {uv[:, 0].min():.0f}-{uv[:, 0].max():.0f}"
            + (" ⚠️ and one is behind the frame" if hit else ", clear of it"))
    say("⚠️ In the simulator the end frame stands 370 mm from the arm, behind this post, and is in neither picture.")

    # ---------------- noticed on the way: the crate ----------------
    import truss_scene as ts                                    # read only: the dataset's crate (world "crate23")
    (cx, cy), (sx, sy, _) = ts.WORLDS["crate23"]
    lo = np.array([cx - sx / 2, cy - sy / 2]) - np.asarray(scene.ARM_BASE[:2])
    hi = np.array([cx + sx / 2, cy + sy / 2]) - np.asarray(scene.ARM_BASE[:2])
    c1 = np.sort([sim_to_rig((*lo, 0), 0)[:2], sim_to_rig((*hi, 0), 0)[:2]], axis=0)
    c2 = np.sort([sim_to_rig((*lo, 0), 1)[:2], sim_to_rig((*hi, 0), 1)[:2]], axis=0)
    over = np.minimum(c1[1], c2[1]) - np.maximum(c1[0], c2[0])
    box = lambda c: f"x {c[0][0]:.0f}-{c[1][0]:.0f}, y {c[0][1]:.0f}-{c[1][1]:.0f}"      # noqa: E731
    say(f"the dataset's crate lies {lo[0] * 1000:.0f} to {hi[0] * 1000:.0f} mm to the arm's right. On this cart: "
        f"arm 1's at {box(c1)}; arm 2's at {box(c2)}: they overlap by {over[0]:.0f} mm along the cart and "
        f"{over[1]:.0f} mm across; the deck is y {DECK_Y[0]:.0f}-{DECK_Y[1]:.0f}, so each reaches "
        f"{DECK_Y[0] - c1[0][1]:.0f} mm past its edge. ⚠️ No crate is drawn on the rig.")

    # ---------------- the sketch ----------------
    ink, ink2, muted, surf = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb"
    orange, blue, aqua, wood = "#eb6834", "#2a78d6", "#1baf7a", "#d9c7a3"
    fig = plt.figure(figsize=(16, 9.4), dpi=110)
    fig.patch.set_facecolor(surf)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1.15, 1, 1], hspace=0.28, wspace=0.16)
    top, side = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    pics = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    note = fig.add_subplot(gs[1, 2])
    for a in (top, side):
        a.set_facecolor(surf); a.set_aspect("equal")
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(colors=muted, labelsize=8, length=0)
        a.grid(True, color="#e9e8e4", linewidth=0.7); a.set_axisbelow(True)

    # from above: the closed end of the rig with arm 1
    top.set_title("From above: the closed end of the rig, arm 1", color=ink, fontsize=13, loc="left", fontweight="bold")
    top.set_xlim(-560, 420); top.set_ylim(-95, 500)
    top.set_xlabel("along the rows, mm (0 = inside face of the end posts)", color=ink2, fontsize=9)
    top.set_ylabel("across, mm (row A at 0, row B at 400)", color=ink2, fontsize=9)
    for ry in (0, P):
        top.add_patch(Rectangle((-T, ry - T / 2), 500, T, facecolor=wood, alpha=0.45, edgecolor=muted, lw=0.8))
        top.add_patch(Rectangle((-T, ry - T / 2), T, T, facecolor=wood, edgecolor=ink2, lw=1.2))
    top.add_patch(Rectangle((-T, T / 2), T, P - T, facecolor=wood, alpha=0.75, edgecolor=ink2, lw=0.8))
    top.text(-T / 2, 80, "end frame", color=ink2, fontsize=9, rotation=90, ha="center", va="center")
    top.text(290, P + 30, "row B: the row arm 1 picks", color=ink, fontsize=10, ha="center")
    for wy in WHEEL_Y:
        top.plot([TUBE_X[0], 420], [wy, wy], color=muted, lw=5, alpha=0.35, solid_capstyle="butt")
        top.add_patch(Rectangle((WHEEL_X[0] - 20, wy - 15), 40, 30, facecolor=blue, alpha=0.55, edgecolor=blue, lw=1))
    top.add_patch(Rectangle((DECK_X[0], DECK_Y[0]), 420 - DECK_X[0], PLANK_W * 2, facecolor="#e9e8e4", alpha=0.75,
                            edgecolor=ink2, lw=1.2))
    top.text(350, AISLE - 30, "cart deck", color=ink2, fontsize=10, ha="center")
    top.add_patch(Rectangle((CLEAT_X[0], DECK_Y[0]), PLANK_W, PLANK_W * 2, facecolor="none", edgecolor=muted, lw=1,
                            ls=(0, (4, 3))))
    top.add_patch(Circle((ARM_X[0], AISLE), 45, facecolor=surf, edgecolor=ink, lw=1.5))
    top.text(ARM_X[0], AISLE, "arm 1", color=ink, fontsize=10, ha="center", va="center")
    for dx in STEM_DX:
        top.add_patch(Circle((ARM_X[0] + dx, P - STEM_OFF), 6, color=aqua))
        top.add_patch(Rectangle((ARM_X[0] + dx - 60, P - FRUIT_REACH), 120, FRUIT_REACH - STEM_OFF, facecolor=aqua,
                                alpha=0.18, lw=0))
    top.add_patch(Polygon(corners, closed=True, facecolor=orange, alpha=0.55, edgecolor=orange, lw=1.5))
    top.add_patch(Circle(r, 9, facecolor=ink, edgecolor=surf, lw=1.5, zorder=6))
    lens_xy = sim_to_rig(off0)[:2]
    fdir = cam["high"][1][:2] / np.linalg.norm(cam["high"][1][:2])
    for sgn in (-1, 1):
        a = np.radians(sgn * 36)
        d = np.array([fdir[0] * np.cos(a) - fdir[1] * np.sin(a), fdir[0] * np.sin(a) + fdir[1] * np.cos(a)])
        top.plot([lens_xy[0], lens_xy[0] + 560 * d[0]], [lens_xy[1], lens_xy[1] + 560 * d[1]], color=orange, lw=0.9,
                 ls=(0, (5, 4)))
    top.plot([lens_xy[0], lens_xy[0] + 330 * fdir[0]], [lens_xy[1], lens_xy[1] + 330 * fdir[1]], color=orange, lw=1.6)
    top.annotate("camera post: a threaded rod,\nboth cameras on it", r, xytext=(-545, 366), color=ink, fontsize=10,
                 fontweight="bold", va="bottom", arrowprops=dict(arrowstyle="-", color=ink, lw=1))
    top.annotate("boom: a 44 mm batten screwed under\nthe deck's end, between the wheels", (-90, yc(-90) - 20),
                 xytext=(-545, 30), color=ink, fontsize=10, fontweight="bold", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=orange, lw=1))
    top.annotate("", (r[0], -55), (ARM_X[0], -55), arrowprops=dict(arrowstyle="<->", color=ink2, lw=1))
    top.text((r[0] + ARM_X[0]) / 2, -66, f"rod {ARM_X[0] - r[0]:.0f} mm to the arm's left (lens 350)", color=ink2,
             fontsize=9, ha="center", va="top")
    top.annotate("", (r[0] - 55, AISLE), (r[0] - 55, r[1]), arrowprops=dict(arrowstyle="<->", color=ink2, lw=1))
    top.text(r[0] - 62, (AISLE + r[1]) / 2, f"{r[1] - AISLE:.0f} mm\ntoward\nthe row", color=ink2, fontsize=9,
             ha="right", va="center")
    top.plot([r[0] - 55, ARM_X[0] - 50], [AISLE, AISLE], color=muted, lw=0.8, ls=":")
    top.text(-545, 478, "orange lines: what the cameras take in\n(the end frame's row post is inside it)",
             color=orange, fontsize=9.5, ha="left", va="center")

    # from the side
    side.set_title("From the side", color=ink, fontsize=13, loc="left", fontweight="bold")
    side.set_xlim(-400, 330); side.set_ylim(-20, 770)
    side.set_xlabel("along the rows, mm", color=ink2, fontsize=9)
    side.set_ylabel("height above the floor, mm", color=ink2, fontsize=9)
    side.add_patch(Rectangle((-T, 0), T, CROP_Z + 2 * T, facecolor=wood, alpha=0.5, edgecolor=ink2, lw=1))
    side.add_patch(Rectangle((-T, CROP_Z), 374, T, facecolor=wood, alpha=0.45, edgecolor=muted, lw=0.8))
    side.add_patch(Rectangle((-T, 0), T, T, facecolor=wood, edgecolor=ink2, lw=1))
    side.text(T / 2, CROP_Z + 2 * T + 8, "end frame", color=ink2, fontsize=9, ha="left")
    side.plot([TUBE_X[0], 330], [TUBE_Z, TUBE_Z], color=muted, lw=6, alpha=0.4, solid_capstyle="butt")
    side.add_patch(Rectangle((SLEEPER_X[0] - T / 2, 0), T, T, facecolor=wood, edgecolor=muted, lw=0.8))
    side.add_patch(Circle((WHEEL_X[0], WHEEL_Z), WHEEL_R, facecolor=blue, alpha=0.55, edgecolor=blue))
    side.add_patch(Rectangle((DECK_X[0], DECK_Z), 330 - DECK_X[0], PLANK_T, facecolor="#e9e8e4", edgecolor=ink2,
                             lw=1.2))
    side.add_patch(Rectangle((outer[0], boom_z[0]), inner[0] - outer[0], BOOM, facecolor=orange, alpha=0.55,
                             edgecolor=orange, lw=1.5))
    side.plot([r[0], r[0]], [boom_z[0], hi_c + 5], color=ink, lw=3.2, solid_capstyle="butt")
    side.plot([-400, 330], [ARM_Z, ARM_Z], color=ink2, lw=0.9, ls=(0, (2, 3)))
    side.text(-395, 762, "dotted line: the deck top.\nOnly the rod and the\ncameras stand above it.", color=ink2,
              fontsize=8.5, ha="left", va="top")
    side.add_patch(Rectangle((ARM_X[0] - 45, ARM_Z), 90, 60, facecolor=surf, edgecolor=ink, lw=1.5))
    side.text(ARM_X[0], ARM_Z + 30, "arm 1", color=ink, fontsize=9.5, ha="center", va="center")
    side.text(outer[0] - 10, sum(boom_z) / 2, "boom", color=orange, fontsize=9, ha="right", va="center",
              fontweight="bold")
    for name, label in (("high", "HIGH camera,\nlens 420 mm\nabove the deck"),
                        ("low", "LOW camera,\nlens 50 mm\nabove the deck")):
        lp, f = sim_to_rig(cam[name][0]), cam[name][1]
        side.plot([lp[0], lp[0] + 330 * f[0]], [lp[2], lp[2] + 330 * f[2]], color=orange, lw=1.5)
        side.add_patch(Circle((lp[0], lp[2]), 11, facecolor=orange, edgecolor=surf, lw=1.5, zorder=6))
        side.text(lp[0] - 22, lp[2], label, color=ink, fontsize=9, ha="right", va="center")
    side.add_patch(Rectangle((ARM_X[0] - 150, fruit_lo), 225, fruit_hi - fruit_lo, facecolor=aqua, alpha=0.16, lw=0))
    side.text(ARM_X[0] + 70, fruit_hi + 8, "printed fruit", color=ink2, fontsize=8.5, ha="right")

    # what each camera sees of the end frame
    for a, name in zip(pics, cam):
        m, uv = masks[name]
        a.set_facecolor(surf)
        a.imshow(np.where(m > 0, 0.55, 1.0), cmap="gray", vmin=0, vmax=1, extent=(0, 640, 480, 0))
        a.add_patch(Polygon(uv[[0, 1, 3, 2]], closed=True, facecolor=aqua, alpha=0.25, edgecolor=aqua, lw=1.5))
        a.text(uv[:, 0].max() + 12, uv[:, 1].mean(), "where the\ntrusses hang", color=ink, fontsize=9, ha="left",
               va="center")
        a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(muted)
        a.set_title(f"{name.upper()} camera's picture: grey = end frame ({100 * m.mean():.0f} %)", color=ink,
                    fontsize=11, loc="left", fontweight="bold")
    note.axis("off")
    note.text(0, 1.0, "Why the pictures show the end frame", color=ink, fontsize=11, fontweight="bold", va="top")
    note.text(0, 0.86, "Dataset v6's pictures were made with the cameras\n35 cm to the arm's left. On this rig the end "
              "frame\nis only 22.5 cm from the arm, so the post stands\noutside it and looks in through it: about a "
              "quarter\nof each picture is the end frame.\n\nIn the simulator the end frame is 37 cm away,\nbehind "
              "the cameras, and in neither picture.\n\nThe trusses themselves are not hidden.\n\nArm 2 gets the same "
              "post turned half a turn\n(not a mirror image), outside the open end.\n\nIn the drawings: orange = "
              "the new parts, blue = wheels,\ngrey bars = rail tubes, dashed box = the cleat under\nthe arm, green = "
              "stems and fruit. Nothing is built yet.",
              color=ink2, fontsize=9.5, va="top", linespacing=1.35)
    fig.suptitle("Dataset v6's camera post on the plank rig's cart (proposal, to scale)", color=ink, fontsize=16,
                 x=0.012, ha="left", fontweight="bold", y=0.985)
    fig.savefig(out, facecolor=surf, bbox_inches="tight")
    say(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
