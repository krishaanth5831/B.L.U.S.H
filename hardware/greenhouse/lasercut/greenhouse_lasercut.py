#!/usr/bin/env python3
"""SO-101 test greenhouse, laser-cut version: parametric generator.

Same layout as the 3D-printed design (so101_greenhouse.fs): 3 vine rows at 400 mm pitch,
740 mm between end frames, crop rails at 520 mm, the SO-101 mid-aisle with nothing it can
reach except the floor. Built from 6 mm birch plywood:

  end frames (x = 0 and x = 740)   one gable-shaped panel each, YZ plane, glued from pieces
                                   when the laser bed is smaller than the panel
  crop rails (z = 520)             T-beams: vertical web on top of a horizontal keyhole flange
  anchor rails (table)             inverted T: web on the table, keyhole flange on top
  ridge                            a web along the apex
  every rail end                   a tusk tenon through the end frame, locked by a wedge

Usage:
  python greenhouse_lasercut.py check                 geometry + 3D checks + previews (no cut files)
  python greenhouse_lasercut.py build [--bed 600x400] [--t 6.0] [--out DIR]
                                                      checks, then nested cut sheets (DXF + SVG)

Needs: numpy shapely trimesh manifold3d mapbox-earcut ezdxf matplotlib
"""
import argparse
import csv
import math
import pathlib
import sys

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import polylabel, unary_union

HERE = pathlib.Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------
# Parameters (mm). Change here; everything else is derived.
# ----------------------------------------------------------------------------------------
T = 6.0                 # sheet thickness. ⚠️ MEASURE the real sheet; every slot is sized from this.
HOUSE_L = 740.0         # inner length between end frames (x = 0 .. 740). <740 hit the reach sphere in print.
ROWS = [400.0, 0.0, -400.0]                     # vine rows (y)
ROW_NAME = {400.0: "R1", 0.0: "R2", -400.0: "R3"}
ARM_X, ARM_YS, ARM_Z, REACH = 370.0, (200.0, -200.0), 117.0, 400.0   # SO-101 shoulder + full reach
FLOOR_Z = 40.0          # below this a part counts as floor (allowed inside the reach sphere, as in the print)

POST_W = 60.0           # end-frame post width (y)
EAR = 40.0              # clamp ears: base tie runs this far past the outer posts
FIX_HOLES_Y = (-450.0, -200.0, 200.0, 450.0)   # 4.5 mm holes in the base tie for steel angle brackets to the table
FIX_HOLE_D = 4.5
BASE_H = 30.0           # base tie height
CROP_Z = 520.0          # underside of the crop flange = where the vines hang from
BAND = (500.0, 590.0)   # crop tie band in the end frame (z)
ROOF_DEG = 15.0
RAFTER_W = 45.0         # rafter depth, measured square to the roof line

WEB_H = 50.0            # crop rail web height
ANCHOR_WEB_H = 30.0     # anchor rail web height (knots hang under its flange)
FLANGE_W = 50.0         # keyhole flange width
FLANGE_TAB_W = 40.0     # flange end tab through the end frame
TENON_INSET = 5.0       # tenon is this much shorter than the web, top and bottom
TENON_OUT = 22.0        # tenon length past the end frame's outer face
WEDGE_HOLE = 13.0       # wedge hole length along x (starts 1 mm inside the panel face)
WEDGE_LEN, WEDGE_W0, WEDGE_W1 = 70.0, 9.0, 15.0
RIDGE_H = 40.0
RIDGE_BELOW_APEX = 15.0

KEY_PITCH, KEY_OFF, KEY_D, KEY_SLOT, KEY_LEN = 20.0, 15.0, 5.0, 1.8, 6.0   # twine keyholes
WEB_TAB_LEN, WEB_TAB_PITCH, TAB_KEEP_OUT = 24.0, 110.0, 40.0              # web-to-flange tabs

# Camera mast (3D-printed, unchanged): default pose from the printed README
MAST_BASE = dict(x=-120.0, y=151.0, size=110.0)
LENS = np.array([-120.0, 200.0, 423.0])          # lens sits 49 mm to the side of the pole, in line with the aisle
LENS_TILT_DEG, LENS_HFOV, LENS_VFOV = 20.0, 80.0, 60.0
PICK_ZONE = dict(x=(300.0, 420.0), z=(100.0, 350.0))

YO = max(abs(r) for r in ROWS) + POST_W / 2       # outer post outer edge (430)
TAN = math.tan(math.radians(ROOF_DEG))
APEX = BAND[1] + YO * TAN
RIDGE_Z = (APEX - RIDGE_BELOW_APEX - RIDGE_H, APEX - RIDGE_BELOW_APEX)
QS = 16                 # circle segments per quarter

# rail cross-sections (z ranges)
CROP_FLANGE_Z = (CROP_Z, CROP_Z + T)
CROP_WEB_Z = (CROP_Z + T, CROP_Z + T + WEB_H)
ANCHOR_WEB_Z = (0.0, ANCHOR_WEB_H)
ANCHOR_FLANGE_Z = (ANCHOR_WEB_H, ANCHOR_WEB_H + T)


def tenon_z(zr):
    return zr[0] + TENON_INSET, zr[1] - TENON_INSET


# ----------------------------------------------------------------------------------------
# 2D part geometry. Each part: polygon in its own (u, v) plane + how it sits in 3D.
# ----------------------------------------------------------------------------------------
class Part:
    def __init__(self, name, poly, plane, qty=1, engrave=None, note=""):
        self.name, self.poly, self.plane, self.qty, self.note = name, poly, plane, qty, note
        self.engrave = engrave or []          # list of LineStrings (score lines), in part coords
        self.placements = []                  # list of 4x4 transforms (one per physical copy)


def plane_matrix(origin, u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    n = np.cross(u, v)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = u, v, n, origin
    return M


def keyhole(cx, cy):
    entry = Point(cx, cy).buffer(KEY_D / 2, quad_segs=QS)
    slot = box(cx, cy - KEY_SLOT / 2, cx + KEY_LEN, cy + KEY_SLOT / 2)
    end = Point(cx + KEY_LEN, cy).buffer(KEY_SLOT / 2, quad_segs=QS)
    return unary_union([entry, slot, end])


def end_frame():
    """Gable panel in (u=y, v=z), thickness along x."""
    members = [box(-YO - EAR, 0, YO + EAR, BASE_H), box(-YO, BAND[0], YO, BAND[1])]
    for r in ROWS:
        top = APEX if r == 0 else BAND[1]
        members.append(box(r - POST_W / 2, 0, r + POST_W / 2, top))
    drop = RAFTER_W / math.cos(math.radians(ROOF_DEG))
    for s in (1, -1):
        members.append(Polygon([(s * YO, BAND[1]), (0, APEX), (0, APEX - drop), (s * YO, BAND[1] - drop)]))
    envelope = Polygon([(-YO - EAR, 0), (YO + EAR, 0), (YO + EAR, BASE_H), (YO, BASE_H), (YO, BAND[1]),
                        (0, APEX), (-YO, BAND[1]), (-YO, BASE_H), (-YO - EAR, BASE_H)])
    frame = unary_union(members).intersection(envelope)
    holes = []
    for r in ROWS:
        holes.append(box(r - T / 2, *tenon_z(ANCHOR_WEB_Z)[:1], r + T / 2, tenon_z(ANCHOR_WEB_Z)[1]))
        holes.append(box(r - FLANGE_TAB_W / 2, ANCHOR_FLANGE_Z[0], r + FLANGE_TAB_W / 2, ANCHOR_FLANGE_Z[1]))
        holes.append(box(r - FLANGE_TAB_W / 2, CROP_FLANGE_Z[0], r + FLANGE_TAB_W / 2, CROP_FLANGE_Z[1]))
        holes.append(box(r - T / 2, tenon_z(CROP_WEB_Z)[0], r + T / 2, tenon_z(CROP_WEB_Z)[1]))
    holes.append(box(-T / 2, tenon_z(RIDGE_Z)[0], T / 2, tenon_z(RIDGE_Z)[1]))
    holes += [Point(y, BASE_H / 2).buffer(FIX_HOLE_D / 2, quad_segs=QS) for y in FIX_HOLES_Y]
    frame = frame.difference(unary_union(holes))
    eng = []
    for r in ROWS:   # row label above each crop mortise
        eng += text_lines(ROW_NAME[r], r - 7, CROP_FLANGE_Z[0] - 14, 9)
    return frame, eng


def web(zr, tabs_down, tab_xs):
    """Rail web in (u=x, v=z), thickness along y. Tusk tenons both ends, wedge holes, flange tabs."""
    tz = tenon_z(zr)
    parts = [box(0, zr[0], HOUSE_L, zr[1]),
             box(-T - TENON_OUT, tz[0], 0, tz[1]),
             box(HOUSE_L, tz[0], HOUSE_L + T + TENON_OUT, tz[1])]
    for xc in tab_xs:
        if tabs_down:
            parts.append(box(xc - WEB_TAB_LEN / 2, zr[0] - T, xc + WEB_TAB_LEN / 2, zr[0]))
        else:
            parts.append(box(xc - WEB_TAB_LEN / 2, zr[1], xc + WEB_TAB_LEN / 2, zr[1] + T))
    zc = (zr[0] + zr[1]) / 2
    holes = [box(-T - WEDGE_HOLE + 1, zc - T / 2, -T + 1, zc + T / 2),
             box(HOUSE_L + T - 1, zc - T / 2, HOUSE_L + T - 1 + WEDGE_HOLE, zc + T / 2)]
    return unary_union(parts).difference(unary_union(holes))


def ridge_web():
    tz = tenon_z(RIDGE_Z)
    zc = (RIDGE_Z[0] + RIDGE_Z[1]) / 2
    p = unary_union([box(0, RIDGE_Z[0], HOUSE_L, RIDGE_Z[1]),
                     box(-T - TENON_OUT, tz[0], 0, tz[1]),
                     box(HOUSE_L, tz[0], HOUSE_L + T + TENON_OUT, tz[1])])
    holes = [box(-T - WEDGE_HOLE + 1, zc - T / 2, -T + 1, zc + T / 2),
             box(HOUSE_L + T - 1, zc - T / 2, HOUSE_L + T - 1 + WEDGE_HOLE, zc + T / 2)]
    return p.difference(unary_union(holes))


def flange(tab_xs):
    """Keyhole flange in (u=x, v=y - row), thickness along z."""
    p = unary_union([box(0, -FLANGE_W / 2, HOUSE_L, FLANGE_W / 2),
                     box(-T, -FLANGE_TAB_W / 2, 0, FLANGE_TAB_W / 2),
                     box(HOUSE_L, -FLANGE_TAB_W / 2, HOUSE_L + T, FLANGE_TAB_W / 2)])
    holes = [box(xc - WEB_TAB_LEN / 2, -T / 2, xc + WEB_TAB_LEN / 2, T / 2) for xc in tab_xs]
    eng = []
    x = KEY_PITCH
    while x <= HOUSE_L - KEY_PITCH + 1e-6:
        for s in (1, -1):
            holes.append(keyhole(x, s * KEY_OFF))
            eng.append(LineString([(x, s * (FLANGE_W / 2 - 4)), (x, s * FLANGE_W / 2 - s * 0.01)]))
        if abs(round(x) % 100) < 1e-6:
            eng += text_lines(f"{int(round(x / 10))}", x + 8.5, FLANGE_W / 2 - 5.5, 4)    # cm, outer edge between keyholes
        x += KEY_PITCH
    return p.difference(unary_union(holes)), eng


def wedge():
    """Flat tapered wedge in (u along insertion, v = width), lies in the XY plane."""
    return Polygon([(0, 0), (WEDGE_LEN, 0), (WEDGE_LEN, WEDGE_W1), (0, WEDGE_W0)])


def test_comb():
    """Cut first: slot widths around the sheet thickness + the three keyhole slot widths."""
    widths = [T - 0.6, T - 0.4, T - 0.2, T, T + 0.2, T + 0.4]
    W = 20 + len(widths) * 22 + 40
    p = box(0, 0, W, 45)
    holes, eng = [], []
    for i, w in enumerate(widths):
        x = 20 + i * 22
        holes.append(box(x - w / 2, 25, x + w / 2, 45))
        eng += text_lines(f"{w:.1f}", x - 7, 14, 5)
    for i, s in enumerate((1.5, 1.8, 2.2)):
        cx, cy = 20 + len(widths) * 22 + 2, 8 + i * 13
        holes.append(unary_union([Point(cx, cy).buffer(KEY_D / 2, quad_segs=QS),
                                  box(cx, cy - s / 2, cx + KEY_LEN, cy + s / 2),
                                  Point(cx + KEY_LEN, cy).buffer(s / 2, quad_segs=QS)]))
        eng += text_lines(f"{s}", cx + 10, cy - 2.5, 5)
    eng += text_lines(f"TEST t={T:.1f}", 4, 3, 5)
    return p.difference(unary_union(holes)), eng


# ----------------------------------------------------------------------------------------
# Text as vector outlines (DXF TEXT is dropped by many laser programs)
# ----------------------------------------------------------------------------------------
def text_lines(s, x, y, h):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), s, size=h, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    out = []
    for poly in tp.to_polygons():
        if len(poly) >= 3:
            pts = [(x + px, y + py) for px, py in poly]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            out.append(LineString(pts))
    return out


# ----------------------------------------------------------------------------------------
# Splitting with glued dovetail joints, so every piece fits the laser bed
# ----------------------------------------------------------------------------------------
def dovetail(axis, c, vm, neck, head, depth, direction):
    """axis 'u': cut line u=c, joint spans v around vm; head pokes `depth` into side `direction`."""
    pts = [(c - direction * 0.5, vm - neck / 2), (c + direction * depth, vm - head / 2),
           (c + direction * depth, vm + head / 2), (c - direction * 0.5, vm + neck / 2)]
    p = Polygon(pts)
    return p if axis == "u" else affinity.affine_transform(p, [0, 1, 1, 0, 0, 0])   # swap u/v


def cut(poly, axis, c):
    """Split poly at u=c (axis 'u') or v=c (axis 'v'); add one dovetail per member crossing."""
    minx, miny, maxx, maxy = poly.bounds
    big = 1e4
    if axis == "u":
        lo, hi = box(-big, -big, c, big), box(c, -big, big, big)
        line = LineString([(c, miny - 1), (c, maxy + 1)])
    else:
        lo, hi = box(-big, -big, big, c), box(-big, c, big, big)
        line = LineString([(minx - 1, c), (maxx + 1, c)])
    A, B = poly.intersection(lo), poly.intersection(hi)
    inter = poly.intersection(line)
    segs = [inter] if inter.geom_type == "LineString" else [g for g in getattr(inter, "geoms", []) if g.geom_type == "LineString"]
    shrunk = poly.buffer(-2.0)
    for s in segs:
        coords = np.array(s.coords)
        k = 1 if axis == "u" else 0
        a, b = coords[:, k].min(), coords[:, k].max()
        w = b - a
        if w < 12:
            continue
        placed = False
        for scale in (1.0, 0.8, 0.6):
            head = min(0.45 * w, 28.0) * scale
            neck = 0.62 * head
            depth = min(max(0.55 * head, 7.0), 16.0)
            for direction in (1, -1):
                for off in (0, -0.15, 0.15, -0.3, 0.3):
                    vm = (a + b) / 2 + off * w
                    d = dovetail(axis, c, vm, neck, head, depth, direction)
                    probe = dovetail(axis, c, vm, neck, head, depth + 2.0, direction)
                    if shrunk.buffer(2.0).contains(probe) and poly.contains(d.buffer(3.0).intersection(hi if direction > 0 else lo)):   # >= 3 mm wood round the head, holes included
                        if direction > 0:
                            A, B = A.union(d.intersection(poly)), B.difference(d)
                        else:
                            B, A = B.union(d.intersection(poly)), A.difference(d)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError(f"no room for a dovetail at {axis}={c}, span {a:.0f}..{b:.0f}")
    return A, B


def single(p, what):
    if isinstance(p, MultiPolygon):
        ps = sorted(p.geoms, key=lambda g: -g.area)
        if ps[1].area > 1.0:
            raise RuntimeError(f"{what}: split left {len(ps)} separate islands")
        p = ps[0]
    return p.buffer(0)


def fits(poly, bed):
    w = poly.bounds[2] - poly.bounds[0]
    h = poly.bounds[3] - poly.bounds[1]
    W, H = bed[0] - 20 - 5, bed[1] - 20 - 5      # sheet border 10 each side + 5 mm part gap
    return (w <= W and h <= H) or (h <= W and w <= H)


def split_end_frame(frame, bed):
    """Try cut sets from fewest pieces up; return the first whose pieces all fit."""
    options = [[], [("v", 270.0)], [("v", 270.0), ("u", -175.0), ("u", 175.0)],
               [("v", 270.0), ("u", -175.0), ("u", 175.0), ("v", 420.0)]]
    for cuts in options:
        pieces = [frame]
        try:
            for axis, c in cuts:
                nxt = []
                for p in pieces:
                    lo_hi = (p.bounds[0] < c < p.bounds[2]) if axis == "u" else (p.bounds[1] < c < p.bounds[3])
                    if lo_hi:
                        a, b = cut(p, axis, c)
                        nxt += [single(a, "end frame"), single(b, "end frame")]
                    else:
                        nxt.append(p)
                pieces = nxt
        except RuntimeError:
            continue
        if all(fits(p, bed) for p in pieces):
            return pieces
    raise RuntimeError("end frame doesn't fit this bed with any cut set")


def split_strip(poly, positions, what):
    pieces, rest = [], poly
    for c in sorted(positions):
        a, b = cut(rest, "u", c)
        pieces.append(single(a, what))
        rest = single(b, what)
    pieces.append(rest)
    return pieces


def strip_splits(length_lo, length_hi, bed, stagger):
    usable = max(bed) - 25 - 16          # border + gap + room for the dovetail head
    total = length_hi - length_lo
    n = math.ceil(total / usable)
    if n <= 1:
        return []
    return [length_lo + k * total / n + stagger for k in range(1, n)]


# ----------------------------------------------------------------------------------------
# Build every part + its 3D placements
# ----------------------------------------------------------------------------------------
def build_parts(bed):
    parts = []
    # end frames
    frame, frame_eng = end_frame()
    pieces = split_end_frame(frame, bed)
    pieces.sort(key=lambda p: (round(-p.bounds[1]), p.bounds[0]))
    M0 = plane_matrix([-T, 0, 0], [0, 1, 0], [0, 0, 1])                # x in [-T, 0]
    M1 = plane_matrix([HOUSE_L, 0, 0], [0, 1, 0], [0, 0, 1])           # x in [L, L+T]
    for i, p in enumerate(pieces):
        nm = f"EF-{chr(65 + i)}" if len(pieces) > 1 else "EF"
        eng = [e for e in frame_eng if p.buffer(-0.5).contains(e)]
        pt = Part(nm, p, "YZ", qty=2, engrave=eng, note="end frame" + (" piece, glue" if len(pieces) > 1 else ""))
        pt.placements = [M0, M1]
        parts.append(pt)

    web_len = (-T - TENON_OUT, HOUSE_L + T + TENON_OUT)
    w_splits = strip_splits(*web_len, bed, -60.0)
    f_splits = strip_splits(-T, HOUSE_L + T, bed, +60.0)
    r_splits = strip_splits(*web_len, bed, 0.0)
    keep_out = w_splits + f_splits
    tab_xs = [x for x in np.arange(55.0, HOUSE_L - 54.0, WEB_TAB_PITCH)
              if all(abs(x - s) > TAB_KEEP_OUT for s in keep_out)]

    def add_strip(name, poly, plane, qty, mats, splits, eng=None, note=""):
        pcs = split_strip(poly, splits, name) if splits else [poly]
        for j, pc in enumerate(pcs):
            nm = name if len(pcs) == 1 else f"{name}{j + 1}"
            e = [g for g in (eng or []) if pc.buffer(-0.3).contains(g)]
            pt = Part(nm, pc, plane, qty=qty, engrave=e, note=note + (" (spliced, glue)" if len(pcs) > 1 else ""))
            pt.placements = mats
            parts.append(pt)

    crop_web_M = [plane_matrix([0, r + T / 2, 0], [1, 0, 0], [0, 0, 1]) for r in ROWS]   # thickness -y .. ok
    add_strip("CW", web(CROP_WEB_Z, True, tab_xs), "XZ", 3, crop_web_M, w_splits, note="crop rail web")
    anchor_web_M = [plane_matrix([0, r + T / 2, 0], [1, 0, 0], [0, 0, 1]) for r in ROWS]
    add_strip("AW", web(ANCHOR_WEB_Z, False, tab_xs), "XZ", 3, anchor_web_M, w_splits, note="anchor rail web")
    fl, fl_eng = flange(tab_xs)
    crop_fl_M = [plane_matrix([0, r, CROP_FLANGE_Z[0]], [1, 0, 0], [0, 1, 0]) for r in ROWS]
    add_strip("CF", fl, "XY", 3, crop_fl_M, f_splits, fl_eng, note="crop rail keyhole flange")
    anchor_fl_M = [plane_matrix([0, r, ANCHOR_FLANGE_Z[0]], [1, 0, 0], [0, 1, 0]) for r in ROWS]
    add_strip("AF", fl, "XY", 3, anchor_fl_M, f_splits, fl_eng, note="anchor rail keyhole flange")
    add_strip("RG", ridge_web(), "XZ", 1, [plane_matrix([0, T / 2, 0], [1, 0, 0], [0, 0, 1])], r_splits,
              note="ridge web")

    # wedges: one per tenon (7 rails x 2 ends), lying flat in the wedge hole
    wd = Part("WG", wedge(), "XY", qty=14, note="wedge, drive narrow end first")
    for zr, ys in ((CROP_WEB_Z, ROWS), (ANCHOR_WEB_Z, ROWS), (RIDGE_Z, [0.0])):
        zc = (zr[0] + zr[1]) / 2
        for y in ys:
            # long axis along y, width outward along x from the panel face, thickness z in zc +- T/2
            # left end: u=+y, v=-x -> normal +z, so start at the bottom face
            wd.placements.append(plane_matrix([-T, y - WEDGE_LEN / 2, zc - T / 2], [0, 1, 0], [-1, 0, 0]))
            # right end: u=+y, v=+x -> normal -z, so start at the top face
            wd.placements.append(plane_matrix([HOUSE_L + T, y - WEDGE_LEN / 2, zc + T / 2], [0, 1, 0], [1, 0, 0]))
    parts.append(wd)

    comb, comb_eng = test_comb()
    tc = Part("TEST", comb, "XY", qty=1, engrave=comb_eng, note="cut first: fit test")
    parts.append(tc)
    return parts, tab_xs, (w_splits, f_splits, r_splits)


# ----------------------------------------------------------------------------------------
# 3D: extrude, place, check
# ----------------------------------------------------------------------------------------
def extrude(poly, h):
    """Robust extrusion (manifold3d), closed volume guaranteed."""
    import manifold3d as mf
    import trimesh
    rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
    cs = mf.CrossSection(rings, mf.FillRule.EvenOdd)
    m = mf.Manifold.extrude(cs, h).to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def meshes(part):
    base = extrude(part.poly, T)
    out = []
    for M in part.placements:
        m = base.copy()
        m.apply_transform(M)
        if m.volume < 0:
            m.invert()
        out.append(m)
    return out


def reach_check(all_meshes):
    import trimesh
    worst = []
    for name, m in all_meshes:
        if m.bounds[1][2] <= FLOOR_Z:
            continue
        sliced = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, FLOOR_Z], cap=True, engine="earcut")
        if sliced is None or len(sliced.faces) == 0:
            continue
        for ay in ARM_YS:
            sh = np.array([[ARM_X, ay, ARM_Z]])
            _, d, _ = trimesh.proximity.closest_point(sliced, sh)
            worst.append((d[0] - REACH, name, ay))
    worst.sort()
    return worst


def overlap_check(all_meshes, tol=2.0):
    import trimesh
    bad = []
    n = len(all_meshes)
    for i in range(n):
        for j in range(i + 1, n):
            (ni, a), (nj, b) = all_meshes[i], all_meshes[j]
            if np.any(a.bounds[1] < b.bounds[0] - 0.01) or np.any(b.bounds[1] < a.bounds[0] - 0.01):
                continue
            inter = trimesh.boolean.intersection([a, b], engine="manifold")
            v = inter.volume if inter is not None and len(inter.faces) else 0.0
            if v > tol:
                bad.append((v, ni, nj))
    bad.sort(reverse=True)
    return bad


def mast_check(all_meshes):
    """Line of sight from the printed mast's default lens pose to the pick zone, through the end frame."""
    import trimesh
    scene = trimesh.util.concatenate([m for n, m in all_meshes if not n.startswith("WG")] )
    tilt = math.radians(LENS_TILT_DEG)
    axis = np.array([math.cos(tilt), 0, -math.sin(tilt)])
    results = []
    blocked = 0
    for row in (ROWS[0], ROWS[1]):          # rows served by the +Y aisle
        for x in np.linspace(*PICK_ZONE["x"], 4):
            for z in np.linspace(*PICK_ZONE["z"], 6):
                for dy in (-KEY_OFF, KEY_OFF):
                    tgt = np.array([x, row + dy, z])
                    d = tgt - LENS
                    dist = np.linalg.norm(d)
                    d = d / dist
                    # angles relative to camera axis (pan 0, tilt down)
                    h = math.degrees(math.atan2(d[1], d[0]))
                    v = math.degrees(math.asin(-d[2])) - LENS_TILT_DEG
                    locs, _, _ = scene.ray.intersects_location([LENS], [d])
                    hit = [l for l in locs if np.linalg.norm(l - LENS) < dist - 25]   # ignore the vine's own rail
                    if hit:
                        blocked += 1
                    results.append((abs(h), abs(v)))
    hmax = max(r[0] for r in results)
    vmax = max(r[1] for r in results)
    # mast base vs anything sticking out of the end frame
    mb = box(MAST_BASE["x"] - MAST_BASE["size"] / 2, MAST_BASE["y"] - MAST_BASE["size"] / 2,
             MAST_BASE["x"] + MAST_BASE["size"] / 2, MAST_BASE["y"] + MAST_BASE["size"] / 2)
    clash = []
    for n, m in all_meshes:
        fp = box(m.bounds[0][0], m.bounds[0][1], m.bounds[1][0], m.bounds[1][1])
        if m.bounds[0][2] < 60 and fp.intersects(mb):
            clash.append(n)
    return blocked, len(results), hmax, vmax, clash


def placed_meshes(parts):
    out = []
    for p in parts:
        if p.name == "TEST":
            continue
        for k, m in enumerate(meshes(p)):
            out.append((f"{p.name}#{k}", m))
    return out


# ----------------------------------------------------------------------------------------
# Previews
# ----------------------------------------------------------------------------------------
def render_assembly(all_meshes, path, views=((25, -60), (12, -8), (88, -90))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(18, 6.5))
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        for n, m in all_meshes:
            col = {"EF": "#c9a26b", "CW": "#7a9e7e", "CF": "#4f7a55", "AW": "#7a8e9e", "AF": "#4f6a7a",
                   "RG": "#9e7a7a", "WG": "#333333"}.get(n[:2], "#999999")
            pc = Poly3DCollection(m.triangles, facecolor=col, edgecolor="none", alpha=0.9)
            ax.add_collection3d(pc)
        # reach spheres (wire) + shoulders
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        for ay in ARM_YS:
            ax.plot_wireframe(ARM_X + REACH * np.cos(u) * np.sin(v), ay + REACH * np.sin(u) * np.sin(v),
                              ARM_Z + REACH * np.cos(v), color="#3a7bd5", linewidth=0.3, alpha=0.35)
        ax.scatter([LENS[0]], [LENS[1]], [LENS[2]], color="red", s=20)
        ax.set_xlim(-200, 900); ax.set_ylim(-550, 550); ax.set_zlim(0, 800)
        ax.set_box_aspect((1100, 1100, 800))
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_poly(ax, poly, color, lw=0.6):
    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xs, ys = ring.xy
            ax.plot(xs, ys, color=color, linewidth=lw)


def render_parts(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(parts)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(18, 3.2 * rows))
    for ax in axs.flat:
        ax.axis("off")
    for ax, p in zip(axs.flat, parts):
        draw_poly(ax, p.poly, "red")
        for e in p.engrave:
            xs, ys = e.xy
            ax.plot(xs, ys, color="blue", linewidth=0.4)
        ax.set_aspect("equal")
        b = p.poly.bounds
        ax.set_title(f"{p.name} x{p.qty}  {b[2]-b[0]:.0f} x {b[3]-b[1]:.0f}  ({p.note})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
def run_checks(parts, outdir):
    allm = placed_meshes(parts)
    ok = True
    print("\n== single-polygon + bed fit")
    for p in parts:
        if not isinstance(p.poly, Polygon) or not p.poly.is_valid:
            print("  ✗", p.name, "not a single valid polygon"); ok = False
    print("\n== reach (shoulder", (ARM_X, ARM_YS, ARM_Z), "radius", REACH, "; parts above z =", FLOOR_Z, ")")
    rc = reach_check(allm)
    for margin, name, ay in rc[:6]:
        print(f"  {name:10s} aisle y={ay:+.0f}: {margin:+.1f} mm outside the sphere")
    if rc and rc[0][0] < 5.0:
        print("  ✗ something above the floor is within 5 mm of the reach sphere"); ok = False
    print("\n== overlaps between parts (should be none: tenons pass through mortises)")
    ov = overlap_check(allm)
    wedge_bite = [o for o in ov if "WG" in o[1] or "WG" in o[2]]
    real = [o for o in ov if o not in wedge_bite]
    for v, a, b in real[:10]:
        print(f"  ✗ {a} ∩ {b}: {v:.1f} mm3")
    if real:
        ok = False
    else:
        print("  none")
    if wedge_bite:   # intended: the wedge is wider than the hole mid-length, that's what locks the tenon
        print(f"  (wedges bite into their tenons as designed: {len(wedge_bite)} wedges, "
              f"max {max(o[0] for o in wedge_bite):.1f} mm3 each)")
    print("\n== birdseye mast (printed, default pose) line of sight")
    blocked, n, hmax, vmax, clash = mast_check(allm)
    print(f"  {n - blocked}/{n} pick-zone points visible; widest angle off-axis {hmax:.1f}° h, {vmax:.1f}° v "
          f"(field {LENS_HFOV/2:.0f}° / {LENS_VFOV/2:.0f}° each side)")
    print(f"  mast base clashes: {clash or 'none'}")
    if blocked or clash or hmax > LENS_HFOV / 2 or vmax > LENS_VFOV / 2:
        ok = False
    outdir.mkdir(parents=True, exist_ok=True)
    render_assembly(allm, outdir / "assembly.png")
    render_parts([p for p in parts], outdir / "parts.png")
    import trimesh
    trimesh.util.concatenate([m for _, m in allm]).export(outdir / "assembly.stl")
    print("\npreviews:", outdir)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "build"])
    ap.add_argument("--bed", default="600x400")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--out", default=str(HERE / "cut_order"))
    a = ap.parse_args()
    global T
    if a.t:
        T = a.t
    bed = tuple(float(v) for v in a.bed.lower().split("x"))
    parts, tab_xs, splits = build_parts(bed)
    print(f"bed {bed[0]:.0f}x{bed[1]:.0f}, sheet t={T}: {len(parts)} unique parts, "
          f"{sum(p.qty for p in parts)} pieces; splices web {splits[0]}, flange {splits[1]}, ridge {splits[2]}")
    ok = run_checks(parts, HERE / "preview")
    print("\nCHECKS", "PASS" if ok else "FAIL")
    if a.mode == "build":
        if not ok:
            sys.exit("not building cut files: fix the failed checks first")
        import nest_and_export
        nest_and_export.export(parts, bed, pathlib.Path(a.out), T)


if __name__ == "__main__":
    main()
