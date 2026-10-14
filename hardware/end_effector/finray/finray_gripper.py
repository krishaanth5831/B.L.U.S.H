#!/usr/bin/env python3
"""SO-101 fin-ray gripper: parametric generator.

Keeps the two stock jaw parts' servo ends and swaps only the fingers:

  stub_wrist_roll_follower   stock Wrist_Roll_Follower_SO101 cut below the gripper body, plus a
                             clevis (two cheeks) that clamps the fixed TPU finger     -> PLA/PETG
  stub_moving_jaw            stock Moving_Jaw_SO101 cut below the fork, plus a clevis  -> PLA/PETG
  finger_fixed, finger_moving  TPU fin rays in the stock fingers' outline: two thin flanks that
                             meet at the tip, joined by ribs, and a solid base block that the
                             clevis clamps with 2 x M3 cross-bolts                     -> TPU 95A

Each fin ray keeps the stock finger's outline, so the jaw still closes tip-to-tip at the same servo
angle and the `follower_white` gripper calibration stays valid.

Frames: everything is built in the URDF `gripper_link` frame (mm), the moving jaw at q = 0.
The stock print STLs are in the URDF's own mesh frames (checked: ICP of the print STLs onto the
URDF meshes gave 0.01 mm RMS, det +1), so the URDF origins below place them exactly.

Usage:
  python finray_gripper.py check     build + checks + preview PNG (no STL export)
  python finray_gripper.py build     the same, then STLs in out/ (print orientation, Z up)

Watch it: python view_gripper.py   (a window with a jaw-angle slider)

Needs: numpy shapely trimesh manifold3d matplotlib
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from trimesh.transformations import euler_matrix, rotation_matrix

HERE = pathlib.Path(__file__).resolve().parent
STOCK = HERE / "stock"
OUT = HERE / "out"

# ----------------------------------------------------------------------------------------
# Stock assembly, from SO-ARM100 Simulation/SO101/so101_new_calib.urdf (commit a758567)
# ----------------------------------------------------------------------------------------
def urdf_T(xyz_m, rpy):
    M = euler_matrix(*rpy, "sxyz")
    M[:3, 3] = np.array(xyz_m) * 1000.0
    return M

# gripper_link visual of wrist_roll_follower_so101_v1 (URDF)
FIXED_VIS = urdf_T([8.32667e-17, -0.000218214, 0.000949706], [-3.14159, -5.55112e-17, 0])
# joint "gripper": gripper_link -> moving_jaw_so101_v1_link, axis z (URDF)
JOINT = urdf_T([0.0202, 0.0188, -0.0234], [1.5708, -5.24284e-08, -1.41553e-15])
# moving_jaw_so101_v1_link visual (URDF)
MOVING_VIS = urdf_T([-5.55112e-17, -5.55112e-17, 0.0189], [9.53145e-17, 6.93889e-18, 1.24077e-24])
# gripper_link visual of the gripper servo sts3215_03a_v1 (URDF)
SERVO_VIS = urdf_T([0.0077, 0.0001, -0.0234], [-1.5708, -5.19179e-17, -1.66533e-16])
Q_LIMITS_DEG = (-10.0, 100.0)   # URDF joint limits -0.1745 .. 1.7453 rad. -10 deg = stock tips touch (measured below)

def moving_T(q_deg):
    """print-STL frame of the moving jaw -> gripper_link at jaw angle q."""
    return JOINT @ rotation_matrix(math.radians(q_deg), [0, 0, 1]) @ MOVING_VIS

# ----------------------------------------------------------------------------------------
# Parameters (mm). Change here; everything else is derived.
# ----------------------------------------------------------------------------------------
W = 18.0            # finger width along the jaw axis (y). Stock finger is 28 at the root, 9 at the tip
                    # (measured); 18 is a constant-width middle so a 20-30 mm cherry tomato sits on a
                    # broad flank. guess, unverified
FLANK = 1.6         # flank thickness = 4 perimeters at 0.4 mm, so the slicer prints it as walls only
RIB = 1.2           # rib thickness = 3 perimeters. Thin enough to buckle before the flank does. unverified
RIB_PITCH = 9.0     # mm between ribs along the finger. ~6 ribs per finger, as in common fin-ray fingers. unverified
RIB_ANGLE = 10.0    # deg; ribs tilt so their back-flank end sits closer to the base. Crooks et al. 2016
                    # report better wrapping with ribs angled toward the base; value unverified
TIP_CAP = 4.0       # solid tip length, so the two flanks are glued together over a real area
BLOCK = 10.0        # TPU base block length inside the clevis (>= 10 so the clevis clamps both flanks)
CHEEK = 3.0         # clevis cheek thickness; holds a 2.5 mm deep M3 nut trap
CLR = 0.15          # gap per side between TPU block and cheeks: TPU slides in, bolts squeeze it. unverified
BOLT_D = 3.4        # M3 clearance hole
NUT_AF = 5.8        # M3 nut 5.5 across flats + 0.3 print allowance
NUT_DEPTH = 2.5     # M3 nut is 2.4 thick
BOLT_X = (0.28, 0.72)   # bolt positions across the base block, as fractions of its x span

# Finger outlines in the x-z plane of gripper_link (moving jaw at q = 0), traced from stock sections
# every 5 mm (finray_gripper.py check prints them again). Each finger = a front (contact) line and a
# back line from the base (z_fin0) to the tip; the stock finger's notches and holes are dropped.
FINGERS = {
    "fixed": dict(
        z_cut=-38.0,                 # stock gripper body ends at z -37.2; below that is finger only (measured)
        front_base=-14.0, back_base=-33.5,   # stock section at z = -48: x -33.9 .. -13.3 (measured)
        front_tip=(-8.0, -103.5), back_tip=(-13.5, -103.0),   # stock tip ends at z -104.4, x -14.5 .. -7.9 at z -100
        y_mid=-0.2,                  # centre of the stock section in y (measured)
    ),
    "moving": dict(
        z_cut=-46.0,                 # below the fork's bottom bridge (fork is y +-19.5 down to z ~ -46, measured)
        front_base=12.0, back_base=27.0,     # stock section at z = -56: x 11.9 .. 26.9 (measured)
        front_tip=(8.5, -104.5), back_tip=(13.0, -104.0),     # stock tip ends at z -105.4, x 8.1 .. 12.5 at z -105
        y_mid=-0.1,
    ),
}

# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def load_stock(name, T):
    m = trimesh.load_mesh(STOCK / name)
    if m.extents.max() < 1.0:
        m.apply_scale(1000.0)   # ⚠️ the URDF meshes (sts3215_03a_v1.stl) are in metres, the print STLs in mm
    m.apply_transform(T)
    return m

def extrude(poly, h):
    """extrude_polygon for Polygon or MultiPolygon (each piece its own body)."""
    return trimesh.util.concatenate([trimesh.creation.extrude_polygon(g, h) for g in getattr(poly, "geoms", [poly])])

def xz_to_mesh(poly, y0, y1):
    """Extrude a shapely polygon in the (x, z) plane between y = y0 and y = y1 (y0 < y1)."""
    m = extrude(poly, y1 - y0)                            # in (X=x, Y=z, Z = 0..h)
    R = np.eye(4)
    R[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]       # X->x, Z->-y, Y->z (det +1, keeps normals)
    m.apply_transform(R)
    m.apply_translation([0, y1, 0])
    return m

def line_x(p0, p1, z):
    (x0, z0), (x1, z1) = p0, p1
    return x0 + (x1 - x0) * (z - z0) / (z1 - z0)

def finger_profile(f):
    """2D (x, z) profile of the TPU finger: base block + hollow fin ray. Returns (profile, info)."""
    zc, z0 = f["z_cut"], f["z_cut"] - BLOCK
    fb, bb = (f["front_base"], z0), (f["back_base"], z0)
    ft, bt = f["front_tip"], f["back_tip"]
    x_lo, x_hi = sorted([fb[0], bb[0]])
    block = box(x_lo, z0, x_hi, zc)
    outline = Polygon([fb, ft, bt, bb]).buffer(0)
    outer = unary_union([block, outline])
    z_tip = min(ft[1], bt[1])
    # cavity: inside the flanks, below the block, above the tip cap
    cavity = outline.buffer(-FLANK, join_style=2).intersection(box(-200, z_tip + TIP_CAP, 200, z0 - FLANK * 0))
    cavity = cavity.intersection(box(-200, -400, 200, z0))
    # ribs from front flank to back flank, angled toward the base on the back side
    n = max(2, round((z0 - (z_tip + TIP_CAP)) / RIB_PITCH))
    ribs = []
    for i in range(1, n + 1):
        z = z0 - i * (z0 - (z_tip + TIP_CAP)) / (n + 1)
        xf, xb = line_x(fb, ft, z), line_x(bb, bt, z)
        dz = abs(xb - xf) * math.tan(math.radians(RIB_ANGLE))
        ribs.append(LineString([(xf, z), (xb, z + dz)]).buffer(RIB / 2, cap_style=2))
    solid = outer.difference(cavity.difference(unary_union(ribs)))
    zb = z0 + BLOCK / 2
    bolts = [Point(x_lo + t * (x_hi - x_lo), zb) for t in BOLT_X]
    solid = solid.difference(unary_union([b.buffer(BOLT_D / 2, 32) for b in bolts]))
    return solid, dict(block=block, bolts=bolts, n_ribs=n, z0=z0, x_lo=x_lo, x_hi=x_hi)

def clevis(f, info):
    """Two cheeks either side of the TPU block, with bolt holes, nut traps on the -y cheek."""
    zc, z0, y = f["z_cut"], info["z0"], f["y_mid"]
    x_lo, x_hi = info["x_lo"], info["x_hi"]
    yin = W / 2 + CLR
    cheek2d = box(x_lo, z0, x_hi, zc + 1.0)      # 1 mm up into the stub so the union is one body
    holes = unary_union([b.buffer(BOLT_D / 2, 32) for b in info["bolts"]])
    cheek2d = cheek2d.difference(holes)
    plus = xz_to_mesh(cheek2d, y + yin, y + yin + CHEEK)
    minus = xz_to_mesh(cheek2d, y - yin - CHEEK, y - yin)
    # hex nut traps from the outer face of the -y cheek
    r = NUT_AF / 2 / math.cos(math.radians(30))
    hexes = unary_union([Polygon([(b.x + r * math.cos(math.radians(60 * k + 30)), b.y + r * math.sin(math.radians(60 * k + 30)))
                                  for k in range(6)]) for b in info["bolts"]])
    trap = xz_to_mesh(hexes, y - yin - CHEEK - 1.0, y - yin - CHEEK + NUT_DEPTH)
    minus = trimesh.boolean.difference([minus, trap], engine="manifold")
    return trimesh.boolean.union([plus, minus], engine="manifold")

def stub(stock_mesh, f, cl):
    keep = trimesh.creation.box(extents=[400, 400, 400])
    keep.apply_translation([0, 0, f["z_cut"] + 200])
    top = trimesh.boolean.intersection([stock_mesh, keep], engine="manifold")
    return trimesh.boolean.union([top, cl], engine="manifold"), top

def vol(meshes_a, meshes_b):
    a = trimesh.util.concatenate(meshes_a) if isinstance(meshes_a, list) else meshes_a
    b = trimesh.util.concatenate(meshes_b) if isinstance(meshes_b, list) else meshes_b
    try:
        i = trimesh.boolean.intersection([a, b], engine="manifold")
        return 0.0 if i.is_empty else float(i.volume)
    except Exception:
        return float("nan")

# ----------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------
def build():
    stock_fixed = load_stock("Wrist_Roll_Follower_SO101.stl", FIXED_VIS)
    stock_moving = load_stock("Moving_Jaw_SO101.stl", moving_T(0.0))
    parts = {}
    for name, stock in [("fixed", stock_fixed), ("moving", stock_moving)]:
        f = FINGERS[name]
        prof, info = finger_profile(f)
        finger = xz_to_mesh(prof, f["y_mid"] - W / 2, f["y_mid"] + W / 2)
        cl = clevis(f, info)
        st, top = stub(stock, f, cl)
        parts[name] = dict(finger=finger, stub=st, top=top, clevis=cl, stock=stock, info=info, profile=prof)
    return parts

def to_print_frames(parts):
    """Stubs go back to the stock print orientation; fingers lie with the fin profile flat (Z = width)."""
    inv_fixed = np.linalg.inv(FIXED_VIS)
    inv_moving = np.linalg.inv(moving_T(0.0))
    out = {}
    s = parts["fixed"]["stub"].copy(); s.apply_transform(inv_fixed); out["stub_wrist_roll_follower_PLA"] = s
    s = parts["moving"]["stub"].copy(); s.apply_transform(inv_moving); out["stub_moving_jaw_PLA"] = s
    for name in ("fixed", "moving"):
        f = FINGERS[name]
        m = extrude(parts[name]["profile"], W)   # profile in XY, width along Z
        out[f"finger_{name}_TPU"] = m
    for m in out.values():
        m.apply_translation(-np.array([*m.bounds[0][:2] * 0, m.bounds[0][2]]))   # sit on Z = 0
    return out

# ----------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------
def checks(parts):
    ok = True
    def report(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ok   " if cond else "  FAIL ") + msg)

    print("Parts")
    for name, p in parts.items():
        for k in ("finger", "stub"):
            m = p[k]
            # count solid bodies only: the stock fork has a sealed internal void (negative-volume shell)
            nb = sum(b.volume > 0 for b in m.split(only_watertight=False))
            report(m.is_watertight and nb == 1, f"{name} {k}: watertight={m.is_watertight}, solid bodies={nb}, "
                   f"volume {m.volume / 1000:.1f} cm3, size {np.round(m.extents, 1).tolist()}")
        # everything above the cut must still be the stock part (servo interfaces untouched)
        keep = trimesh.creation.box(extents=[400, 400, 400]); keep.apply_translation([0, 0, FINGERS[name]["z_cut"] + 200])
        stock_top = trimesh.boolean.intersection([p["stock"], keep], engine="manifold")
        lost = trimesh.boolean.difference([stock_top, p["stub"]], engine="manifold")
        lost_v = 0.0 if lost.is_empty else lost.volume
        report(lost_v < 0.5, f"{name} stub keeps all of the stock part above z {FINGERS[name]['z_cut']:.0f} "
               f"(stock material lost: {lost_v:.3f} mm3)")
        removed = trimesh.boolean.difference([p["stock"], keep], engine="manifold")
        report(len(removed.split(only_watertight=False)) >= 1, f"{name}: removed stock finger {removed.volume / 1000:.1f} cm3, "
               f"z {removed.bounds[0][2]:.1f} .. {removed.bounds[1][2]:.1f}")
        print(f"       {name} finger: {p['info']['n_ribs']} ribs, base block x {p['info']['x_lo']:.1f} .. {p['info']['x_hi']:.1f}")

    print("Sweep q (moving jaw) against the fixed jaw and the servo")
    servo = load_stock("sts3215_03a_v1.stl", SERVO_VIS)
    fixed_new = [parts["fixed"]["finger"], parts["fixed"]["clevis"]]
    fixed_all = trimesh.util.concatenate([parts["fixed"]["stub"], parts["fixed"]["finger"]])
    report(vol(fixed_new, servo) < 1.0, f"fixed clevis + finger vs servo: {vol(fixed_new, servo):.2f} mm3")
    stock_fixed_tip, touch_q = None, None
    rows = []
    for q in np.arange(Q_LIMITS_DEG[0], Q_LIMITS_DEG[1] + 0.1, 2.5):
        Tq = moving_T(q) @ np.linalg.inv(moving_T(0.0))
        mv_new = []
        for k in ("finger", "clevis"):
            m = parts["moving"][k].copy(); m.apply_transform(Tq); mv_new.append(m)
        v_fix = vol(mv_new, fixed_all)
        v_servo = vol(mv_new, servo)
        rows.append((q, v_fix, v_servo))
        if v_fix > 1.0 and (touch_q is None or q > touch_q):
            touch_q = q
    for q, v_fix, v_servo in rows:
        if q <= 5 or q % 20 == 0:
            print(f"       q {q:6.1f} deg: overlap with fixed jaw {v_fix:8.1f} mm3, with servo {v_servo:6.2f} mm3")
    report(all(v < 1.0 for q, v, _ in rows if q >= 0), "no contact with the fixed jaw for q >= 0 (stock is open there too)")
    report(all(s < 1.0 for _, _, s in rows), "moving clevis + finger never hit the servo")
    mv = parts["moving"]["finger"].copy()
    mv.apply_transform(moving_T(Q_LIMITS_DEG[0]) @ np.linalg.inv(moving_T(0.0)))
    tip_gap = trimesh.proximity.closest_point(parts["fixed"]["finger"], mv.vertices)[1].min()
    report(touch_q is None and tip_gap < 3.0, f"at the closed limit q = {Q_LIMITS_DEG[0]:.0f} deg the TPU tips are "
           f"{tip_gap:.1f} mm apart (stock tips: ~1 mm, measured), so small fruit can still be pinched")

    print("Grip gap between contact flanks at mid-finger")
    zmid_f = (parts["fixed"]["info"]["z0"] + FINGERS["fixed"]["front_tip"][1]) / 2
    for q in (-10, 0, 10, 20, 30, 45):
        Tq = moving_T(q) @ np.linalg.inv(moving_T(0.0))
        mv = parts["moving"]["finger"].copy(); mv.apply_transform(Tq)
        fx = parts["fixed"]["finger"]
        s1 = fx.section(plane_origin=[0, 0, zmid_f], plane_normal=[0, 0, 1])
        s2 = mv.section(plane_origin=[0, 0, zmid_f], plane_normal=[0, 0, 1])
        if s1 is None or s2 is None:
            print(f"       q {q:4d}: finger above z {zmid_f:.0f}"); continue
        print(f"       q {q:4d} deg: gap at z {zmid_f:.0f} = {s2.vertices[:, 0].min() - s1.vertices[:, 0].max():5.1f} mm")
    return ok

def preview(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    servo = load_stock("sts3215_03a_v1.stl", SERVO_VIS)
    for ax, q in zip(axs[:3], (-10, 0, 40)):
        Tq = moving_T(q) @ np.linalg.inv(moving_T(0.0))
        items = [(servo, "0.75", 0.2), (parts["fixed"]["stub"], "tab:blue", 0.35), (parts["fixed"]["finger"], "tab:orange", 0.5)]
        for k, c, a in (("stub", "tab:blue", 0.35), ("finger", "tab:orange", 0.5)):
            m = parts["moving"][k].copy(); m.apply_transform(Tq); items.append((m, c, a))
        for m, c, a in items:
            ax.add_collection(matplotlib.collections.PolyCollection(m.vertices[m.faces][:, :, [0, 2]], lw=0, alpha=a, color=c))
        for name in ("fixed", "moving"):
            m = parts[name]["finger"].copy()
            if name == "moving": m.apply_transform(Tq)
            s = m.section(plane_origin=[0, FINGERS[name]["y_mid"], 0], plane_normal=[0, 1, 0])
            if s is not None:
                for e in s.discrete: ax.plot(e[:, 0], e[:, 2], color="saddlebrown", lw=0.8)
        ax.set_xlim(-60, 110); ax.set_ylim(-125, 25); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"jaw q = {q} deg (side view, x-z)\nblue = PLA stubs, orange = TPU fin rays")
    ax = axs[3]
    for name, c in (("fixed", "tab:orange"), ("moving", "tab:red")):
        prof = parts[name]["profile"]
        for g in getattr(prof, "geoms", [prof]):
            ax.fill(*g.exterior.xy, color=c, alpha=0.6)
            for r in g.interiors: ax.fill(*r.xy, color="white")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title("TPU finger profiles (print flat, width = Z)")
    fig.tight_layout(); fig.savefig(path, dpi=80)
    print(f"preview -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "build"])
    a = ap.parse_args()
    parts = build()
    ok = checks(parts)
    preview(parts, HERE / "preview.png")
    if a.cmd == "build":
        if not ok:
            sys.exit("checks failed, no STLs written")
        OUT.mkdir(exist_ok=True)
        for name, m in to_print_frames(parts).items():
            p = OUT / f"{name}.stl"; m.export(p)
            print(f"  {p.relative_to(HERE)}  {np.round(m.extents, 1).tolist()} mm, {m.volume / 1000:.1f} cm3")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")

if __name__ == "__main__":
    main()
