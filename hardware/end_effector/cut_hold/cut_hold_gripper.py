#!/usr/bin/env python3
"""SO-101 cut-and-hold jaw v6 (K7 2026-11-24): parametric generator for the truss harvest's gripper.

One servo, the stock fixed + moving jaw. Each jaw is ONE rigid PLA print: the stock part's servo end (cut below the
gripper body / the fork, as the fin-ray stubs) with a new rigid finger grown out of it. Along most of each finger:

  blade   a 9 mm snap-off cutter blade (0.38-0.40 mm thick) in a slot along the stem-side edge of the inner face,
          its edge BLADE_OUT past the face. Fixed and moving blades sit in neighbouring planes (bypass shears) and
          cross by BLADE_CROSS at the closed stop. Two M2 pinch screws across the slot hold it.
  pads    a TPU 95A face strip on two compression springs in a pocket on the rest of the inner face (the truss side),
          its face PAD_PROUD past the blade edge at rest: the two pads clamp the stalk BEFORE either blade touches it
          (post-mortem §5.7: 46 of 51 trusses lost at the cut had one pad pressing).
  flare   the last FLARE_L of each finger bends outward (a V mouth), so a stalk a little to the side is steered in.

The inner faces are parallel and FACE_GAP apart at the closed stop (q = -10 deg, the follower_white calibration's
"tips touch"), over the whole cutting zone ZONE_Z. The cutting + holding zone covers ZONE_Z instead of the v5 jaw's
16 mm tip (post-mortem: 15 of 86 misses had the stalk between the fingers but deeper than the tip blade).

Frames: the URDF gripper_link frame (mm), the moving jaw at q = 0 (as finray_gripper.py). The stock print STLs and
URDF origins are finray_gripper.py's (checked there: 0.01 mm RMS).

  python cut_hold_gripper.py check    build + checks + force budget + preview PNG
  python cut_hold_gripper.py build    the same, then print STLs (out/) and the simulator's jaw (sim/assets/cut_hold/)

Watch it: python view_cut_hold.py (a window with a jaw-angle slider).
Needs: numpy shapely trimesh manifold3d matplotlib networkx (a throwaway venv works).
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon, box

HERE = pathlib.Path(__file__).resolve().parent
FINRAY = HERE.parent / "finray"
sys.path.insert(0, str(FINRAY))
import finray_gripper as fg  # noqa: E402  (stock frames, stock STLs, helpers)

OUT = HERE / "out"
SIM_OUT = HERE.parents[2] / "sim" / "assets" / "cut_hold"

# ----------------------------------------------------------------------------------------
# Parameters (mm). Everything else is derived. Values marked guess are to be checked on the bench.
# ----------------------------------------------------------------------------------------
Q_CLOSED = -10.0      # deg: the closed stop (scene.GRIPPER_RANGE_RAD[0], follower_white calibration)
W = 18.0              # finger width along gripper y (the stalk's direction across the jaw), as the v5 sim jaw
T_FINGER = 10.0       # finger thickness behind the inner face (PLA, rigid). guess
ZONE_Z = (-62.0, -98.0)   # cutting + holding zone along the finger (gripper z, at the closed stop). The top stays
                      # 15 mm below the fixed jaw's body (z -37) so the stem is never pinched by the palm
FLARE_L = 12.0        # length of the outward-bent tip below the zone
FLARE_OUT = 6.0       # how far each tip bends outward over FLARE_L (27 deg): mouth 12 mm wider than the zone. guess
FACE_GAP = 4.0        # inner faces this far apart at the closed stop = 2 x BLADE_OUT - BLADE_CROSS
BLADE_OUT = 3.0       # blade edge past the inner face
BLADE_CROSS = 2.0     # the two blade edges overlap this much at the closed stop (bypass)
BLADE_T = 0.40        # 9 mm snap-off blade thickness (0.38-0.40 typical)
BLADE_H = 9.0         # blade height (the snap-off blade's width): BLADE_H - BLADE_OUT sits in the slot
SLOT_CLR = 0.10       # slot clearance per side; the pinch screws close it
# Blade planes (gripper y): fixed then moving, neighbouring (bypass). The stem side is -y (truss.py: the stalk's
# stem end is on the blade side). SIM_BLADE_Y are the simulator's 2 mm boxes centred on them (contact robustness, as
# the v5 sim jaw's BLADE_Y).
BLADE_Y = {"fixed": (-7.45, -7.05), "moving": (-6.95, -6.55)}
SIM_BLADE_Y = {"fixed": (-9.0, -7.0), "moving": (-7.0, -5.0)}
PAD_Y = (-4.5, 9.0)   # pad across the rest of the finger width, on the truss side of the blades (as v5)
PAD_PROUD = 3.0       # pad face past the blade edge at rest -> both pads touch the stalk 2 x 3 mm of jaw travel
                      # before the blades do (v5: 1 mm)
PAD_FACE_T = 2.0      # TPU face strip
PAD_TRAVEL = 5.5      # pad can be pushed back this far (at the closed stop the faces overlap 2 x 4 = 8 mm)
PAD_POCKET = PAD_TRAVEL + PAD_FACE_T - (BLADE_OUT + PAD_PROUD - PAD_FACE_T) + 1.0   # pocket depth into the face
PAD_PRELOAD_N = 1.5   # spring force at rest (both springs together). guess: the stalk can't push the pads back
PAD_K_N_PER_MM = 0.5  # both springs together: 1.5 N at rest, 4.25 N at full travel. Two springs ~4 x 10 mm,
                      # 0.25 N/mm each (common hardware assortment), guess
M2_D = 2.2            # pinch screw clearance
SPRING_D = 4.4        # spring pocket diameter (4 mm OD springs)
GRIPPER_TORQUE_NM = 0.5 * 3.35   # scene.GRIPPER_TORQUE: lerobot sets Max_Torque_Limit 50 % of the MJCF's 3.35 N.m
OVERLOAD_TORQUE_NM = 0.25 * 3.35  # what the servo drops to once stalled long enough (lerobot Overload_Torque 25 %)
CUT_FORCE_PER_MM = 15.0 / 7.0    # truss.CUT_FORCE_PER_MM (SHEAR placeholder, unverified for tomato peduncles)
STALK_D = (3.0, 5.0)             # truss.PEDUNCLE_R's range x 2

FIX = fg.FINGERS["fixed"]
MOV = fg.FINGERS["moving"]


def to_q0(points_closed):
    """Points (x, z) given with the moving jaw at the closed stop -> the same jaw points at q = 0."""
    T = fg.moving_T(0.0) @ np.linalg.inv(fg.moving_T(Q_CLOSED))
    P = np.array([[x, 0.0, z, 1.0] for x, z in points_closed]).T
    Q = T @ P
    return [(float(a), float(c)) for a, c in zip(Q[0], Q[2])]


def inner_x():
    """The inner faces at the closed stop: fixed at x_f, moving at x_f + FACE_GAP, both parallel to gripper z.
    x_f is the stock fixed finger's front line at the zone's middle, so the jaw stays where the stock one closed."""
    zm = np.mean(ZONE_Z)
    x_f = fg.line_x((FIX["front_base"], FIX["z_cut"] - 10.0), FIX["front_tip"], zm) - 1.0
    return x_f, x_f + FACE_GAP


def finger_outlines():
    """Closed-stop outlines (x, z) of each finger, as two convex pieces: 'main' (from the stub's cut face down to the
    zone's bottom) and 'flare' (the outward-bent tip). The moving finger is returned at q = 0."""
    x_f, x_m = inner_x()
    z_top, z_bot = ZONE_Z
    z_tip = z_bot - FLARE_L
    # fixed: inner face at x_f (faces +x), back at x_f - T; widens to the stock section at the cut face
    fx = dict(
        main=[(FIX["back_base"], FIX["z_cut"] + 1.0), (FIX["front_base"], FIX["z_cut"] + 1.0), (x_f, z_top),
              (x_f, z_bot), (x_f - T_FINGER, z_bot), (x_f - T_FINGER, z_top - 8.0)],
        flare=[(x_f, z_bot), (x_f - FLARE_OUT, z_tip), (x_f - FLARE_OUT - T_FINGER * 0.6, z_tip),
               (x_f - T_FINGER, z_bot)])
    # moving at the closed stop: inner face at x_m (faces -x); its base must meet the stock fork's section, so the
    # base points are mapped from q = 0 into the closed pose first
    T_inv = fg.moving_T(Q_CLOSED) @ np.linalg.inv(fg.moving_T(0.0))
    base = []
    for x, z in ((MOV["front_base"], MOV["z_cut"] + 1.0), (MOV["back_base"], MOV["z_cut"] + 1.0)):
        p = T_inv @ np.array([x, 0.0, z, 1.0])
        base.append((float(p[0]), float(p[2])))
    mv_closed = dict(
        main=[base[0], base[1], (x_m + T_FINGER, z_top - 8.0), (x_m + T_FINGER, z_bot), (x_m, z_bot), (x_m, z_top)],
        flare=[(x_m, z_bot), (x_m + T_FINGER, z_bot), (x_m + FLARE_OUT + T_FINGER * 0.6, z_tip),
               (x_m + FLARE_OUT, z_tip)])
    mv = {k: to_q0(v) for k, v in mv_closed.items()}
    return fx, mv, mv_closed


def solid(points, y0, y1):
    poly = Polygon(points).convex_hull           # each piece is convex by design (MuJoCo collides convex hulls)
    return fg.xz_to_mesh(poly, y0, y1), poly


def zone_box(part, x_face, inward, y0, y1, depth, out=0.0, z=ZONE_Z):
    """A box along the zone at the closed stop: from `depth` inside the inner face to `out` past it (fixed frame)."""
    s = 1.0 if inward == "-x" else -1.0          # fixed finger's inside is -x; moving's is +x
    xa, xb = x_face - s * depth, x_face + s * out
    lo, hi = min(xa, xb), max(xa, xb)
    m = trimesh.creation.box(bounds=[[lo, y0, z[1]], [hi, y1, z[0]]])
    return m


def build():
    fx, mv, mv_closed = finger_outlines()
    x_f, x_m = inner_x()
    y_mid = {"fixed": FIX["y_mid"], "moving": MOV["y_mid"]}
    stock = {"fixed": fg.load_stock("Wrist_Roll_Follower_SO101.stl", fg.FIXED_VIS),
             "moving": fg.load_stock("Moving_Jaw_SO101.stl", fg.moving_T(0.0))}
    T_open = fg.moving_T(0.0) @ np.linalg.inv(fg.moving_T(Q_CLOSED))   # closed pose -> q = 0, for the moving cuts
    parts = {}
    for name, outl, face, inward in (("fixed", fx, x_f, "-x"), ("moving", mv_closed, x_m, "+x")):
        y0, y1 = y_mid[name] - W / 2, y_mid[name] + W / 2
        pieces = {k: solid(v, y0, y1) for k, v in outl.items()}
        finger = trimesh.boolean.union([pieces["main"][0], pieces["flare"][0]], engine="manifold")
        # cuts, made at the closed stop in the fixed frame: blade slot and pad pocket along the zone
        by = BLADE_Y[name]
        slot = zone_box(name, face, inward, by[0] - SLOT_CLR, by[1] + SLOT_CLR, BLADE_H - BLADE_OUT + 0.3, out=1.0)
        pocket = zone_box(name, face, inward, PAD_Y[0], PAD_Y[1], PAD_POCKET, out=1.0,
                          z=(ZONE_Z[0] - 1.0, ZONE_Z[1] + 1.0))
        cuts = [slot, pocket]
        # 2 x M2 pinch screws across the blade slot (along y), 2 spring pockets behind the pad (along x)
        zs = np.linspace(ZONE_Z[0], ZONE_Z[1], 4)[1:3]
        s = 1.0 if inward == "-x" else -1.0
        for z in zs:
            hole = trimesh.creation.cylinder(radius=M2_D / 2, height=W + 2, sections=24)
            hole.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            hole.apply_translation([face - s * (BLADE_H - BLADE_OUT) / 2, y_mid[name], z])
            cuts.append(hole)
            spring = trimesh.creation.cylinder(radius=SPRING_D / 2, height=8.0, sections=24)
            spring.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            spring.apply_translation([face - s * (PAD_POCKET + 4.0), np.mean(PAD_Y), z])
            cuts.append(spring)
        finger = trimesh.boolean.difference([finger] + cuts, engine="manifold")
        blade = zone_box(name, face, inward, by[0], by[1], BLADE_H - BLADE_OUT, out=BLADE_OUT,
                         z=(ZONE_Z[0] + 1.0, ZONE_Z[1] - 1.0))
        rest = BLADE_OUT + PAD_PROUD                                  # pad face past the inner face at rest
        pad = zone_box(name, face, inward, PAD_Y[0] + 0.3, PAD_Y[1] - 0.3, PAD_FACE_T - rest, out=rest)
        items = dict(finger=finger, blade=blade, pad=pad, main=pieces["main"][0], flare=pieces["flare"][0])
        if name == "moving":                                          # back to the q = 0 build frame
            for k in items:
                items[k] = items[k].copy()
                items[k].apply_transform(T_open)
        keep = trimesh.creation.box(extents=[400, 400, 400])
        keep.apply_translation([0, 0, fg.FINGERS[name]["z_cut"] + 200])
        top = trimesh.boolean.intersection([stock[name], keep], engine="manifold")
        jaw = trimesh.boolean.union([top, items["finger"]], engine="manifold")
        parts[name] = dict(jaw=jaw, top=top, stock=stock[name], **items)
    return parts


def force_budget():
    """Moving-finger force available along the zone vs what clamping + cutting needs."""
    P = fg.moving_T(Q_CLOSED)[:3, 3]                     # the joint axis passes through the moving link's origin
    x_f, x_m = inner_x()
    rows = []
    need = [CUT_FORCE_PER_MM * d + PAD_PRELOAD_N + PAD_K_N_PER_MM * (PAD_PROUD + BLADE_OUT - BLADE_CROSS / 2)
            for d in STALK_D]
    for z in np.linspace(ZONE_Z[0], ZONE_Z[1], 5):
        r = math.hypot(x_m - P[0], z - P[2]) / 1000.0
        rows.append((z, r * 1000, GRIPPER_TORQUE_NM / r, OVERLOAD_TORQUE_NM / r))
    return rows, need


def checks(parts):
    ok = True

    def report(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ok   " if cond else "  FAIL ") + msg)

    print("Parts")
    for name in ("fixed", "moving"):
        p = parts[name]
        nb = sum(b.volume > 0 for b in p["jaw"].split(only_watertight=False))
        report(p["jaw"].is_watertight and nb == 1, f"{name} jaw: watertight={p['jaw'].is_watertight}, solid bodies="
               f"{nb}, {p['jaw'].volume / 1000:.1f} cm3, size {np.round(p['jaw'].extents, 1).tolist()}")
        keep = trimesh.creation.box(extents=[400, 400, 400])
        keep.apply_translation([0, 0, fg.FINGERS[name]["z_cut"] + 200])
        lost = trimesh.boolean.difference([p["top"], p["jaw"]], engine="manifold")
        lost_v = 0.0 if lost.is_empty else lost.volume
        report(lost_v < 0.5, f"{name}: the stock servo end above z {fg.FINGERS[name]['z_cut']:.0f} is untouched "
               f"(lost {lost_v:.3f} mm3)")
    print("Sweep the moving jaw")
    servo = fg.load_stock("sts3215_03a_v1.stl", fg.SERVO_VIS)
    fixed_all = trimesh.util.concatenate([parts["fixed"]["jaw"]])
    rows = []
    for q in np.arange(Q_CLOSED, fg.Q_LIMITS_DEG[1] + 0.1, 2.5):
        Tq = fg.moving_T(q) @ np.linalg.inv(fg.moving_T(0.0))
        mv = parts["moving"]["finger"].copy()
        mv.apply_transform(Tq)
        rows.append((q, fg.vol(mv, fixed_all), fg.vol(mv, servo)))
    report(all(v < 1.0 for _, v, _ in rows), "the rigid fingers never touch each other, closed stop included: "
           f"worst overlap {max(v for _, v, _ in rows):.2f} mm3")
    report(all(s < 1.0 for _, _, s in rows), "the moving finger never hits the servo")
    bl = {}
    for name in ("fixed", "moving"):
        b = parts[name]["blade"].copy()
        if name == "moving":
            b.apply_transform(fg.moving_T(Q_CLOSED) @ np.linalg.inv(fg.moving_T(0.0)))
        bl[name] = b
    cross = min(bl["moving"].bounds[1][0], bl["fixed"].bounds[1][0]) - max(bl["moving"].bounds[0][0], 0) * 0
    x_f, x_m = inner_x()
    edge_f, edge_m = x_f + BLADE_OUT, x_m - BLADE_OUT
    report(abs((edge_f - edge_m) - BLADE_CROSS) < 1e-6, f"the blade edges cross by {edge_f - edge_m:.1f} mm at the "
           f"closed stop (bypass, in neighbouring planes y {BLADE_Y['fixed']} / {BLADE_Y['moving']})")
    report(BLADE_Y["fixed"][1] <= BLADE_Y["moving"][0], "the blades sit in separate planes (they slide past each "
           "other)")
    print("Jaw opening at the zone's middle (inner faces; blades 3 mm, pads 6 mm in from them)")
    zm = np.mean(ZONE_Z)
    for q in (Q_CLOSED, 0, 10, 20, 30):
        Tq = fg.moving_T(q) @ np.linalg.inv(fg.moving_T(Q_CLOSED))
        p = Tq @ np.array([x_m, 0, zm, 1.0])
        gap = p[0] - x_f
        print(f"       q {q:5.1f} deg: faces {gap:5.1f} mm apart, blade edges {gap - 2 * BLADE_OUT:5.1f}, "
              f"pad faces {gap - 2 * (BLADE_OUT + PAD_PROUD):5.1f}")
    print("Force budget (moving finger, closed stop)")
    rows, need = force_budget()
    for z, r, f, fo in rows:
        print(f"       z {z:6.1f}: {r:5.1f} mm from the hinge -> {f:5.1f} N at 50 % torque ({fo:4.1f} N if the servo "
              f"has dropped to its 25 % overload limit)")
    print(f"       needed at the cut: {need[0]:.1f} N (3 mm stalk) .. {need[1]:.1f} N (5 mm), cut force "
          f"{CUT_FORCE_PER_MM:.2f} N/mm (unverified) + both springs")
    report(min(f for _, _, f, _ in rows) >= need[1], "50 % torque cuts a 5 mm stalk anywhere in the zone")
    return ok


def to_sim(parts):
    """The simulator's jaw: convex collision pieces (STL, body frames, metres) + blade/pad boxes (jaw.json)."""
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    to_link = {"fixed": np.eye(4), "moving": np.linalg.inv(fg.JOINT)}        # gripper_link -> body frame (q = 0)
    spec = {}
    x_f, x_m = inner_x()
    T_close = fg.moving_T(Q_CLOSED) @ np.linalg.inv(fg.moving_T(0.0))
    for name in ("fixed", "moving"):
        p = parts[name]
        S = np.diag([1e-3, 1e-3, 1e-3, 1.0]) @ to_link[name]
        for k in ("main", "flare"):
            m = p[k].copy()
            m.apply_transform(S)
            m.export(SIM_OUT / f"finger_{name}_{k}.stl")
        top = p["top"].copy()
        top.apply_transform(S)
        top.export(SIM_OUT / f"stub_{name}.stl")
        # blade / pad boxes, closed-stop geometry described in the gripper frame, then into the body frame at q = 0
        face, s_in = (x_f, 1.0) if name == "fixed" else (x_m, -1.0)   # s_in: +x points at the other finger for fixed
        zc, hz = np.mean(ZONE_Z), abs(ZONE_Z[0] - ZONE_Z[1]) / 2
        R_back = np.eye(4) if name == "fixed" else np.linalg.inv(T_close)

        def placed(c):
            q = to_link[name] @ R_back @ np.array([*c, 1.0])
            return (q[:3] / 1000.0).tolist()
        Rb = (to_link[name] @ R_back)[:3, :3]
        spec[name] = dict(
            R=Rb.tolist(),                          # box axes (gripper x, y, z at the closed stop) in the body frame
            s_in=s_in, face_m=placed([face, 0, zc]),
            blade=dict(centre=placed([face + s_in * BLADE_OUT / 2, np.mean(SIM_BLADE_Y[name]), zc]),
                       half=[BLADE_OUT / 2e3, (SIM_BLADE_Y[name][1] - SIM_BLADE_Y[name][0]) / 2e3, (hz - 1.0) / 1e3]),
            pad=dict(centre=placed([face, np.mean(PAD_Y), zc]), rest_m=(BLADE_OUT + PAD_PROUD) / 1e3,
                     half=[PAD_FACE_T / 2e3, (PAD_Y[1] - PAD_Y[0] - 0.6) / 2e3, hz / 1e3],
                     travel_m=PAD_TRAVEL / 1e3, preload_n=PAD_PRELOAD_N, k_n_per_m=PAD_K_N_PER_MM * 1e3))
    spec["params"] = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple, dict))
                      and k not in ("FIX", "MOV")}
    (SIM_OUT / "jaw.json").write_text(json.dumps(spec, indent=1, default=list))
    print(f"simulator jaw -> {SIM_OUT}")


def to_print(parts):
    OUT.mkdir(exist_ok=True)
    inv = {"fixed": np.linalg.inv(fg.FIXED_VIS), "moving": np.linalg.inv(fg.moving_T(0.0))}
    names = {"fixed": "jaw_fixed_cut_hold_PLA", "moving": "jaw_moving_cut_hold_PLA"}
    for name in ("fixed", "moving"):
        m = parts[name]["jaw"].copy()
        m.apply_transform(inv[name])                 # the stock print orientation (as the fin-ray stubs)
        p = OUT / f"{names[name]}.stl"
        m.export(p)
        print(f"  {p.relative_to(HERE)}  {np.round(m.extents, 1).tolist()} mm, {m.volume / 1000:.1f} cm3")
    zl = abs(ZONE_Z[0] - ZONE_Z[1])
    pad = trimesh.creation.box(extents=[zl, PAD_Y[1] - PAD_Y[0] - 0.6, PAD_FACE_T])
    pad.apply_translation(-pad.bounds[0])
    pad.export(OUT / "pad_face_TPU_x2.stl")
    print(f"  out/pad_face_TPU_x2.stl  {np.round(pad.extents, 1).tolist()} mm")


def preview(parts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(22, 7))
    servo = fg.load_stock("sts3215_03a_v1.stl", fg.SERVO_VIS)
    for ax, q in zip(axs[:3], (Q_CLOSED, 0.0, 25.0)):
        Tq = fg.moving_T(q) @ np.linalg.inv(fg.moving_T(0.0))
        items = [(servo, "0.75", 0.25), (parts["fixed"]["jaw"], "tab:blue", 0.45),
                 (parts["fixed"]["blade"], "0.2", 0.9), (parts["fixed"]["pad"], "tab:green", 0.7)]
        for k, c, a in (("jaw", "tab:blue", 0.45), ("blade", "0.2", 0.9), ("pad", "tab:green", 0.7)):
            m = parts["moving"][k].copy()
            m.apply_transform(Tq)
            items.append((m, c, a))
        for m, c, a in items:
            ax.add_collection(matplotlib.collections.PolyCollection(m.vertices[m.faces][:, :, [0, 2]], lw=0,
                                                                    alpha=a, color=c))
        ax.set_xlim(-50, 70)
        ax.set_ylim(-125, 20)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"jaw q = {q:.0f} deg (side view)\nblue PLA jaws, dark = blades, green = pads at rest")
    ax = axs[3]                                       # cross-section at the zone's middle, closed stop
    zm = np.mean(ZONE_Z)
    Tc = fg.moving_T(Q_CLOSED) @ np.linalg.inv(fg.moving_T(0.0))
    for name, c in (("fixed", "tab:blue"), ("moving", "tab:blue")):
        for k, col in (("finger", c), ("blade", "0.2"), ("pad", "tab:green")):
            m = parts[name][k].copy()
            if name == "moving":
                m.apply_transform(Tc)
            s = m.section(plane_origin=[0, 0, zm], plane_normal=[0, 0, 1])
            if s is None:
                continue
            for e in s.discrete:
                ax.fill(e[:, 0], e[:, 1], color=col, alpha=0.6)
    ax.add_patch(matplotlib.patches.Circle((np.mean(inner_x()), 0), 2.0, color="tab:olive"))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"cross-section at z {zm:.0f}, closed stop (x across the jaw, y along the stalk)\n"
                 "pads overlap here = their spring travel; olive = a 4 mm stalk")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
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
            sys.exit("checks failed, nothing written")
        to_print(parts)
        to_sim(parts)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
