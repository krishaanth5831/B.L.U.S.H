#!/usr/bin/env python3
"""Export the fin-ray gripper parts as MuJoCo meshes (metres, in the MJCF body frames).

  python build_finray_meshes.py      writes sim/assets/finray/*.stl + chain.json

Runs hardware/end_effector/finray/finray_gripper.py's build(), so the sim fingers are the exact
geometry that gets printed. Frames:
  stub_fixed, finger_fixed    body "gripper"             (= URDF gripper_link)
  stub_moving, finger_moving  body "moving_jaw_so101_v1" (= URDF moving_jaw_so101_v1_link)
Checked: the MJCF body poses equal the URDF origins finray_gripper.py uses (FIXED_VIS = geom quat
0 1 0 0, JOINT = body pos 0.0202 0.0188 -0.0234 quat 0.7071 0.7071 0 0, MOVING_VIS = pos 0 0 0.0189).

The bending fingers (scene.py FINRAY_BENDING) are built in MuJoCo from plates, so this only adds:
  finger_<side>_base.stl   the TPU block the clevis clamps (the finger cut at the block's bottom face),
                           rigid with the jaw
  chain.json               per finger: the width axis, the "ladder" (flank and rib nodes on the plate
                           centrelines, tip cap box, plate thicknesses, TPU density) in the jaw frame,
                           and the contact flank line in gripper_link mm (check_grasp.py places fruit by it)

Needs: numpy shapely trimesh manifold3d (a throwaway venv works, like the finray generator).
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardware" / "end_effector" / "finray"))
import finray_gripper as fg   # noqa: E402

OUT = HERE / "assets" / "finray"
TPU_DENSITY = 1210.0       # kg/m3, TPU 95A datasheet range 1.20-1.23 g/cm3


def to_manifold(mesh):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                               tri_verts=np.asarray(mesh.faces, np.uint32)))


def to_trimesh(man):
    import trimesh
    mm = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def split(man, point, normal):
    """(part on the normal's side, part on the other side) of a plane through `point`."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    return man.split_by_plane(tuple(n), float(n @ point))


def ladder_geometry(name):
    """The fin ray as a ladder of plates (gripper_link mm, moving jaw at q = 0): flank nodes where each
    rib meets the front and back flank centrelines, the root nodes on the block's bottom face, and the
    nodes where both flanks enter the solid tip cap. Same rib layout as finray_gripper.finger_profile."""
    f = fg.FINGERS[name]
    z0 = f["z_cut"] - fg.BLOCK
    fb, bb = np.array([f["front_base"], z0], float), np.array([f["back_base"], z0], float)
    ft, bt = np.array(f["front_tip"], float), np.array(f["back_tip"], float)
    z_tip = min(ft[1], bt[1])
    z_cap = z_tip + fg.TIP_CAP
    n = max(2, round((z0 - z_cap) / fg.RIB_PITCH))
    zr = [z0 - i * (z0 - z_cap) / (n + 1) for i in range(1, n + 1)]

    def unit(v):
        return v / np.linalg.norm(v)

    def inward(p0, p1, toward):
        d = unit(p1 - p0)
        nrm = np.array([-d[1], d[0]])
        return nrm if (toward - p0) @ nrm > 0 else -nrm

    def at_z(line, z):
        p0, p1 = line
        return p0 + (z - p0[1]) / (p1[1] - p0[1]) * (p1 - p0)

    def meet(line, p, d):
        p0, p1 = line
        s, _ = np.linalg.solve(np.c_[p1 - p0, -d], p - p0)
        return p0 + s * (p1 - p0)

    off_f = inward(fb, ft, bb) * fg.FLANK / 2           # outer surface -> flank centreline
    off_b = inward(bb, bt, fb) * fg.FLANK / 2
    front_c, back_c = (fb + off_f, ft + off_f), (bb + off_b, bt + off_b)
    back_side = np.sign(bb[0] - fb[0])
    rib = unit(np.array([back_side, math.tan(math.radians(fg.RIB_ANGLE))]))
    F, B = [at_z(front_c, z0)], [at_z(back_c, z0)]
    for z in zr:
        start = np.array([fg.line_x(fb, ft, z), z])     # a rib starts on the front outer surface
        F.append(meet(front_c, start, rib))
        B.append(meet(back_c, start, rib))
    cap = np.array([at_z((fb, ft), z_cap), ft, bt, at_z((bb, bt), z_cap)])
    return dict(F=F, B=B, F_cap=at_z(front_c, z_cap), B_cap=at_z(back_c, z_cap),
                cap_lo=cap.min(0), cap_hi=cap.max(0), z_tip=z_tip, y_mid=f["y_mid"], n=n)


def main():
    import trimesh
    OUT.mkdir(parents=True, exist_ok=True)
    parts = fg.build()
    to_jaw = np.linalg.inv(fg.JOINT)          # gripper_link (q = 0) -> moving jaw link
    for name, part, T in (("stub_fixed", parts["fixed"]["stub"], np.eye(4)),
                          ("finger_fixed", parts["fixed"]["finger"], np.eye(4)),
                          ("stub_moving", parts["moving"]["stub"], to_jaw),
                          ("finger_moving", parts["moving"]["finger"], to_jaw)):
        m = part.copy()
        m.apply_transform(T)
        m.apply_scale(0.001)                  # ⚠️ finray_gripper works in mm, MuJoCo in metres
        m.export(OUT / f"{name}.stl")
        lo, hi = m.bounds
        print(f"{name:14s} {len(m.faces):6d} faces  bounds mm {np.round(lo*1000,1)} .. {np.round(hi*1000,1)}")

    chain = {}
    for side, T in (("fixed", np.eye(4)), ("moving", to_jaw)):
        R, t = T[:3, :3], T[:3, 3]
        to_body = lambda p: (R @ np.asarray(p, float) + t) / 1000.0   # noqa: E731  mm gripper frame -> m jaw frame
        f = fg.FINGERS[side]
        z0 = f["z_cut"] - fg.BLOCK
        base, fin = split(to_manifold(parts[side]["finger"]), np.array([0.0, 0.0, z0]), np.array([0.0, 0.0, 1.0]))
        assert abs(base.volume() + fin.volume() - parts[side]["finger"].volume) < 1.0   # mm3: the cut loses nothing
        m = to_trimesh(base)
        m.apply_transform(T)
        m.apply_scale(0.001)
        m.export(OUT / f"finger_{side}_base.stl")
        lad = ladder_geometry(side)
        p3 = lambda p: to_body([p[0], lad["y_mid"], p[1]]).tolist()   # noqa: E731
        lo = np.array([lad["cap_lo"][0], lad["y_mid"] - fg.W / 2, lad["cap_lo"][1]])
        hi = np.array([lad["cap_hi"][0], lad["y_mid"] + fg.W / 2, lad["cap_hi"][1]])
        ladder = dict(F=[p3(p) for p in lad["F"]], B=[p3(p) for p in lad["B"]], F_cap=p3(lad["F_cap"]),
                      B_cap=p3(lad["B_cap"]), cap=dict(centre=to_body((lo + hi) / 2).tolist(),
                                                       half=((hi - lo) / 2000).tolist(), R=R.tolist()),
                      flank_t=fg.FLANK / 1000, rib_t=fg.RIB / 1000, width=fg.W / 1000, density=TPU_DENSITY)
        chain[side] = dict(base=f"finger_{side}_base.stl", axis=(R @ np.array([0.0, 1.0, 0.0])).tolist(), ladder=ladder,
                           front_line_gripper_mm=[[f["front_base"], z0], list(f["front_tip"])], y_mid_mm=f["y_mid"])
        print(f"ladder {side}: {lad['n']} ribs, block {base.volume() / 1000:.2f} cm3, fin ray {fin.volume() / 1000:.2f} cm3")
    (OUT / "chain.json").write_text(json.dumps(chain, indent=1))
    print("chain.json written")


if __name__ == "__main__":
    main()
