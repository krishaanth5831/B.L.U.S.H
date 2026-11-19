"""Posed SO-101 meshes as binary STLs (metres, arm base frame) for the Onshape rig assembly (2026-11-19).

Run from B.L.U.S.H/sim:
    PYTHONPATH=<dir with fast_simplification> python ../hardware/greenhouse/pipe_rail_rig/pose_so101_stl.py OUT_DIR
Writes so101_<pose>_a.stl (base, shoulder, upper arm) and so101_<pose>_b.stl (the rest): each file stays under the
Chrome extension's 10 MB upload cap.
Poses: 'start' = scene.START_NORM (the sim's tucked start pose); 'reach' = normalised [0, -50, 10, 60, 0, 60]: a grid
search for the jaw ~0.17 m in front of the base and ~0.29 m up (the upper trusses), jaw level (not a planned pose).
⚠️ Onshape only puts CLOSED meshes into an assembly (open ones import as surfaces and are skipped). So each visual
mesh gets the strongest quadric simplification that keeps it closed (tried 0.95 / 0.8 / 0.5 reduction, else the
original); the two meshes that are open even in the original (motor_holder_so101_base_v1, wrist_camera_so101_v1)
are replaced by their convex hulls. Vertex clustering (the first try) broke every part into fragments.
"""
import struct
import sys

import fast_simplification
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, '.')
import arm_map  # noqa: E402
import scene  # noqa: E402

FIRST_HALF_BODIES = {'base', 'shoulder', 'upper_arm'}   # file a; everything else goes to file b


def closed(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    return (np.unique(e, axis=0, return_counts=True)[1] == 2).all()


def hull(V):
    h = ConvexHull(V)
    F = h.simplices.copy()
    c = V.mean(0)
    for i, f in enumerate(F):                       # outward winding
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, V[f[0]] - c) < 0:
            F[i] = f[[0, 2, 1]]
    return V, F


m = mujoco.MjModel.from_xml_path(str(scene.SO101_XML))
d = mujoco.MjData(m)
fm = arm_map.FollowerMap()
poses = {
    'start': fm.norm_to_rad(np.asarray(scene.START_NORM, float)),
    'reach': fm.norm_to_rad(np.asarray([0.0, -50.0, 10.0, 60.0, 0.0, 60.0], float)),
}
simple = {}
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
        continue
    mid = m.geom_dataid[g]
    if mid in simple:
        continue
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
    pick = (V, F) if closed(F) else hull(V)
    for red in (0.95, 0.8, 0.5):
        v, f = fast_simplification.simplify(V, F, target_reduction=red)
        if closed(f):
            pick = (v, f)
            break
    simple[mid] = pick
    print(f'{m.mesh(mid).name}: {len(F)} -> {len(pick[1])}')


def write(path, T):
    with open(path, 'wb') as fh:
        fh.write(b'so101 posed'.ljust(80, b' '))
        fh.write(struct.pack('<I', len(T)))
        for t in T:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln else n
            fh.write(struct.pack('<12fH', *n, *t[0], *t[1], *t[2], 0))


out = sys.argv[1]
for name, q in poses.items():
    d.qpos[:] = 0
    d.qpos[:len(q)] = q[:m.nq]
    mujoco.mj_forward(m, d)
    halves = {'a': [], 'b': []}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        V, F = simple[m.geom_dataid[g]]
        W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        halves['a' if m.body(m.geom_bodyid[g]).name in FIRST_HALF_BODIES else 'b'].append(W[F])
    for k, tris in halves.items():
        T = np.concatenate(tris)
        write(f'{out}/so101_{name}_{k}.stl', T)
        print(name, k, len(T), 'tris')
