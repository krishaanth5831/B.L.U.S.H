#!/usr/bin/env python3
"""Where should the fixed cameras stand? A score per candidate position, from built scenes only (no plan is played,
nothing is recorded). K7 2026-11-26: "3 cameras instead of 2 but also placed in a more calculated location and
position"; the criteria are the post-mortem's (vault note 5.5 and step 5: every cut point in view, at least 1 pixel
per mm, two fixed views whose lines of sight cross at 45 deg or more).

    python whatif/camera_score.py collect --seeds 9730000:9730090 --old-start --out ../outputs/v6/cams/old_0.npz
    python whatif/camera_score.py rank ../outputs/v6/cams/old_*.npz [--json out.json]
    python whatif/camera_score.py selftest --seeds 9730000:9730010 --old-start
    python whatif/camera_score.py render --seeds 9730000,9730003 --old-start --cams base,g_0.02_0.10_0.28 --out x.png

collect, per scene with a target and a cut pose (TrussTeacher._cut_geom at u 0.5, the first approach blend with IK;
no collision check, so nearly every target counts): the arm is put at three poses in the teacher's planning copy of
the scene, START (the scene's first frame), PRE (the pre-cut pose, jaw open, SLIDE_M before the stalk) and CUT (jaw
open around the stalk), and every candidate camera is asked:
    in      the cut point projects inside the 640 x 480 frame (IN_MARGIN_PX from the edge), through lens.py's model
    vis     a ray from the lens to the cut point arrives (nothing nearer than VIS_TOL_M before it)
    ppm     pixels per mm at the cut point
    blade   a ray to one of the blades arrives (PRE and CUT)
    stalk   CUT: a ray to the stalk within STALK_NEAR_M above or below the cut point arrives
    clear   the lens' distance to the arm's bounding spheres at those poses and at the teacher's via-poses
Candidates: a grid of lens positions a post from the table could hold (GX x GY x GZ), all aimed at
scene.BASE_CAM_AIM (the middle of the cut points' band), plus today's two cameras, "base" and "birdseye", as they
are. ⚠️ The lens is lens.py's guess (InnoMaker spec, uncalibrated): 72 x 52 deg.
The claw camera is scored too (does it see the cut point at START and at PRE?).

rank: per camera and per pair of fixed cameras over all scenes of the files given.
selftest: the projection and the ray test against the model's own "base" camera and its segmentation render.
lerobot-mj4, outside the sandbox.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sim/
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402

import scene  # noqa: E402

GX = np.round(np.arange(0.02, 0.73, 0.05), 3)      # along the aisle, between the end frames (HOUSE_L 0.74)
GY = np.round(np.arange(-0.20, 0.33, 0.05), 3)     # from behind row R0 (the birdseye's mast stands at -0.15) to 7 cm
#                                                    before the vine row's stems (y 0.385)
GZ = (0.05, 0.10, 0.16, 0.22, 0.28, 0.34, 0.42, 0.50)   # lens height; the crop rail is at 0.52
IN_MARGIN_PX = 20          # "inside the frame" leaves this much to the edge (guess: room for the target box)
VIS_TOL_M = 0.005          # a ray "arrives" when nothing is hit more than this before the point (stalk radius + 2-3 mm)
BLADE_TOL_M = 0.003        # ... for a blade's centre (the blade is 1 mm thick; its own face is hit first)
STALK_NEAR_M = (-0.012, -0.006, 0.006, 0.012)   # CUT: points along the stalk either side of the cut point (guess)
STALK_AT_M = (-0.004, 0.0, 0.004)   # START / PRE: "the cut point" = the stalk here; one ray alone called a stalk hidden
#                                     that the segmentation render showed 3 px from the cut point (selftest, 9730001)
GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)   # what a MuJoCo camera draws by default (geom groups 0-2)
PPM_MIN = 1.0              # post-mortem step 5: at least 1 pixel per mm
CROSS_DEG = 45.0           # post-mortem 5.5: lines of sight that cross at 45 deg or more place a point in 3D
CLEAR_MIN_M = 0.04         # rank: a lens or its post nearer than this to the arm's or the plants' bounding spheres
#                            is out (guess; the spheres overestimate both)
Z_MAX_M = 0.34             # rank: no lens above the truss band (KNUCKLE_Z tops at 0.32). ⚠️ The sim has no leaves:
#                            a real crop is only stripped at and below the ripening trusses, so a view down through
#                            the head of the plant would score here and see leaves in the greenhouse.
CARRY_X_M = 0.40           # rank: no post at x beyond this on the aisle side (y >= 0): the cut truss is carried from
#                            the row to the crate there (crate x 0.43-0.73), poses this score does not visit


def setup(base, jaw, old_start):
    import base_grid
    b = [float(v) for v in base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    if not old_start:
        scene.START_NORM = tuple(float(v) for v in base_grid.find_start(jaw)[1])
    import expert
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    rob = SO101Mujoco(SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False,
                                        fps=expert.FPS, domain_rand=False, settle_s=1.0, snap_to_first_action=False,
                                        rig="base", start_jitter_deg=0.0, yolo_weights=None, jaw=jaw))
    rob.connect()
    return rob


def seeds_of(spec):
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":"))
        return list(range(a, b))
    return [int(x) for x in spec.split(",")]


def candidates():
    """(names, lens positions (n, 3), aims (n, 3)): today's two cameras first, then the grid."""
    bp = np.asarray(scene.base_cam_pos(), float)
    be = np.asarray(scene.BIRDSEYE_POS, float)
    t = np.radians(scene.BIRDSEYE_TILT_DEG)
    names = ["base", "birdseye"]
    pos = [bp, be]
    aim = [np.asarray(scene.BASE_CAM_AIM, float), be + np.array([0.0, np.cos(t), -np.sin(t)])]   # level, at the row
    for x in GX:
        for y in GY:
            for z in GZ:
                names.append(f"g_{x:.2f}_{y:.2f}_{z:.2f}")
                pos.append(np.array([x, y, z]))
                aim.append(np.asarray(scene.BASE_CAM_AIM, float))
    return names, np.array(pos), np.array(aim)


def axes(pos, aim):
    """Per camera: image right, image up, forward (world), as scene._look_xyaxes."""
    f = aim - pos
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    x = np.cross(f, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.cross(x, f), f


def project(lens, pos, ax, p):
    """Pixels (n, 2) of world point p in every camera (NaN behind it), and pixels per mm there."""
    x, y, f = ax
    v = p[None, :] - pos
    cam = np.stack([(v * x).sum(1), -(v * y).sum(1), (v * f).sum(1)], 1)      # OpenCV frame: x right, y down, z forward
    uv = lens.project(cam)
    du = lens.project(cam + np.array([0.001, 0.0, 0.0])) - uv
    dv = lens.project(cam + np.array([0.0, 0.001, 0.0])) - uv
    return uv, (np.linalg.norm(du, axis=1) + np.linalg.norm(dv, axis=1)) / 2


def inside(lens, uv):
    return ((uv[:, 0] >= IN_MARGIN_PX) & (uv[:, 0] <= lens.w - 1 - IN_MARGIN_PX)
            & (uv[:, 1] >= IN_MARGIN_PX) & (uv[:, 1] <= lens.h - 1 - IN_MARGIN_PX))


def arrive(m, d, origin, pts, tol, want=None, exclude=-1):
    """Per point: does a ray from origin get there? want: geom ids that count as arriving when they are what is hit."""
    import mujoco
    v = pts - origin[None, :]
    dist_to = np.linalg.norm(v, axis=1)
    u = np.ascontiguousarray((v / dist_to[:, None]).reshape(-1))
    n = len(pts)
    gid, hit = np.zeros(n, np.int32), np.zeros(n)
    mujoco.mj_multiRay(m, d, origin, u, GROUPS, 1, exclude, gid, hit, None, n, 10.0)
    ok = (hit < 0) | (hit >= dist_to - tol)
    if want is not None:
        ok |= np.array([g in w for g, w in zip(gid, want)])
    return ok


def model_cam_uv(m, d, cam_name, lens, p):
    """Pixel of world point p in one of the model's own cameras (a MuJoCo camera looks down its -z, y up)."""
    c = m.camera(cam_name).id
    R = d.cam_xmat[c].reshape(3, 3)
    q = R.T @ (p - d.cam_xpos[c])
    return lens.project(np.array([[q[0], -q[1], -q[2]]]))[0], d.cam_xpos[c].copy()


def scene_facts(rob, seed):
    """None, or the teacher and the cut geometry of this seed's target."""
    import truss_teacher as tt
    rob.new_scene(seed, log=False)
    rob.get_observation()
    if rob._tgt_k is None:
        return None, "no target"
    T = tt.TrussTeacher(rob.m, rob.d, rob.info, rob._tgt_k)
    r = T.t["r"]
    for beta in tt.APPROACH_BLEND_DEG:
        g = T._cut_geom(0.5, beta, r, 0.0, False)
        if not isinstance(g, str):
            return (T, g), ""
    return None, "no cut pose"


def balls(m, d, geoms, step=0.01):
    """(centres, radii) of balls covering the geoms: a capsule or cylinder as a row of balls every `step` along its
    axis, anything else as its bounding sphere. ⚠️ A stem is ONE 47 cm capsule: by its bounding sphere it would rule
    out every post within 24 cm of the row (the run of 2026-11-26 20:33 did exactly that; stopped, not used)."""
    import mujoco
    C, R = [], []
    for g in geoms:
        if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, h = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            for t in np.linspace(-h, h, max(2, int(np.ceil(2 * h / step)) + 1)):
                C.append(d.geom_xpos[g] + axis * t)
                R.append(r)
        else:
            C.append(np.array(d.geom_xpos[g]))
            R.append(float(m.geom_rbound[g]))
    return np.array(C), np.array(R)


def one_scene(rob, T, g, pos, ax, lens, claw_lens):
    """The answers of every candidate camera for one scene (see the module text), and the claw camera's."""
    import mujoco
    import truss_expert as tx
    n = len(pos)
    cm, cd = T.chk.m, T.chk.d
    p_cut = np.asarray(g["p_cut"], float)
    s = np.asarray(g["s"], float) / np.linalg.norm(g["s"])
    blades = sorted(T.chk.blades)
    arm = np.flatnonzero(T.chk.is_arm)
    claw_body = cm.camera("arm/claw").bodyid[0]
    clear, post = np.full(n, np.inf), np.full(n, np.inf)
    out = {}

    def to_post(c, r):
        """Per camera: distance of its post (the vertical line from the table up to the lens) to spheres (c, r)."""
        dz = c[None, :, 2] - np.clip(c[None, :, 2], 0.0, pos[:, None, 2])
        return (np.sqrt((pos[:, None, 0] - c[None, :, 0]) ** 2 + (pos[:, None, 1] - c[None, :, 1]) ** 2 + dz ** 2)
                - r[None, :]).min(1)

    def pose(q, jaw):
        cd.qpos[T.chk.adr] = q
        cd.qpos[T.g_adr] = jaw
        mujoco.mj_fwdPosition(cm, cd)
        c, r = balls(cm, cd, arm)                               # the arm: bounding spheres, capsules as rows
        dist = np.linalg.norm(pos[:, None, :] - c[None, :, :], axis=2) - r[None, :]
        np.minimum(clear, dist.min(1), out=clear)
        np.minimum(post, to_post(c, r), out=post)

    def claw_sees():
        uv, o = model_cam_uv(cm, cd, "arm/claw", claw_lens, p_cut)
        ok_in = bool(np.isfinite(uv).all() and inside(claw_lens, uv[None, :])[0])
        return ok_in, bool(arrive(cm, cd, o, np.array([p_cut + s * k for k in STALK_AT_M]), VIS_TOL_M,
                                  exclude=claw_body).any())

    # START
    pose(T.q_now(), float(rob.d.qpos[T.g_adr]))
    plants = np.array(sorted(T.chk.soft | T.target_all | {
        g for g in range(cm.ngeom) if (cm.geom(g).name or "").startswith("stem")}))
    out["plant"] = to_post(*balls(cm, cd, plants)).astype(np.float32)   # lens and post, to stems, stalks and fruit
    uv, ppm = project(lens, pos, ax, p_cut)
    out["s_in"] = inside(lens, uv)
    out["s_ppm"] = ppm.astype(np.float32)
    at = np.array([p_cut + s * k for k in STALK_AT_M])
    out["s_vis"] = np.array([arrive(cm, cd, pos[i], at, VIS_TOL_M).any() for i in range(n)])
    claw = list(claw_sees())
    # PRE (jaw open, SLIDE_M before the stalk)
    pose(np.asarray(g["slide"][0], float), np.radians(tx.JAW_OPEN_DEG))
    pts = np.vstack([at, cd.geom_xpos[blades]])
    want = [set()] * len(at) + [{b} for b in blades]
    r_ = np.array([arrive(cm, cd, pos[i], pts, VIS_TOL_M, want) for i in range(n)])
    out["p_vis"], out["p_blade"] = r_[:, :len(at)].any(1), r_[:, len(at):].any(1)
    claw += list(claw_sees())
    # CUT (jaw still open, around the stalk)
    pose(np.asarray(g["x_cut"], float), np.radians(tx.JAW_OPEN_DEG))
    near = np.array([p_cut + s * k for k in STALK_NEAR_M])
    pts = np.vstack([near, cd.geom_xpos[blades]])
    want = [set()] * len(near) + [{b} for b in blades]
    r_ = np.array([arrive(cm, cd, pos[i], pts, VIS_TOL_M, want) for i in range(n)])
    out["c_stalk"], out["c_blade"] = r_[:, :len(near)].any(1), r_[:, len(near):].any(1)
    out["c_in"] = out["s_in"]                      # the cut point itself has not moved
    for v in T.vias() + T.vias_v5():               # clearance only: the fixed poses the arm passes through
        pose(v, np.radians(tx.JAW_OPEN_DEG))
    out["clear"] = clear.astype(np.float32)
    out["post"] = post.astype(np.float32)
    out["claw"] = np.array(claw, bool)
    out["p_cut"] = p_cut
    return out


def collect(a):
    from lens import LENSES
    rob = setup(a.base, a.jaw, a.old_start)
    names, pos, aim = candidates()
    ax = axes(pos, aim)
    keys = ("p_cut", "claw", "s_in", "s_vis", "s_ppm", "p_vis", "p_blade", "c_in", "c_stalk", "c_blade", "clear",
            "post", "plant")
    rows = {k: [] for k in ("seed", "kz", "ntruss", "nfruit") + keys}
    skipped = {}
    b, e = names.index("base"), names.index("birdseye")
    for seed in seeds_of(a.seeds):
        facts, why = scene_facts(rob, seed)
        if facts is None:
            skipped[why] = skipped.get(why, 0) + 1
            print(f"{seed}: {why}", flush=True)
            continue
        T, g = facts
        out = one_scene(rob, T, g, pos, ax, LENSES["base"], LENSES["claw"])
        rows["seed"].append(seed)
        rows["kz"].append(float(T.t["knuckle"][2]))
        rows["ntruss"].append(len(rob.info["trusses"]))
        rows["nfruit"].append(sum(len(t["fruit"]) for t in rob.info["trusses"]))
        for k in keys:
            rows[k].append(out[k])
        print(f"{seed}: base start {int(out['s_in'][b] and out['s_vis'][b])} cut {int(out['c_stalk'][b])}"
              f"{int(out['c_blade'][b])} | birdseye start {int(out['s_in'][e] and out['s_vis'][e])} cut "
              f"{int(out['c_stalk'][e])}{int(out['c_blade'][e])} | claw {out['claw'].astype(int).tolist()}",
              flush=True)
    rob.disconnect()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), pos=pos, aim=aim, skipped=json.dumps(skipped),
                        setup=json.dumps(dict(base=a.base, jaw=a.jaw, old_start=a.old_start,
                                              start_norm=list(scene.START_NORM))),
                        **{k: np.array(v) for k, v in rows.items()})
    print(f"wrote {a.out}: {len(rows['seed'])} scenes, skipped {skipped}", flush=True)


def load(files):
    zs = [np.load(f, allow_pickle=False) for f in files]
    z0 = zs[0]
    cat = lambda k: np.concatenate([z[k] for z in zs if len(z["seed"])])   # noqa: E731
    D = {k: cat(k) for k in ("seed", "p_cut", "kz", "ntruss", "nfruit", "claw", "s_in", "s_vis", "s_ppm", "p_vis",
                             "p_blade", "c_in", "c_stalk", "c_blade", "clear", "post", "plant")}
    D.update(names=[str(x) for x in z0["names"]], pos=z0["pos"], aim=z0["aim"], setup=json.loads(str(z0["setup"])))
    return D


def mountable(D, zmax=Z_MAX_M):
    """Grid positions a post from the table can hold, with what rules each one out. Out: above the truss band
    (Z_MAX_M), in the carry zone (CARRY_X_M) or the crate (+3 cm), on the arm's base plate or row R0's anchor rail,
    the lens or its post within CLEAR_MIN_M of the arm (at the poses visited) or of a plant, in any scene."""
    import truss_scene as ts
    p = D["pos"]
    (cx, cy), (sx, sy, _) = ts.CRATE_CENTRE, ts.CRATE_SIZE
    b = np.asarray(D["setup"]["base"].split(","), float)
    why = {
        "above the truss band": p[:, 2] > zmax + 1e-9,
        "carry zone or crate": ((p[:, 0] >= CARRY_X_M) & (p[:, 1] >= 0.0)) | (
            (np.abs(p[:, 0] - cx) < sx / 2 + 0.03) & (np.abs(p[:, 1] - cy) < sy / 2 + 0.03)),
        "on the base plate or a rail": (np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 0.09) | (np.abs(p[:, 1]) < 0.03),
        "arm too near": np.minimum(D["clear"].min(0), D["post"].min(0)) < CLEAR_MIN_M,
        "plants too near": D["plant"].min(0) < CLEAR_MIN_M,
    }
    ok = ~np.any(list(why.values()), axis=0)
    ok[:2] = True                                                       # today's cameras: listed whatever they score
    return ok, {k: int(v[2:].sum()) for k, v in why.items()}


def rank(a):
    D = load(a.files)
    names, pos = D["names"], D["pos"]
    N = len(D["seed"])
    A = D["s_in"] & D["s_vis"] & (D["s_ppm"] >= PPM_MIN)        # START: the cut point is in view, sharp enough
    P = D["c_in"] & D["p_vis"] & D["p_blade"]                    # PRE: the cut point and a blade
    B = D["c_in"] & D["c_stalk"] & D["c_blade"]                  # CUT: the stalk next to the cut and a blade
    ok, why = mountable(D, a.zmax)
    claw = D["claw"].astype(bool)
    print(f"{N} scenes ({D['setup']}), {len(names)} cameras, {int(ok.sum()) - 2} mountable grid positions "
          f"(ruled out, overlapping: {why})")
    print(f"per scene: {D['ntruss'].mean():.1f} trusses, {D['nfruit'].mean():.0f} fruit")
    print(f"claw camera: cut point in frame and not hidden at START {100 * (claw[:, 0] & claw[:, 1]).mean():.0f} %, "
          f"at PRE {100 * (claw[:, 2] & claw[:, 3]).mean():.0f} %")

    def line(i):
        return (f"{names[i]:>18s}  start {100 * A[:, i].mean():3.0f} %  pre {100 * P[:, i].mean():3.0f} %  "
                f"cut {100 * B[:, i].mean():3.0f} %  in frame {100 * D['s_in'][:, i].mean():3.0f} %  "
                f"px/mm {np.median(D['s_ppm'][:, i]):.1f} (min {D['s_ppm'][:, i].min():.1f})  "
                f"arm {100 * min(D['clear'][:, i].min(), D['post'][:, i].min()):.0f} cm  "
                f"plants {100 * D['plant'][:, i].min():.0f} cm")

    one = A.mean(0) + B.mean(0)                                  # one fixed camera: start and cut, equal weight
    order = [i for i in np.argsort(-one) if ok[i]]
    print("\nOne fixed camera (start = sees the cut point at >= 1 px/mm; pre / cut = sees the stalk and a blade):")
    for i in (0, 1):
        print(line(i))
    for i in order[:a.top]:
        print(line(i))
    # pairs: among the best 120 singles by start and the best 120 by cut
    pool = sorted(set([i for i in np.argsort(-A.mean(0)) if ok[i]][:120]) |
                  set([i for i in np.argsort(-B.mean(0)) if ok[i]][:120]) | {0, 1})
    v = D["p_cut"][:, None, :] - pos[None, pool, :]              # scene x cam
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    best = []
    for ai, i in enumerate(pool):
        for bi in range(ai + 1, len(pool)):
            j = pool[bi]
            ang = np.degrees(np.arccos(np.clip((v[:, ai] * v[:, bi]).sum(1), -1, 1)))
            both = A[:, i] & A[:, j] & (ang >= CROSS_DEG)        # START: a 3D fix of the cut point
            cut = B[:, i] | B[:, j]                              # CUT: at least one sees stalk and blade
            pre = P[:, i] | P[:, j]
            best.append((both.mean() + cut.mean(), both.mean(), pre.mean(), cut.mean(), float(np.median(ang)), i, j))
    best.sort(reverse=True)
    print("\nTwo fixed cameras (3D = both see the cut point at START and their lines of sight cross at >= 45 deg; "
          "pre / cut = at least one sees stalk and blade):")
    ref = [b for b in best if {b[5], b[6]} == {0, 1}]
    for _, both, pre, cut, ang, i, j in ref + best[:a.top]:
        print(f"{names[i]:>18s} + {names[j]:<18s}  3D {100 * both:3.0f} %  pre {100 * pre:3.0f} %  cut "
              f"{100 * cut:3.0f} %  crossing {ang:3.0f} deg")
    if a.json:
        js = dict(scenes=N, setup=D["setup"],
                  claw=dict(start=float((claw[:, 0] & claw[:, 1]).mean()), pre=float((claw[:, 2] & claw[:, 3]).mean())),
                  single=[dict(name=names[i], pos=pos[i].tolist(), start=float(A[:, i].mean()),
                               pre=float(P[:, i].mean()), cut=float(B[:, i].mean()),
                               ppm=float(np.median(D["s_ppm"][:, i])), clear=float(D["clear"][:, i].min()))
                          for i in [0, 1] + order[:40]],
                  pairs=[dict(a=names[i], b=names[j], pos_a=pos[i].tolist(), pos_b=pos[j].tolist(), fix3d=both,
                              pre=pre, cut=cut, crossing=ang) for _, both, pre, cut, ang, i, j in ref + best[:40]])
        json.dump(js, open(a.json, "w"), indent=1)


def selftest(a):
    """The look-at projection against the model's own "base" camera, and the ray test against its segmentation
    render at the scene's first frame."""
    import mujoco
    from lens import LENSES
    rob = setup(a.base, a.jaw, a.old_start)
    names, pos, aim = candidates()
    ax = axes(pos, aim)
    lens = LENSES["base"]
    worst, table = 0.0, {}
    for seed in seeds_of(a.seeds):
        facts, why = scene_facts(rob, seed)
        if facts is None:
            continue
        T, g = facts
        p = np.asarray(g["p_cut"], float)
        uv_mine = project(lens, pos[:1], tuple(x[:1] for x in ax), p)[0][0]
        mujoco.mj_fwdPosition(rob.m, rob.d)
        uv_model, _ = model_cam_uv(rob.m, rob.d, "base", lens, p)
        worst = max(worst, float(np.abs(uv_mine - uv_model).max()))
        cd = T.chk.d
        cd.qpos[T.chk.adr], cd.qpos[T.g_adr] = T.q_now(), rob.d.qpos[T.g_adr]
        mujoco.mj_fwdPosition(T.chk.m, cd)
        sd = np.asarray(g["s"], float) / np.linalg.norm(g["s"])
        ray = bool(arrive(T.chk.m, cd, pos[0], np.array([p + sd * k for k in STALK_AT_M]), VIS_TOL_M).any())
        ids = rob._cameras().ids(rob.d, "base") if hasattr(rob, "_cameras") else None
        seg = None
        if ids is not None and np.isfinite(uv_model).all():
            u, v = int(round(uv_model[0])), int(round(uv_model[1]))
            if 3 <= u < lens.w - 3 and 3 <= v < lens.h - 3:
                seg = bool(np.isin(ids[v - 3:v + 4, u - 3:u + 4], list(T.target_stalk)).any())
        table[(ray, seg)] = table.get((ray, seg), 0) + 1
        print(f"{seed}: mine {uv_mine.round(1)} model {uv_model.round(1)} ray {ray} seg {seg}", flush=True)
    print(f"largest pixel difference look-at vs model camera: {worst:.2f}")
    print("(ray says visible, segmentation says visible): count", table)
    rob.disconnect()


def render(a):
    """A contact sheet: one row per scene and pose (START, CUT), one column per camera named in --cams, each the
    640 x 480 frame through the lens model with a cross on the cut point."""
    import cv2
    import mujoco
    import truss_expert as tx
    from lens import LENSES
    rob = setup(a.base, a.jaw, a.old_start)
    names, pos, aim = candidates()
    pick = [names.index(c) for c in a.cams.split(",")]
    ax = axes(pos, aim)
    lens = LENSES["base"]
    sheet = []
    for seed in seeds_of(a.seeds):
        facts, why = scene_facts(rob, seed)
        if facts is None:
            print(f"{seed}: {why}")
            continue
        T, g = facts
        m = rob.m
        keep = float(m.vis.global_.fovy)
        m.vis.global_.fovy = lens.fovy                      # a free camera takes the model's global field of view
        r = mujoco.Renderer(m, lens.pin_h, lens.pin_w)
        d2 = mujoco.MjData(m)
        p_cut = np.asarray(g["p_cut"], float)
        for label, q, jaw in (("start", T.q_now(), float(rob.d.qpos[T.g_adr])),
                              ("cut", np.asarray(g["x_cut"], float), np.radians(tx.JAW_OPEN_DEG))):
            d2.qpos[:] = rob.d.qpos
            d2.qpos[T.qidx[:5]], d2.qpos[T.g_adr] = q, jaw
            mujoco.mj_forward(m, d2)
            row = []
            for i in pick:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                f = aim[i] - pos[i]
                dist = float(np.linalg.norm(f))
                f = f / dist
                cam.lookat[:], cam.distance = aim[i], dist
                cam.azimuth, cam.elevation = np.degrees(np.arctan2(f[1], f[0])), np.degrees(np.arcsin(f[2]))
                r.update_scene(d2, camera=cam)
                img = np.ascontiguousarray(lens.distort(r.render())[..., ::-1])
                uv, _ = project(lens, pos[i:i + 1], tuple(x[i:i + 1] for x in ax), p_cut)
                if np.isfinite(uv).all():
                    cv2.drawMarker(img, (int(round(uv[0, 0])), int(round(uv[0, 1]))), (0, 255, 0),
                                   cv2.MARKER_CROSS, 22, 1)
                cv2.putText(img, f"{names[i]}  {seed} {label}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                row.append(img)
            sheet.append(np.hstack(row))
        r.close()
        m.vis.global_.fovy = keep
    rob.disconnect()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, np.vstack(sheet))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("collect", "selftest", "render"):
        s = sub.add_parser(name)
        s.add_argument("--base", default="0.37,0.20")
        s.add_argument("--jaw", default="v5", choices=scene.JAWS)
        s.add_argument("--old-start", action="store_true", help="keep scene.START_NORM (else base_grid.find_start)")
        s.add_argument("--seeds", required=True, help="a:b (range) or a comma list")
        if name != "selftest":
            s.add_argument("--out", required=True)
        if name == "render":
            s.add_argument("--cams", required=True, help="comma list of camera names (base, birdseye, g_x_y_z)")
    s = sub.add_parser("rank")
    s.add_argument("files", nargs="+")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--zmax", type=float, default=Z_MAX_M, help="highest lens allowed (see Z_MAX_M)")
    s.add_argument("--json")
    a = ap.parse_args()
    dict(collect=collect, rank=rank, selftest=selftest, render=render)[a.mode](a)
