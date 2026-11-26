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
# The cameras must ride on the CART: on the real rig (hardware/greenhouse/pipe_rail_rig) the arm rolls along the row
# on a cart, so a camera standing on the floor would not move with it, and the row behind the arm (R0, empty in
# this sim) carries plants there. The cart's deck is two 119 mm planks side by side, centred on the arm base
# (pipe_rail_rig.fs PLANK_W, deck strips D-1 / D-2), and ends 140 mm from the arm base on the robot's left (deckX0).
DECK_HALF_W_M = 0.119      # rank: a post stands within this of the arm base's y (on the deck's width)
DECK_END_M = 0.140         # rank: left of base_x - this a post needs an outrigger batten past the deck's end


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
    long_ = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
    for g in geoms:
        # ⚠️ int(...) on both sides: a numpy int32 never equals a mujoco enum member, so `geom_type[g] in (enums)`
        # is always False (the run of 20:43 took every capsule as one sphere that way; stopped, not used)
        if int(m.geom_type[g]) in long_:
            r, h = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            for t in np.linspace(-h, h, max(2, int(np.ceil(2 * h / step)) + 1)):
                C.append(d.geom_xpos[g] + axis * t)
                R.append(r)
        else:
            C.append(np.array(d.geom_xpos[g]))
            R.append(float(m.geom_rbound[g]))
    return np.array(C), np.array(R)


def box_dist(m, d, geoms, pts):
    """Per point of pts (K, 3): the distance to the nearest of the geoms, each taken as its bounding BOX in its own
    frame (model.geom_aabb: centre + half size; for a box geom that is the box itself, for a mesh the box around
    it). Negative inside a box. Tighter than bounding spheres: the arm's longest link has a 9.5 cm sphere around a
    part about 4 cm thick."""
    out = np.full(len(pts), np.inf)
    for g in geoms:
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        q = np.abs((pts - c) @ R) - m.geom_aabb[g, 3:]
        np.minimum(out, np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0), out=out)
    return out


def post_points(pos, step=0.01):
    """(points, owner): each camera's post as points every `step` from the deck (z 0) up to its lens (the last
    point is the lens itself), and which camera each point belongs to."""
    P, own = [], []
    for i, (x, y, z) in enumerate(pos):
        zs = np.append(np.arange(0.0, z, step), z)
        P.append(np.column_stack([np.full(len(zs), x), np.full(len(zs), y), zs]))
        own.append(np.full(len(zs), i))
    return np.vstack(P), np.concatenate(own)


def one_scene(rob, T, g, pos, ax, lens, claw_lens, posts):
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
        np.minimum(clear, box_dist(cm, cd, arm, pos), out=clear)            # the lens to the arm's bounding boxes
        np.minimum.at(post, posts[1], box_dist(cm, cd, arm, posts[0]))      # the whole post (lens included)

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
    posts = post_points(pos)
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
        out = one_scene(rob, T, g, pos, ax, LENSES["base"], LENSES["claw"], posts)
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


def mountable(D, zmax=Z_MAX_M, anywhere=False):
    """Grid positions a post can hold, with what rules each one out. Out: above the truss band (zmax), off the
    cart's width (DECK_HALF_W_M; lifted by anywhere=True, which then only keeps posts off row R0's rail), in the
    carry zone (CARRY_X_M) or the crate (+3 cm), on the arm's base plate, the lens or its post within CLEAR_MIN_M of
    the arm (at the poses visited) or of a plant, in any scene."""
    import truss_scene as ts
    p = D["pos"]
    (cx, cy), (sx, sy, _) = ts.CRATE_CENTRE, ts.CRATE_SIZE
    b = np.asarray(D["setup"]["base"].split(","), float)
    why = {
        "above the truss band": p[:, 2] > zmax + 1e-9,
        "off the cart" if not anywhere else "on a rail": (np.abs(p[:, 1] - b[1]) > DECK_HALF_W_M + 1e-9
                                                          if not anywhere else np.abs(p[:, 1]) < 0.03),
        "carry zone or crate": ((p[:, 0] >= CARRY_X_M) & (p[:, 1] >= 0.0)) | (
            (np.abs(p[:, 0] - cx) < sx / 2 + 0.03) & (np.abs(p[:, 1] - cy) < sy / 2 + 0.03)),
        "on the base plate": np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 0.09,
        "arm too near": np.minimum(D["clear"].min(0), D["post"].min(0)) < CLEAR_MIN_M,
        "plants too near": D["plant"].min(0) < CLEAR_MIN_M,
    }
    ok = ~np.any(list(why.values()), axis=0)
    ok[:2] = True                                                       # today's cameras: listed whatever they score
    return ok, {k: int(v[2:].sum()) for k, v in why.items()}


def neighbours(pos):
    """Per grid camera: the grid cameras one step away in x, y or z (today's two cameras have none)."""
    key = {(round(float(x), 2), round(float(y), 2), round(float(z), 2)): i for i, (x, y, z) in enumerate(pos) if i >= 2}
    zs = [round(float(z), 2) for z in GZ]
    out = {0: [], 1: []}
    for (x, y, z), i in key.items():
        k = zs.index(z)
        cand = [(round(x - 0.05, 2), y, z), (round(x + 0.05, 2), y, z), (x, round(y - 0.05, 2), z),
                (x, round(y + 0.05, 2), z)] + [(x, y, zs[q]) for q in (k - 1, k + 1) if 0 <= q < len(zs)]
        out[i] = [key[c] for c in cand if c in key]
    return out


def rank(a):
    D = load(a.files)
    names, pos = D["names"], D["pos"]
    N = len(D["seed"])
    base = np.asarray(D["setup"]["base"].split(","), float)
    A = D["s_in"] & D["s_vis"] & (D["s_ppm"] >= PPM_MIN)        # START: the cut point is in view, sharp enough
    P = D["c_in"] & D["p_vis"] & D["p_blade"]                    # PRE: the cut point and a blade
    B = D["c_in"] & D["c_stalk"] & D["c_blade"]                  # CUT: the stalk next to the cut and a blade
    ok, why = mountable(D, a.zmax, a.anywhere)
    nb = neighbours(pos)
    claw = D["claw"].astype(bool)
    print(f"{N} scenes ({D['setup']}), {len(names)} cameras, {int(ok.sum()) - 2} mountable grid positions "
          f"(lens <= {a.zmax:.2f} m, {'anywhere in the house' if a.anywhere else 'on the cart'}; ruled out, "
          f"overlapping: {why})")
    print(f"per scene: {D['ntruss'].mean():.1f} trusses, {D['nfruit'].mean():.0f} fruit")
    print(f"claw camera: cut point in frame and not hidden at START {100 * (claw[:, 0] & claw[:, 1]).mean():.0f} %, "
          f"at PRE {100 * (claw[:, 2] & claw[:, 3]).mean():.0f} %")

    def mount(i):
        if i < 2:
            return "today's"
        out = (base[0] - DECK_END_M) - pos[i, 0]
        return "on the deck" if out <= 0 else f"outrigger {100 * out:.0f} cm"

    def line(i):
        return (f"{names[i]:>18s}  start {100 * A[:, i].mean():3.0f} %  pre {100 * P[:, i].mean():3.0f} %  "
                f"cut {100 * B[:, i].mean():3.0f} %  in frame {100 * D['s_in'][:, i].mean():3.0f} %  "
                f"px/mm {np.median(D['s_ppm'][:, i]):.1f} (min {D['s_ppm'][:, i].min():.1f})  "
                f"arm {100 * min(D['clear'][:, i].min(), D['post'][:, i].min()):.0f} cm  "
                f"plants {100 * D['plant'][:, i].min():.0f} cm  {mount(i)}")

    one = A.mean(0) + B.mean(0)                                  # one fixed camera: start and cut, equal weight
    order = [i for i in np.argsort(-one) if ok[i]]
    print("\nOne fixed camera (start = sees the cut point at >= 1 px/mm; pre / cut = sees the stalk and a blade):")
    for i in (0, 1):
        print(line(i))
    for i in order[:a.top]:
        print(line(i))
    # pairs: every mountable position if they are few, else the best 150 by start and the best 150 by cut
    cand = [i for i in range(len(names)) if ok[i]]
    if len(cand) > 320:
        cand = sorted(set([i for i in np.argsort(-A.mean(0)) if ok[i]][:150]) |
                      set([i for i in np.argsort(-B.mean(0)) if ok[i]][:150]) | {0, 1})
    unit = D["p_cut"][:, None, :] - pos[None, :, :]              # scene x camera: line of sight to the cut point
    unit /= np.linalg.norm(unit, axis=2, keepdims=True)

    def pair(i, j):
        ang = np.degrees(np.arccos(np.clip((unit[:, i] * unit[:, j]).sum(1), -1, 1)))
        both = A[:, i] & A[:, j] & (ang >= CROSS_DEG)            # START: a 3D fix of the cut point
        cut = B[:, i] | B[:, j]                                  # CUT: at least one sees stalk and blade
        pre = P[:, i] | P[:, j]
        return both.mean() + cut.mean(), both.mean(), pre.mean(), cut.mean(), float(np.median(ang))

    best = sorted(((*pair(i, j), i, j) for q, i in enumerate(cand) for j in cand[q + 1:]), reverse=True)
    ref = [b for b in best if {b[5], b[6]} == {0, 1}]
    rows = []
    for sc, both, pre, cut, ang, i, j in ref + best[:max(a.top, 60)]:
        # plateau, not peak (the lens is a guess and a mount is never exact): the worst score when either camera
        # sits one grid step (5 cm, or one height step) away
        worst = min(pair(i2, j2)[0] for i2 in [i] + nb[i] for j2 in [j] + nb[j]) if i >= 2 and j >= 2 else sc
        rows.append(dict(a=names[i], b=names[j], pos_a=pos[i].tolist(), pos_b=pos[j].tolist(), score=sc, fix3d=both,
                         pre=pre, cut=cut, crossing=ang, worst_neighbour=worst,
                         ppm_min=[float(D["s_ppm"][:, i].min()), float(D["s_ppm"][:, j].min())],
                         mount=[mount(i), mount(j)],
                         arm_cm=[100 * float(min(D["clear"][:, k].min(), D["post"][:, k].min())) for k in (i, j)]))
    print("\nTwo fixed cameras (3D = both see the cut point at START and their lines of sight cross at >= 45 deg; "
          "pre / cut = at least one sees stalk and blade; worst = the score of 2 with either camera one grid step "
          "off):")
    shown = rows[:len(ref)] + sorted(rows[len(ref):], key=lambda r: (-r["worst_neighbour"], -r["score"]))[:a.top]
    for r in shown:
        print(f"{r['a']:>18s} + {r['b']:<18s}  3D {100 * r['fix3d']:3.0f} %  pre {100 * r['pre']:3.0f} %  cut "
              f"{100 * r['cut']:3.0f} %  crossing {r['crossing']:3.0f} deg  worst {r['worst_neighbour']:.2f}  "
              f"px/mm min {r['ppm_min'][0]:.1f} / {r['ppm_min'][1]:.1f}  arm {r['arm_cm'][0]:.0f} / "
              f"{r['arm_cm'][1]:.0f} cm  {r['mount'][0]} / {r['mount'][1]}")
    if a.json:
        js = dict(scenes=N, setup=D["setup"], zmax=a.zmax, anywhere=a.anywhere, ruled_out=why,
                  claw=dict(start=float((claw[:, 0] & claw[:, 1]).mean()), pre=float((claw[:, 2] & claw[:, 3]).mean())),
                  single=[dict(name=names[i], pos=pos[i].tolist(), start=float(A[:, i].mean()),
                               pre=float(P[:, i].mean()), cut=float(B[:, i].mean()),
                               in_frame=float(D["s_in"][:, i].mean()), ppm=float(np.median(D["s_ppm"][:, i])),
                               ppm_min=float(D["s_ppm"][:, i].min()), clear=float(D["clear"][:, i].min()),
                               post=float(D["post"][:, i].min()), plant=float(D["plant"][:, i].min()),
                               mount=mount(i)) for i in [0, 1] + order[:60]],
                  pairs=rows)
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
    s.add_argument("--anywhere", action="store_true",
                   help="lift the cart rule (DECK_HALF_W_M): posts anywhere in the house. For comparison only: such "
                        "a camera does not move with the arm on the real rig.")
    s.add_argument("--json")
    a = ap.parse_args()
    dict(collect=collect, rank=rank, selftest=selftest, render=render)[a.mode](a)
