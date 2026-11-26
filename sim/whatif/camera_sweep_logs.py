#!/usr/bin/env python3
"""Would a camera post have been in the way in dataset v5's 800 recorded episodes? READ-ONLY, nothing is recorded.

camera_score.py judges a post against the arm at a handful of poses per scene (start, before the cut, at the cut,
the planner's fixed via poses). The recorded episodes have every frame of the real motion, the carry to the crate
included, in the simulator's state logs (~/.cache/blush_sim/states, see rebuild_v5_check.py). This sweeps those:

    python whatif/camera_sweep_logs.py arm    --cams g_0.02_0.30_0.05,g_0.02_0.20_0.34      # all 800, ~3 min
    python whatif/camera_sweep_logs.py plants --cams ... --n 60                               # 60 rebuilt scenes

arm:    per camera, the smallest distance over every logged frame from its lens and its post (the vertical line
        from the deck up to the lens) to the arm's parts, each part taken as its bounding box (camera_score.box_dist).
        The arm's pose comes from the logged joint readings, so no scene is rebuilt.
plants: per camera, the smallest distance over the logged frames from its post to any plant part: stems, every
        truss's stalk and fruit, the cut truss while the arm carries it. Needs the scene rebuilt from its seed
        (v5's world), so it runs on --n episodes: those whose cut point is nearest the cameras first (the trusses a
        post could touch), then spread over the rest.

Also why it matters for re-making v5's pictures with new cameras: a post the arm or a truss passed through in a
recorded episode would show that in the re-made pictures.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import camera_score as cs  # noqa: E402
import rebuild_v5_check as rc  # noqa: E402

CLEAR_M = cs.CLEAR_MIN_M


def cam_positions(spec):
    """'name,name' (camera_score names: base, g_x_y_z) and/or 'x:y:z' triples -> (names, positions (n, 3))."""
    names, pos, _ = cs.candidates()
    if spec.startswith("@"):                       # a file holding the comma list (camera_score rank --list-mountable)
        spec = open(spec[1:]).read().strip()
    out_n, out_p = [], []
    for tok in spec.split(","):
        if ":" in tok:
            out_n.append(tok)
            out_p.append([float(v) for v in tok.split(":")])
        else:
            out_n.append(tok)
            out_p.append(pos[names.index(tok)])
    return out_n, np.array(out_p, float)


def arm(a):
    import mujoco

    import arm_map
    import scene
    import truss_scene as ts
    names, pos = cam_positions(a.cams)
    P, own = cs.post_points(pos)
    ts.set_world("crate23")
    S = rc.v5_scenes()
    m, d, _ = ts.build(S[1]["seed"], base_mount=True, jaw="v5")     # any scene: only the arm is posed
    S = S[:a.limit] if a.limit else S
    qidx, _ = scene.arm_ids(m)
    arm_g = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("arm/")])
    fm = arm_map.FollowerMap()
    best = np.full(len(names), np.inf)
    where = [None] * len(names)
    close_frames, close_eps, frames = np.zeros(len(names), int), np.zeros(len(names), int), 0
    for n, x in enumerate(S):
        z = np.load(x["npz"], allow_pickle=True)
        st = np.asarray(z["state"], float)[::a.every]
        ep_min = np.full(len(names), np.inf)
        for k, row in enumerate(st):
            d.qpos[qidx] = fm.norm_to_rad(row)
            mujoco.mj_kinematics(m, d)
            dist = np.full(len(names), np.inf)
            np.minimum.at(dist, own, cs.box_dist(m, d, arm_g, P))
            close_frames += dist < CLEAR_M
            new = dist < best
            for i in np.flatnonzero(new):
                where[i] = (x["seed"], k * a.every, x["target"])
            best = np.minimum(best, dist)
            ep_min = np.minimum(ep_min, dist)
        close_eps += ep_min < CLEAR_M
        frames += len(st)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1} episodes, {frames} frames", flush=True)
    print(f"\narm against the posts over {len(S)} recorded episodes ({frames} frames, every {a.every}. frame):")
    for i, nm in enumerate(names):
        print(f"  {nm:>18s}  nearest {100 * best[i]:5.1f} cm (seed {where[i][0]}, frame {where[i][1]}, target "
              f"{where[i][2]})  under {100 * CLEAR_M:.0f} cm in {close_eps[i]} episodes / {close_frames[i]} frames")
    if a.json:
        json.dump({nm: float(best[i]) for i, nm in enumerate(names)}, open(a.json, "w"), indent=1)


def plants(a):
    import mujoco

    import truss_scene as ts
    names, pos = cam_positions(a.cams)
    ts.set_world("crate23")
    S = [x for x in rc.v5_scenes() if not x["hold"]]
    # the scenes whose target hangs nearest the cameras first (the logged cue has no 3D point: by the target's stem,
    # s0 is the stem nearest the robot's left), then every k-th of the rest
    near = sorted(S, key=lambda x: x["target"])[:a.n // 2]
    rest = [x for x in S if x not in near]
    pick = near + rest[::max(1, len(rest) // max(1, a.n - len(near)))][:a.n - len(near)]

    def to_post(c, r):
        dz = c[None, :, 2] - np.clip(c[None, :, 2], 0.0, pos[:, None, 2])
        dist = np.sqrt((pos[:, None, 0] - c[None, :, 0]) ** 2 + (pos[:, None, 1] - c[None, :, 1]) ** 2 + dz ** 2) - r
        return dist.min(1)

    best = np.full(len(names), np.inf)
    where = [None] * len(names)
    close_eps = np.zeros(len(names), int)
    for n, x in enumerate(pick):
        z = np.load(x["npz"], allow_pickle=True)
        m, d, info = ts.build(x["seed"], base_mount=True, jaw="v5")
        if (m.nq, m.neq) != (z["qpos"].shape[1], z["eq_active"].shape[1]):
            print(f"  seed {x['seed']}: rebuilt model differs from the log, skipped", flush=True)
            continue
        pg = sorted({m.geom(g).id for t in info["trusses"] for g in t["geoms"]}
                    | {m.geom(f["geom"]).id for t in info["trusses"] for f in t["fruit"]}
                    | {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("stem")})
        ep_min = np.full(len(names), np.inf)
        for k in range(0, len(z["qpos"]), a.every):
            d.qpos[:] = z["qpos"][k]
            mujoco.mj_kinematics(m, d)
            C, R = cs.balls(m, d, pg)
            dist = to_post(C, R)
            for i in np.flatnonzero(dist < best):
                where[i] = (x["seed"], k, x["target"])
            best = np.minimum(best, dist)
            ep_min = np.minimum(ep_min, dist)
        close_eps += ep_min < CLEAR_M
        print(f"  {n + 1}/{len(pick)} seed {x['seed']} target {x['target']}: nearest "
              f"{(100 * ep_min).round(1).tolist()} cm", flush=True)
    print(f"\nplants (stems, every truss, the carried one included) against the posts over {len(pick)} recorded "
          f"episodes, every {a.every}. frame:")
    for i, nm in enumerate(names):
        print(f"  {nm:>18s}  nearest {100 * best[i]:5.1f} cm (seed {where[i][0]}, frame {where[i][1]}, target "
              f"{where[i][2]})  under {100 * CLEAR_M:.0f} cm in {close_eps[i]} episodes")
    if a.json:
        json.dump({nm: float(best[i]) for i, nm in enumerate(names)}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("arm", "plants"):
        s = sub.add_parser(name)
        s.add_argument("--cams", required=True, help="comma list: camera_score names (base, g_x_y_z) or x:y:z")
        s.add_argument("--every", type=int, default=2 if name == "arm" else 3, help="use every k-th logged frame")
        s.add_argument("--json", help="write {camera: nearest distance in m} here (camera_score.py rank --sweep)")
        if name == "plants":
            s.add_argument("--n", type=int, default=60, help="episodes to rebuild")
        else:
            s.add_argument("--limit", type=int, default=0, help="only the first N episodes (a quick try)")
    a = ap.parse_args()
    dict(arm=arm, plants=plants)[a.mode](a)
