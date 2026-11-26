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
        (v5's world), so it runs on --n episodes, spread over every target position and height.

Also why it matters for re-making v5's pictures with new cameras: a post the arm or a truss passed through in a
recorded episode would show that in the re-made pictures. For that question itself:

    python whatif/camera_sweep_logs.py touch --from fine_arm_eps_*.json fine_plants_eps_*.json \
        --cams g_0.02_0.30_0.42,g_0.02_0.30_0.05

touch:  replays the episodes the sweeps listed as near misses (--episodes-json files, the seeds under --cams) with
        the rig's real brackets in the model (post cylinder and camera housing, as remake_pictures.py shows them)
        and asks the simulator's own collision check, frame by frame, whether the arm or a plant part is inside one.
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
NOTE_M = 0.06               # --episodes-json lists every episode that came nearer than this to a post (guess: the
#                             4 cm rule plus room to see how close the near misses are)


def shard_of(S, spec):
    i, n = (int(v) for v in spec.split("/"))
    return S[i::n]


def note(per_ep, names, x, ep_min):
    for i in np.flatnonzero(ep_min < NOTE_M):
        per_ep.setdefault(names[i], {})[str(x["seed"])] = round(float(ep_min[i]), 4)


def write(a, names, best, per_ep):
    if a.json:
        json.dump({nm: float(best[i]) for i, nm in enumerate(names)}, open(a.json, "w"), indent=1)
    if a.episodes_json:
        json.dump(per_ep, open(a.episodes_json, "w"), indent=1)


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
    S = shard_of(S[:a.limit] if a.limit else S, a.shard)
    qidx, _ = scene.arm_ids(m)
    arm_g = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("arm/")])
    fm = arm_map.FollowerMap()
    best = np.full(len(names), np.inf)
    where = [None] * len(names)
    close_frames, close_eps, frames, per_ep = np.zeros(len(names), int), np.zeros(len(names), int), 0, {}
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
        note(per_ep, names, x, ep_min)
        frames += len(st)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1} episodes, {frames} frames", flush=True)
    print(f"\narm against the posts over {len(S)} recorded episodes ({frames} frames, every {a.every}. frame):")
    for i, nm in enumerate(names):
        print(f"  {nm:>18s}  nearest {100 * best[i]:5.1f} cm (seed {where[i][0]}, frame {where[i][1]}, target "
              f"{where[i][2]})  under {100 * CLEAR_M:.0f} cm in {close_eps[i]} episodes / {close_frames[i]} frames")
    write(a, names, best, per_ep)


def plants(a):
    import mujoco

    import truss_scene as ts
    names, pos = cam_positions(a.cams)
    ts.set_world("crate23")
    S = [x for x in rc.v5_scenes() if not x["hold"]]
    # spread over every position and height: the k-th recorded episode of each target slot in turn (s0t0, s0t1, ...,
    # s4t1; s0 is the stem nearest the robot's left). ⚠️ Until 2026-11-26 the first half was "sorted by target
    # name", which is 30 times s0t0. Every scene tests all its plants; the slot only decides the carried truss's way.
    by = {}
    for x in S:
        by.setdefault(x["target"], []).append(x)
    pick = [by[t][k] for k in range(max(len(v) for v in by.values())) for t in sorted(by) if k < len(by[t])]
    pick = shard_of(pick[:a.n] if a.n else pick, a.shard)        # --n 0: every cut episode

    def to_post(c, r):
        dz = c[None, :, 2] - np.clip(c[None, :, 2], 0.0, pos[:, None, 2])
        dist = np.sqrt((pos[:, None, 0] - c[None, :, 0]) ** 2 + (pos[:, None, 1] - c[None, :, 1]) ** 2 + dz ** 2) - r
        return dist.min(1)

    best = np.full(len(names), np.inf)
    where = [None] * len(names)
    close_eps, per_ep = np.zeros(len(names), int), {}
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
        note(per_ep, names, x, ep_min)
        print(f"  {n + 1}/{len(pick)} seed {x['seed']} target {x['target']}: nearest "
              f"{(100 * ep_min).round(1).tolist()} cm", flush=True)
    print(f"\nplants (stems, every truss, the carried one included) against the posts over {len(pick)} recorded "
          f"episodes, every {a.every}. frame:")
    for i, nm in enumerate(names):
        print(f"  {nm:>18s}  nearest {100 * best[i]:5.1f} cm (seed {where[i][0]}, frame {where[i][1]}, target "
              f"{where[i][2]})  under {100 * CLEAR_M:.0f} cm in {close_eps[i]} episodes")
    write(a, names, best, per_ep)


def touch(a):
    import mujoco

    import truss_scene as ts
    from cameras import RIG_FIXED
    ts.set_world("crate23")
    names = set(a.cams.split(","))
    seeds = {}
    for f in a.from_files:
        for cam, eps in json.load(open(f)).items():
            if cam in names:
                for sd, dist in eps.items():
                    seeds[int(sd)] = min(dist, seeds.get(int(sd), np.inf))
    S = {x["seed"]: x for x in rc.v5_scenes()}
    hit = {}
    print(f"{len(seeds)} episodes listed within {100 * NOTE_M:.0f} cm of {sorted(names)}; replaying them with rig "
          f"{a.rig}'s brackets {RIG_FIXED[a.rig]} in the scene (decided with the old bracket)", flush=True)
    for n, (sd, near) in enumerate(sorted(seeds.items(), key=lambda kv: kv[1])):
        x = S[sd]
        z = np.load(x["npz"], allow_pickle=True)
        m, d, _ = ts.build(sd, hold=bool(x["hold"]), mounts=RIG_FIXED["base"], show_mounts=RIG_FIXED[a.rig], jaw="v5")
        if (m.nq, m.neq) != (z["qpos"].shape[1], z["eq_active"].shape[1]):
            print(f"  seed {sd}: rebuilt model differs from the log, skipped", flush=True)
            continue
        cam_g = {g for g in range(m.ngeom) if "_cam_" in (m.geom(g).name or "")}
        worst, frames, who = 0.0, 0, set()
        for k in range(len(z["qpos"])):
            d.qpos[:] = z["qpos"][k]
            d.eq_active[:] = z["eq_active"][k]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            deep = 0.0
            for c in d.contact[:d.ncon]:
                if (c.geom1 in cam_g) != (c.geom2 in cam_g) and c.dist < 0:
                    other = c.geom2 if c.geom1 in cam_g else c.geom1
                    who.add(f"{m.geom(other).name or m.body(m.geom_bodyid[other]).name} on "
                            f"{m.geom(c.geom1 if c.geom1 in cam_g else c.geom2).name}")
                    deep = min(deep, c.dist)
            frames += deep < 0
            worst = min(worst, deep)
        if frames:
            hit[sd] = dict(frames=frames, deepest_mm=round(-1000 * worst, 1), target=x["target"], what=sorted(who))
        print(f"  {n + 1}/{len(seeds)} seed {sd} target {x['target']} (sweep: {100 * near:.1f} cm): "
              + (f"INSIDE a bracket in {frames} frames, deepest {-1000 * worst:.1f} mm: {sorted(who)}" if frames
                 else "nothing touches a bracket"), flush=True)
    print(f"\n{len(hit)} of {len(seeds)} replayed episodes have the arm or a plant part inside a bracket: "
          f"{sorted(hit)}")
    if a.json:
        json.dump({str(k): v for k, v in hit.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("arm", "plants"):
        s = sub.add_parser(name)
        s.add_argument("--cams", required=True, help="comma list: camera_score names (base, g_x_y_z) or x:y:z")
        s.add_argument("--every", type=int, default=2 if name == "arm" else 3, help="use every k-th logged frame")
        s.add_argument("--json", help="write {camera: nearest distance in m} here (camera_score.py rank --sweep)")
        s.add_argument("--episodes-json", help="write {camera: {seed: nearest distance}} for every episode that "
                                               "came within NOTE_M of that camera's post")
        s.add_argument("--shard", default="0/1", help="i/n: every n-th episode from the i-th (parallel runs)")
        if name == "plants":
            s.add_argument("--n", type=int, default=60, help="episodes to rebuild (0 = every cut episode)")
        else:
            s.add_argument("--limit", type=int, default=0, help="only the first N episodes (a quick try)")
    s = sub.add_parser("touch")
    s.add_argument("--from", dest="from_files", nargs="+", required=True, help="--episodes-json files of arm / plants")
    s.add_argument("--cams", required=True, help="comma list: the sweep's camera names whose near misses to replay")
    s.add_argument("--rig", default="trio", help="the rig whose brackets stand in the replayed scenes")
    s.add_argument("--json", help="write {seed: {frames, deepest_mm, target, what}} of the episodes that touch")
    a = ap.parse_args()
    dict(arm=arm, plants=plants, touch=touch)[a.mode](a)
