#!/usr/bin/env python3
"""Step 2 checks: unit mapping, tomato layouts, cameras. The grasp checks are in check_grasp.py.

  python check_scene.py            all checks, PASS/FAIL per check, exit 1 on a FAIL
  python check_scene.py --quick    fewer layout seeds

Watch it instead: python view_scene.py --cams

Writes nothing. Run in the lerobot env.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")   # headless only; view_scene.py is the windowed one

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import arm_map  # noqa: E402
import scene  # noqa: E402
from cameras import SimCameras  # noqa: E402
import tomato as tm  # noqa: E402

FAILS = 0


def report(ok, msg):
    global FAILS
    FAILS += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def contacts(m, d, a_ids, b_ids):
    a_ids, b_ids = set(a_ids), set(b_ids)
    return [(c.geom1, c.geom2) for c in d.contact[:d.ncon]
            if (c.geom1 in a_ids and c.geom2 in b_ids) or (c.geom2 in a_ids and c.geom1 in b_ids)]


# ----------------------------------------------------------------------------------------------
def check_mapping():
    fm = arm_map.FollowerMap()
    rng = np.random.default_rng(0)
    norms = np.c_[rng.uniform(-100, 100, (200, 5)), rng.uniform(0, 100, (200, 1))]
    err = max(np.abs(fm.rad_to_norm(fm.norm_to_rad(n)) - n).max() for n in norms)
    report(err < 1e-6, f"norm -> rad -> norm round trip, max error {err:.1e}")
    q_closed = fm.norm_to_rad([0, 0, 0, 0, 0, 0])[5]
    q_open = fm.norm_to_rad([0, 0, 0, 0, 0, 100])[5]
    report(abs(np.degrees(q_closed) + 10) < 1e-6 and abs(q_open - scene.GRIPPER_RANGE_RAD[1]) < 1e-6,
           f"gripper 0 -> {np.degrees(q_closed):.1f} deg (closed stop), 100 -> {np.degrees(q_open):.1f} deg")
    m, d, _ = scene.build(0)
    qidx, _ = scene.arm_ids(m)
    jid = [m.jnt_qposadr.tolist().index(q) for q in qidx]
    lo, hi = fm.rad_limits()
    print("     follower_white range vs MJCF joint range (deg):")
    for k, j in enumerate(arm_map.JOINTS):
        mlo, mhi = np.degrees(m.jnt_range[jid[k]])
        print(f"       {j:13s} real {np.degrees(lo[k]):+7.1f} .. {np.degrees(hi[k]):+7.1f}   MJCF {mlo:+7.1f} .. {mhi:+7.1f}")
    over = np.degrees(np.maximum(m.jnt_range[jid, 0] - lo, hi - m.jnt_range[jid, 1])).max()
    report(over < 5, f"real follower range exceeds the MJCF limits by at most {over:.1f} deg (sim clips there)")
    print(f"     calibration: {fm.path} sha256 {fm.sha256[:12]}")


# ----------------------------------------------------------------------------------------------
def check_layouts(n_seeds, n_settle):
    bad_overlap, bad_ripe, bad_settle, counts, n_ripe = [], [], [], [], []
    sh = scene.shoulder_world()
    for seed in range(n_seeds):
        m, d, info = scene.build(seed, arm=False)
        fruit = info["fruit"]
        counts.append(len(fruit))
        tg = [gid(m, f["geom"]) for f in fruit]
        if contacts(m, d, tg, tg):
            bad_overlap.append(seed)
        ripe = [k for k, f in enumerate(fruit) if f["ripeness"] == "ripe"]
        n_ripe.append(len(ripe))
        apart = all(np.linalg.norm(fruit[a]["pos"] - fruit[b]["pos"]) >= scene.RIPE_GAP_M
                    for a in ripe for b in ripe if a < b)
        if (ripe != info["ripe"] or not scene.N_RIPE[0] <= len(ripe) <= scene.N_RIPE[1] or not apart
                or any(np.linalg.norm(fruit[k]["pos"] - sh) >= scene.REACH_M for k in ripe)):
            bad_ripe.append(seed)
        if seed < n_settle:
            det = tm.Detacher(m, d, fruit)
            for _ in range(int(2.0 / scene.TIMESTEP)):
                mujoco.mj_step(m, d)
                det.step()
            if not np.all(np.isnan(det.detached_at)):
                bad_settle.append((seed, int(np.sum(~np.isnan(det.detached_at)))))
    report(not bad_overlap, f"{n_seeds} layouts: no two tomatoes start overlapping {bad_overlap[:10]}")
    report(not bad_ripe, f"{n_seeds} layouts: {scene.N_RIPE[0]}-{scene.N_RIPE[1]} ripe fruits, all within reach, "
                         f">= {scene.RIPE_GAP_M * 100:.0f} cm apart {bad_ripe[:10]}")
    report(not bad_settle, f"{n_settle} layouts: 2 s settle without a single release {bad_settle[:10]}")
    print(f"     fruit per scene: {min(counts)}-{max(counts)}, mean {np.mean(counts):.1f}; "
          f"ripe per scene: {dict(zip(*np.unique(n_ripe, return_counts=True)))}")


# ----------------------------------------------------------------------------------------------
def check_full_scene(seeds):
    for seed in seeds:
        m, d, info = scene.build(seed)
        qidx, uidx = scene.arm_ids(m)
        cams = SimCameras(m)   # ⚠️ one per model: a renderer is bound to its model
        arm_geoms = [i for i in range(m.ngeom) if m.geom(i).name.startswith("arm/") or
                     m.body(m.geom_bodyid[i]).name.startswith("arm/")]
        house = [i for i in range(m.ngeom) if not m.body(m.geom_bodyid[i]).name.startswith(("arm/", "s"))
                 and m.geom(i).name != "table"]
        report(not contacts(m, d, arm_geoms, house), f"seed {seed}: arm starts clear of the greenhouse")
        det = tm.Detacher(m, d, info["fruit"])
        d.ctrl[uidx] = d.qpos[qidx]
        for _ in range(int(2.0 / scene.TIMESTEP)):
            mujoco.mj_step(m, d)
            det.step()
        drift = np.degrees(np.abs(d.qpos[qidx] - d.ctrl[uidx])).max()
        report(np.all(np.isnan(det.detached_at)) and drift < 2.0,
               f"seed {seed}: 2 s at the start pose, no release, arm holds within {drift:.2f} deg")
        # Cameras (through the lens model): the ripe fruits in the birdseye view, the fingers in the claw view.
        seen = [cams.true_box(d, "birdseye", gid(m, info["fruit"][k]["geom"])) for k in info["ripe"]]
        report(max(px for _, px in seen) >= 30,
               f"seed {seed}: ripe fruits visible in birdseye, " + ", ".join(
                   f"{px} px ({b and b[2] - b[0] + 1} px across)" for b, px in seen) + f" at {scene.CAM_W}x{scene.CAM_H}")
        ids = cams.ids(d, "claw")
        fing = [i for i in range(m.ngeom) if m.geom(i).name.startswith(("arm/finger_", "arm/finray_"))]
        fpx = int(np.isin(ids, fing).sum())
        report(fpx > 500, f"seed {seed}: claw camera sees the fin-ray fingers, {fpx} px")
        cams.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 layout seeds instead of 200")
    a = ap.parse_args()
    print("== unit mapping"); check_mapping()
    print("\n== tomato layouts"); check_layouts(20 if a.quick else 200, 10 if a.quick else 50)
    print("\n== full scene + cameras"); check_full_scene((0, 1) if a.quick else (0, 1, 2, 3, 4))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
