#!/usr/bin/env python3
"""Scene checks: unit mapping, truss layouts, the full scene at the start pose, cameras (both rigs).

  python check_scene.py            all checks, PASS/FAIL per check, exit 1 on a FAIL
  python check_scene.py --quick    fewer layout seeds

The fingertips' cut + hold are check_truss_cut.py's, the tethers check_tomato_release.py's.
Watch it instead: python view_scene.py --cams [--rig base]

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
import tomato as tm  # noqa: E402
import truss as tr  # noqa: E402
import truss_scene as ts  # noqa: E402
import truss_target as ttg  # noqa: E402
from cameras import SimCameras  # noqa: E402

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
    m, _ = scene.compose("", "", "", arm=True, scenery=False, cut_hold=True)   # the truss scenes' arm, free space
    qidx, _ = scene.arm_ids(m)
    jid = [m.jnt_qposadr.tolist().index(q) for q in qidx]
    lo, hi = fm.rad_limits()
    print("     follower_white range vs MJCF joint range (deg):")
    for k, j in enumerate(arm_map.JOINTS):
        mlo, mhi = np.degrees(m.jnt_range[jid[k]])
        print(f"       {j:13s} real {np.degrees(lo[k]):+7.1f} .. {np.degrees(hi[k]):+7.1f}   "
              f"MJCF {mlo:+7.1f} .. {mhi:+7.1f}")
    over = np.degrees(np.maximum(m.jnt_range[jid, 0] - lo, hi - m.jnt_range[jid, 1])).max()
    report(over < 5, f"real follower range exceeds the MJCF limits by at most {over:.1f} deg (sim clips there)")
    print(f"     calibration: {fm.path} sha256 {fm.sha256[:12]}")


# ----------------------------------------------------------------------------------------------
def check_layouts(n_seeds, n_settle):
    """truss_scene layouts without the arm: no fruit overlapping, none on the floor, a quiet settle."""
    bad_overlap, bad_floor, bad_settle, counts, n_trusses = [], [], [], [], []
    for seed in range(n_seeds):
        m, d, info = ts.build(seed, arm=False)
        fruit = info["fruit"]
        counts.append(len(fruit))
        n_trusses.append(len(info["trusses"]))
        fg = [gid(m, f["geom"]) for f in fruit]
        if contacts(m, d, fg, fg):
            bad_overlap.append(seed)
        if any(d.geom_xpos[g][2] - m.geom_size[g][0] < 0.02 for g in fg):   # truss_scene.vines_xml's floor rule
            bad_floor.append(seed)
        if seed < n_settle:
            det = tm.Detacher(m, d, fruit)
            for _ in range(int(2.0 / scene.TIMESTEP)):
                mujoco.mj_step(m, d)
                det.step()
            if not np.all(np.isnan(det.detached_at)):
                bad_settle.append((seed, int(np.sum(~np.isnan(det.detached_at)))))
    report(not bad_overlap, f"{n_seeds} layouts: no two tomatoes start overlapping {bad_overlap[:10]}")
    report(not bad_floor, f"{n_seeds} layouts: every tomato hangs >= 2 cm above the floor {bad_floor[:10]}")
    report(not bad_settle, f"{n_settle} layouts: 2 s settle without a single release {bad_settle[:10]}")
    print(f"     trusses per scene: {min(n_trusses)}-{max(n_trusses)}; fruit per scene: {min(counts)}-{max(counts)}, "
          f"mean {np.mean(counts):.1f}")


# ----------------------------------------------------------------------------------------------
def check_full_scene(seeds):
    """The full truss scene (arm, base camera bracket): the arm starts clear and holds still, nothing comes off or
    gets cut, the ready truss shows in the birdseye and base cameras, the claw camera sees the fingers."""
    for seed in seeds:
        m, d, info = ts.build(seed, base_mount=True)
        qidx, uidx = scene.arm_ids(m)
        cams = SimCameras(m)   # ⚠️ one per model: a renderer is bound to its model
        arm_geoms = [i for i in range(m.ngeom) if m.geom(i).name.startswith("arm/") or
                     m.body(m.geom_bodyid[i]).name.startswith("arm/")]
        house = [i for i in range(m.ngeom) if not m.body(m.geom_bodyid[i]).name.startswith(("arm/", "s"))
                 and m.geom(i).name != "table"]
        report(not contacts(m, d, arm_geoms, house),
               f"seed {seed}: arm starts clear of the greenhouse, crate and base camera bracket")
        det = tm.Detacher(m, d, info["fruit"])
        cutters = [tr.TrussCutter(m, d, t) for t in info["trusses"]]
        d.ctrl[uidx] = d.qpos[qidx]
        for _ in range(int(2.0 / scene.TIMESTEP)):
            mujoco.mj_step(m, d)
            det.step()
            for c in cutters:
                c.step()
        drift = np.degrees(np.abs(d.qpos[qidx] - d.ctrl[uidx])).max()
        cut = [info["trusses"][k]["name"] for k, c in enumerate(cutters) if c.cut_t is not None]
        report(np.all(np.isnan(det.detached_at)) and not cut and drift < 2.0,
               f"seed {seed}: 2 s at the start pose, no release, no cut {cut}, arm holds within {drift:.2f} deg")
        # Cameras (through the lens model): the target truss gets a box on the birdseye (the start frame's true-box
        # path) and on the base camera (rig "base", view_target), the claw camera sees the fingers. ⚠️ Crowded on
        # purpose: a ready truss can be hidden behind others; then the box is its outline (truss_target.outline_box).
        if info["ready"]:
            st = ttg.start_frame(cams, d, m, info, [])
            px = {c: [ttg.truss_boxes(cams, d, m, info, cam=c)[k][1] for k in info["ready"]]
                  for c in ("birdseye", "base")}
            report(st["box"] is not None and st["target"] in info["ready"],
                   f"seed {seed}: birdseye start frame boxes ready truss {st['target_name']} ({st['source']}); "
                   f"ready trusses show {px['birdseye']} px on birdseye, {px['base']} px on base")
            if st["target"] is not None:
                box, cut = ttg.view_target(cams, d, m, info["trusses"][st["target"]], "base")
                report(box is not None, f"seed {seed}: base camera boxes it too, box {box}, cut point {cut}")
        else:
            print(f"     seed {seed}: no ready truss (skipped the camera box checks)")
        ids = cams.ids(d, "claw")
        fing = [i for i in range(m.ngeom) if m.geom(i).name.startswith(("arm/finray_", "arm/blade_", "arm/pad_"))]
        fpx = int(np.isin(ids, fing).sum())
        report(fpx > 500, f"seed {seed}: claw camera sees the fingers + tips, {fpx} px")
        cams.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="5 layout seeds instead of 30, 1 full scene instead of 3")
    a = ap.parse_args()
    print("== unit mapping"); check_mapping()
    print("\n== truss layouts"); check_layouts(5 if a.quick else 30, 3 if a.quick else 10)
    print("\n== full scene + cameras"); check_full_scene((0,) if a.quick else (0, 1, 2))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
