#!/usr/bin/env python3
"""Step 2 checks: unit mapping, tomato layouts, cameras, and a real grasp-and-pluck through contact.

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
        fing = [gid(m, "arm/finger_fixed"), gid(m, "arm/finger_moving")]
        fpx = int(np.isin(ids, fing).sum())
        report(fpx > 500, f"seed {seed}: claw camera sees the fin-ray fingers, {fpx} px")
        cams.close()


# ----------------------------------------------------------------------------------------------
def finger_verts(m, d, name, frame_body):
    """Vertices of a mesh geom in a body's frame."""
    g = gid(m, name)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
    b = bid(m, frame_body)
    return (w - d.xpos[b]) @ d.xmat[b].reshape(3, 3)


def grasp_pose():
    """Arm pose with the fingers horizontal and the jaw closing horizontally, plus a grasp point."""
    m, d, _ = scene.build(0)
    qidx, _ = scene.arm_ids(m)
    best = None
    for roll in (np.pi / 2, -np.pi / 2):
        for wf in np.linspace(-1.6, 1.6, 321):
            q = np.array([0.0, -0.5, 0.6, wf, roll, scene.GRIPPER_RANGE_RAD[1] * 0.85])
            d.qpos[qidx] = q
            mujoco.mj_kinematics(m, d)
            R = d.xmat[bid(m, "arm/gripper")].reshape(3, 3)
            score = abs(R[2, 2]) + abs(R[2, 0])    # finger axis (-z) and jaw axis (x) both horizontal
            if best is None or score < best[0]:
                best = (score, q.copy())
    q = best[1]
    d.qpos[qidx] = q
    mujoco.mj_forward(m, d)
    return q, best[0], finger_verts(m, d, "arm/finger_fixed", "arm/gripper")


def grasp_run(q, fixed_verts, offset, threshold, hold_torque, radius=0.0125, pull=True, z_g=-0.070):
    """Hang one tomato mid-finger, `offset` m off the fixed finger's flank, close, then pull away.

    Closes at the configured gripper torque (50 %), then drops to `hold_torque` for the hold and
    pull: the real servo falls to Overload_Torque 25 % once it has stalled on the fruit."""
    near = fixed_verts[np.abs(fixed_verts[:, 2] - z_g) < 0.003]
    m0, d0, _ = scene.build(0)
    qidx0, _ = scene.arm_ids(m0)
    d0.qpos[qidx0] = q
    mujoco.mj_forward(m0, d0)
    G = bid(m0, "arm/gripper")
    p_g = np.array([near[:, 0].max() + radius + offset, near[:, 1].mean(), z_g])
    centre = d0.xpos[G] + d0.xmat[G].reshape(3, 3) @ p_g
    anchor, body, eq, info = tm.single_xml("gt", centre, radius, threshold)
    m, d = scene.compose(anchor, body, eq, arm=True)
    qidx, uidx = scene.arm_ids(m)
    d.qpos[qidx] = q
    d.ctrl[uidx] = q
    mujoco.mj_forward(m, d)
    det = tm.Detacher(m, d, [info])
    T, Gb = bid(m, "gt"), bid(m, "arm/gripper")
    fing = [gid(m, "arm/finger_fixed"), gid(m, "arm/finger_moving")]
    tg = [gid(m, "gt_g")]
    start_contacts = len(contacts(m, d, fing, tg))

    def in_gripper():
        return (d.xpos[T] - d.xpos[Gb]) @ d.xmat[Gb].reshape(3, 3)

    def run(seconds, fn=None):
        for k in range(int(seconds / scene.TIMESTEP)):
            if fn:
                fn(k * scene.TIMESTEP)
            mujoco.mj_step(m, d)
            det.step()

    run(0.5)
    g_open, g_closed = q[5], scene.GRIPPER_RANGE_RAD[0]
    run(0.5, lambda t: d.ctrl.__setitem__(uidx[5], g_open + (g_closed - g_open) * min(1, t / 0.5)))
    run(0.4)
    m.actuator_forcerange[uidx[5]] = (-hold_torque, hold_torque)   # overload kicks in after the stall
    run(0.6)
    out = dict(start_contacts=start_contacts, closed_peak=det.peak[0],
               plucked_on_close=not np.isnan(det.detached_at[0]),
               both_fingers=len({a if a in fing else b for a, b in contacts(m, d, fing, tg)}) == 2,
               jaw_deg=np.degrees(d.qpos[qidx[5]]))
    if not pull or out["plucked_on_close"]:
        return out
    p_grip = in_gripper()
    e0 = d.ctrl[uidx[2]]
    slip_before = 0.0
    released = False
    t_rel = None
    for k in range(int(2.0 / scene.TIMESTEP)):     # pull: bend the elbow back 0.5 rad over 2 s
        d.ctrl[uidx[2]] = e0 - 0.5 * min(1, k * scene.TIMESTEP / 2.0)
        mujoco.mj_step(m, d)
        if len(det.step()):
            released, t_rel = True, d.time
            break
        slip_before = max(slip_before, np.linalg.norm(in_gripper() - p_grip))
    run(0.5)
    out.update(released=released, force_at=det.force_at[0], slip_before=slip_before,
               slip_after=np.linalg.norm(in_gripper() - p_grip),
               held_after=len(contacts(m, d, fing, tg)) > 0)
    return out


def check_grasp():
    q, score, fixed_verts = grasp_pose()
    report(score < 0.05, f"grasp pose: fingers + jaw axis horizontal (residual {score:.3f}), "
                         f"q = {np.round(np.degrees(q), 1)} deg")
    lo_thr, hi_thr = tm.DETACH_FORCE_N
    # Closing must not pluck, even at the weakest break force and with the fruit off-centre,
    # because the moving finger shoves it toward the fixed one first.
    for offset in (0.003, 0.008):
        r = grasp_run(q, fixed_verts, offset, lo_thr, scene.GRIPPER_TORQUE, pull=False)
        report(r["start_contacts"] == 0, f"offset {offset*1000:.1f} mm: jaw opens clear of the fruit")
        report(not r["plucked_on_close"],
               f"offset {offset*1000:.1f} mm, break {lo_thr:.2f} N: closing does not pluck "
               f"(peak tether {r['closed_peak']:.2f} N)")
        report(r["both_fingers"], f"offset {offset*1000:.1f} mm: both fingers on the fruit, jaw at {r['jaw_deg']:.1f} deg")
    # Pulling must pluck through the tether at the strongest break force, holding at full (50 %) and
    # overload (25 %) gripper torque, without the fruit slipping out.
    for frac, label in ((0.5, "50 %"), (0.25, "25 % (overload)")):
        r = grasp_run(q, fixed_verts, 0.003, hi_thr, frac * 3.35)
        print(f"     jaw at {r['jaw_deg']:.1f} deg while holding")
        if r["plucked_on_close"]:
            report(False, f"torque {label}: plucked while closing")
            continue
        report(r["released"], f"torque {label}, break {hi_thr:.2f} N: pull plucks through the tether "
                              f"(released at {r.get('force_at', float('nan')):.2f} N)")
        report(r["slip_before"] < 0.004, f"torque {label}: fruit slips {r['slip_before']*1000:.1f} mm in the grip before release")
        report(r["held_after"] and r["slip_after"] < 0.006,
               f"torque {label}: still in the fingers 0.5 s after release ({r['slip_after']*1000:.1f} mm from grip point)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 layout seeds instead of 200")
    a = ap.parse_args()
    print("== unit mapping"); check_mapping()
    print("\n== tomato layouts"); check_layouts(20 if a.quick else 200, 10 if a.quick else 50)
    print("\n== full scene + cameras"); check_full_scene((0, 1) if a.quick else (0, 1, 2, 3, 4))
    print("\n== grasp through contact"); check_grasp()
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
