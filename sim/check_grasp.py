#!/usr/bin/env python3
"""Grasp checks: close on a cherry tomato, pluck it, then hold it while the arm moves like teleop.

  python check_grasp.py                PASS/FAIL per check, exit 1 on a FAIL
  python check_grasp.py --table        + one line of numbers per grasp case
  python check_grasp.py --windowed     watch the grasp cases in the MuJoCo window, slowed down
  python check_grasp.py --windowed --case down,tip,0,12.5    one case: pose,height,width offset mm,radius mm

Each case hangs one tomato on its tether between the open fingers, closes the jaw like the real
servo (50 % torque, dropping to 25 % once stalled), pulls it off the tether, then shakes the arm
for 6 s (wrist flex, pan, wrist roll, ~1 Hz) and measures how far the fruit moved in the gripper.
Cases: fingers sideways or pointing down x grasped near the base / mid / tip x centred or 5 mm
off-centre across the 18 mm finger x 20 / 25 / 30 mm fruit. K7 saw fruit slip out in teleop
(2026-10-20), which the old single mid-finger check never caught.

Writes nothing. Run in the lerobot env.
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time

import mujoco
import numpy as np

import scene
import tomato as tm

FAILS = 0
CHAIN = json.loads((scene.FINRAY / "chain.json").read_text())
HEIGHTS = {"base": -0.055, "mid": -0.072, "tip": -0.090}   # fruit centre below the gripper frame (m)
SHAKE = (("wrist_flex", 0.4, 1.0), ("shoulder_pan", 0.4, 1.0), ("wrist_roll", 0.8, 0.75))   # joint, rad, Hz
SHAKE_S = 2.0                      # per joint
# The fruit may shift this much in the gripper while shaking and still count as held: a third of a
# 25 mm fruit. (Set at 6 mm first, a guess; the only cases above it are a 30 mm fruit gripped hard
# against the palm, which settles ~6.5 mm out to mid-finger and stays. A fruit that slips out moves 100+ mm.)
DRIFT_OK_M = 0.008
JOINT_IDX = {j: k for k, j in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))}
CASES = [(pose, h, oy, r) for pose in ("side", "down") for h in ("base", "mid", "tip")
         for oy in (0.0, 0.005) for r in (0.010, 0.0125, 0.015)]


def report(ok, msg):
    global FAILS
    FAILS += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def gid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def finger_geoms(m):
    """Colliding finger geoms per side (rigid meshes or bending segments)."""
    out = {"fixed": [], "moving": []}
    for i in range(m.ngeom):
        n = m.geom(i).name
        if m.geom_contype[i] and n.startswith(("arm/finger_", "arm/finray_")):
            out["fixed" if "fixed" in n else "moving"].append(i)
    return out


def contacts(m, d, a_ids, b_ids):
    a_ids, b_ids = set(a_ids), set(b_ids)
    return [(c.geom1, c.geom2) for c in d.contact[:d.ncon]
            if (c.geom1 in a_ids and c.geom2 in b_ids) or (c.geom2 in a_ids and c.geom1 in b_ids)]


def flank_x(z_m):
    """x of the fixed finger's contact flank at height z (gripper frame, m), from the print geometry."""
    (x0, z0), (x1, z1) = CHAIN["fixed"]["front_line_gripper_mm"]
    z = z_m * 1000
    return (x0 + (x1 - x0) * (z - z0) / (z1 - z0)) / 1000


def find_pose(kind):
    """Arm pose (rad) with the jaw open: fingers horizontal ("side") or pointing down ("down"),
    jaw closing horizontally either way."""
    m, d, _ = scene.build(0)
    qidx, _ = scene.arm_ids(m)
    G = bid(m, "arm/gripper")
    best = None
    lifts = (-0.5,) if kind == "side" else np.linspace(-1.2, 0.4, 17)
    elbows = (0.6,) if kind == "side" else np.linspace(-0.4, 1.4, 19)
    rolls = (np.pi / 2, -np.pi / 2) if kind == "side" else (0.0, np.pi / 2)
    for lift, elbow, roll in itertools.product(lifts, elbows, rolls):
        for wf in np.linspace(-1.6, 1.6, 161):
            q = np.array([0.0, lift, elbow, wf, roll, scene.GRIPPER_RANGE_RAD[1] * 0.85])
            d.qpos[qidx] = q
            mujoco.mj_kinematics(m, d)
            R = d.xmat[G].reshape(3, 3)
            if kind == "side":
                score = abs(R[2, 2]) + abs(R[2, 0])        # finger axis (-z) and jaw axis (x) both horizontal
            else:
                if d.xpos[G][2] < 0.25:
                    continue                                # keep the hand well above the table
                score = (1 - R[2, 2]) + abs(R[2, 0])       # finger axis straight down, jaw axis horizontal
            if best is None or score < best[0]:
                best = (score, q.copy())
    return best[1], best[0]


def open_viewer(m, d, look_at):
    import mujoco.viewer
    before = set(threading.enumerate())
    v = mujoco.viewer.launch_passive(m, d)
    v._threads = [t for t in set(threading.enumerate()) - before if "_launch_internal" in t.name]
    v.cam.lookat[:] = look_at
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.28, 150, -20
    return v


def close_viewer(v):
    v.close()
    for t in v._threads:              # ⚠️ join the render loop, or the next viewer / exit hangs (plugin note)
        t.join(timeout=5)


def grasp_case(q, height, off_y, radius, threshold, hold_torque, shake=True, off_x=0.003, watch=False, slow=0.5):
    """One grasp: returns its numbers (see the module docstring)."""
    m0, d0, _ = scene.build(0)
    qidx0, _ = scene.arm_ids(m0)
    d0.qpos[qidx0] = q
    mujoco.mj_forward(m0, d0)
    G = bid(m0, "arm/gripper")
    z_g = HEIGHTS[height]
    p_g = np.array([flank_x(z_g) + radius + off_x, CHAIN["fixed"]["y_mid_mm"] / 1000 + off_y, z_g])
    centre = d0.xpos[G] + d0.xmat[G].reshape(3, 3) @ p_g
    anchor, body, eq, info = tm.single_xml("gt", centre, radius, threshold)
    m, d = scene.compose(anchor, body, eq, arm=True)
    qidx, uidx = scene.arm_ids(m)
    d.qpos[qidx] = q
    d.ctrl[uidx] = q
    mujoco.mj_forward(m, d)
    det = tm.Detacher(m, d, [info])
    T, Gb = bid(m, "gt"), bid(m, "arm/gripper")
    fing = finger_geoms(m)
    allf = fing["fixed"] + fing["moving"] + [gid(m, "arm/palm")]
    fruit = gid(m, "gt_g")
    out = dict(start_contacts=len(contacts(m, d, allf, [fruit])))
    pen = [0.0]
    v = open_viewer(m, d, centre) if watch else None

    def in_gripper():
        return (d.xpos[T] - d.xpos[Gb]) @ d.xmat[Gb].reshape(3, 3)

    def step(n=1, fn=None):
        for k in range(n):
            if fn:
                fn(k * scene.TIMESTEP)
            mujoco.mj_step(m, d)
            det.step()
            for c in d.contact[:d.ncon]:
                if (c.geom1 == fruit and c.geom2 in allf) or (c.geom2 == fruit and c.geom1 in allf):
                    pen[0] = max(pen[0], -c.dist)
            if v is not None and k % 8 == 0:
                if not v.is_running():
                    raise KeyboardInterrupt
                v.sync()
                time.sleep(8 * scene.TIMESTEP / slow)

    S = lambda s: int(s / scene.TIMESTEP)   # noqa: E731
    try:
        step(S(0.5))
        g_open, g_closed = q[5], scene.GRIPPER_RANGE_RAD[0]
        step(S(0.5), lambda t: d.ctrl.__setitem__(uidx[5], g_open + (g_closed - g_open) * min(1, t / 0.5)))
        step(S(0.4))
        m.actuator_forcerange[uidx[5]] = (-hold_torque, hold_torque)   # overload kicks in after the stall
        step(S(0.6))
        sides = {s for s in ("fixed", "moving") if contacts(m, d, fing[s], [fruit])}
        out.update(closed_peak=float(det.peak[0]), plucked_on_close=not np.isnan(det.detached_at[0]),
                   both_fingers=len(sides) == 2, jaw_deg=float(np.degrees(d.qpos[qidx[5]])), pen_close=pen[0])
        released, slip = out["plucked_on_close"], 0.0
        if not released:
            p_grip = in_gripper()
            ue = uidx[JOINT_IDX["elbow_flex"]]
            e0 = d.ctrl[ue]
            for k in range(S(2.0)):               # pull: bend the elbow back 0.5 rad over 2 s
                d.ctrl[ue] = e0 - 0.5 * min(1, k * scene.TIMESTEP / 2.0)
                step()
                if not np.isnan(det.detached_at[0]):
                    released = True
                    break
                slip = max(slip, float(np.linalg.norm(in_gripper() - p_grip)))
        # A fruit that came off while the jaw closed on it is a pick too, if it's in the grip: hold-test it.
        out.update(released=released, force_at=float(det.force_at[0]), slip_pull=slip)
        if not released:
            d.eq_active[det.eq_ids[0]] = 0        # not plucked: cut it free so the hold can still be tested
        step(S(0.3))
        p_hold = in_gripper()
        drift = 0.0
        if shake:
            for joint, amp, hz in SHAKE:
                u = uidx[JOINT_IDX[joint]]
                c0 = d.ctrl[u]
                for k in range(S(SHAKE_S)):
                    d.ctrl[u] = c0 + amp * np.sin(2 * np.pi * hz * k * scene.TIMESTEP)
                    step()
                    drift = max(drift, float(np.linalg.norm(in_gripper() - p_hold)))
                d.ctrl[u] = c0
            step(S(0.3))
        held = bool(contacts(m, d, allf, [fruit])) and np.linalg.norm(in_gripper() - p_hold) < 0.02
        out.update(drift=drift, held=held, pen=pen[0])
        return out
    finally:
        if v is not None:
            close_viewer(v)


def row_text(case, r):
    pose, h, oy, rad = case
    head = f"{pose:4s} {h:4s} off {oy * 1000:3.0f} mm  fruit {2 * rad * 1000:2.0f} mm: "
    pull = ("plucked while closing" if r["plucked_on_close"] else
            f"pull {'released' if r['released'] else 'NO RELEASE'} at {r['force_at']:.2f} N  slip {r['slip_pull'] * 1000:4.1f} mm")
    return (head + f"jaw {r['jaw_deg']:5.1f} deg  both {int(r['both_fingers'])}  {pull}  "
            f"shake drift {r['drift'] * 1000:5.1f} mm  held {int(r['held'])}  sinks {r['pen'] * 1000:.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="print the numbers of every grasp case")
    ap.add_argument("--windowed", action="store_true", help="watch grasp cases in the MuJoCo window")
    ap.add_argument("--case", default=None, help="pose,height,width offset mm,radius mm, e.g. down,tip,0,12.5")
    ap.add_argument("--slow", type=float, default=0.5, help="--windowed playback speed (1 = real time)")
    a = ap.parse_args()
    if not a.windowed:
        os.environ.setdefault("MUJOCO_GL", "egl")
    poses = {}
    for kind, tol in (("side", 0.05), ("down", 0.25)):
        # "down" can't be exactly vertical with the pan at 0 and the hand clear of the table; 0.25 is
        # fingers within ~25 deg of straight down, which is how a teleoperator reaches down onto a truss.
        poses[kind], score = find_pose(kind)
        report(score < tol, f"{kind} grasp pose found (residual {score:.3f}), q = {np.round(np.degrees(poses[kind]), 1)} deg")
    lo_thr, hi_thr = tm.DETACH_FORCE_N

    if a.windowed:
        cases = CASES
        if a.case:
            p, h, oy, rr = a.case.split(",")
            cases = [(p, h, float(oy) / 1000, float(rr) / 1000)]
        try:
            for case in cases:
                pose, h, oy, rad = case
                print(f"{pose} {h}, {oy * 1000:.0f} mm off-centre, {2 * rad * 1000:.0f} mm fruit (close the window to stop)")
                print("    " + row_text(case, grasp_case(poses[pose], h, oy, rad, hi_thr, 0.25 * 3.35, watch=True, slow=a.slow)))
        except KeyboardInterrupt:
            pass
        return

    # 0. Snapping shut on nothing (the tips meet) and reopening: the fingers must spring back straight.
    if scene.FINRAY_BENDING:
        m, d, _ = scene.build(0)
        qidx, uidx = scene.arm_ids(m)
        d.qpos[qidx] = poses["side"]
        d.ctrl[uidx] = poses["side"]
        mujoco.mj_forward(m, d)
        hinges = [m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith("arm/finray_")]
        bend_max = 0.0
        for target, secs in ((scene.GRIPPER_RANGE_RAD[0], 1.0), (poses["side"][5], 1.0)):
            d.ctrl[uidx[5]] = target
            for _ in range(int(secs / scene.TIMESTEP)):
                mujoco.mj_step(m, d)
                bend_max = max(bend_max, float(np.abs(d.qpos[hinges]).max()))
        rest = float(np.degrees(np.abs(d.qpos[hinges]).max()))
        report(np.all(np.isfinite(d.qpos)) and rest < 1.0,
               f"jaw snaps shut on nothing (fingers bend up to {np.degrees(bend_max):.1f} deg) and reopens: "
               f"fingers back straight within {rest:.2f} deg, no blow-up")
    # 1. Closing on the most fragile fruit (weakest break force), 3 or 8 mm off the fixed finger: it
    #    either stays on the truss, or comes off INSIDE the grip and stays there. Coming off on the close
    #    is a pick; what must not happen is a fruit knocked off and dropped. (Until 2026-10-20 this
    #    asked for no pluck at all. The wrapping fingers draw the fruit toward the palm and peak at
    #    ~0.7 N on the tether, so the weakest ~5 % of fruit now come off as the jaw closes on them.)
    for off in (0.003, 0.008):
        r = grasp_case(poses["side"], "mid", 0.0, 0.0125, lo_thr, scene.GRIPPER_TORQUE, off_x=off)
        report(r["start_contacts"] == 0, f"side mid, {off * 1000:.0f} mm off the fixed finger: jaw opens clear of the fruit")
        report(not r["plucked_on_close"] or r["held"],
               f"side mid, {off * 1000:.0f} mm off, break {lo_thr:.2f} N: " +
               (f"comes off while closing (peak tether {r['closed_peak']:.2f} N) and stays in the grip"
                if r["plucked_on_close"] else f"stays on the truss while closing (peak tether {r['closed_peak']:.2f} N)"))
        report(r["both_fingers"], f"side mid, {off * 1000:.0f} mm off: both fingers on the fruit, jaw at {r['jaw_deg']:.1f} deg")
    # 2. Every teleop-like case: pluck at the strongest break force, then hold through the shake.
    rows = []
    for case in CASES:
        pose, h, oy, rad = case
        r = grasp_case(poses[pose], h, oy, rad, hi_thr, 0.25 * 3.35)
        rows.append((case, r))
        if a.table:
            print("     " + row_text(case, r))
    ok = [(c, r) for c, r in rows if not r["plucked_on_close"]]
    report(len(ok) == len(rows), f"{len(ok)}/{len(rows)} cases close without plucking at {hi_thr:.2f} N")
    rel = [c for c, r in ok if r["released"]]
    report(len(rel) == len(ok), f"{len(rel)}/{len(ok)} cases pluck through the tether when pulled")
    bad = [c for c, r in ok if not (r["held"] and r["drift"] < DRIFT_OK_M)]
    report(not bad, f"{len(ok) - len(bad)}/{len(ok)} cases hold the fruit through 6 s of teleop-speed shaking "
                    f"(drift < {DRIFT_OK_M * 1000:.0f} mm)" +
           ("" if not bad else "; dropped or slid: " + ", ".join(f"{p} {h} {oy * 1000:.0f}mm {2 * rr * 1000:.0f}mm"
                                                                  for p, h, oy, rr in bad)))
    pen = max((r["pen"] for _, r in ok), default=0.0)
    report(pen < 0.0015, f"fruit sinks at most {pen * 1000:.1f} mm into a finger (soft contacts; < 1.5 mm)")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
