#!/usr/bin/env python3
"""Cut-and-hold check: one truss's stalk across the fingertips, close, cut, then shake it like teleop.

    python check_truss_cut.py                 PASS/FAIL per stalk size, exit 1 on a FAIL
    python check_truss_cut.py --windowed      watch it in the MuJoCo window, slowed down

The arm holds a fingers-down pose (check_grasp.find_pose), with the truss-harvest fingers (scene.compose
cut_hold=True: rigid PLA fingers, blade + spring pads at the tips). One truss (truss.py, 7 fruit) leaves a stand-in
stem on the gripper's -y side (the blade side), its peduncle crossing the fingertips along gripper y, the
cut zone at the blades. The jaw closes like the expert does (min-jerk, 0.6 s), holds, then the arm shakes
(wrist flex, pan, wrist roll, as check_grasp.py) for 6 s. Checks, per stalk radius:
  - the blades cut it (TrussCutter) before the jaw has finished closing + 0.5 s
  - the pads still hold it: the truss moves less than HOLD_DRIFT_M in the gripper while shaking
  - no fruit came off the truss (tomato.Detacher)
Free space, no greenhouse: this tests the fingertips, not the layout. Writes nothing.
"""
import argparse
import sys
import time

import mujoco
import numpy as np

import check_grasp as cg
import expert
import scene
import tomato as tm
import truss as tr

RADII = (0.0015, 0.002, 0.0025)  # truss.PEDUNCLE_R's ends and middle
HOLD_DRIFT_M = 0.015             # truss origin may move this much in the gripper while shaking (guess: a
                                 # dropped truss moves 100+ mm; a held one swings on its stalk, 5-14 mm, 2026-11-03)
# check_grasp.SHAKE at this fraction of its amplitude. ⚠️ At full amplitude (0.4-0.8 rad at ~1 Hz, made for one
# fruit) the 6-8 fruit truss swung hard enough on its stalk to tear a fruit off its tether (0.58-2.46 N) in 2 of 3
# runs (2026-11-03). The scripted moves are min-jerk and far gentler; generate_truss's judge counts any fruit lost.
SHAKE_SCALE = 0.5
CLOSE_S, HOLD_S = 0.6, 1.0
STEM_R = 0.004
FAILS = 0


def report(ok, msg):
    global FAILS
    FAILS += not ok
    print(("PASS " if ok else "FAIL ") + msg, flush=True)


def build(q, r, knuckle_shift=np.zeros(3), seed=0):
    m0, d0 = scene.compose("", "", "", arm=True, scenery=False, cut_hold=True)
    qidx0, _ = scene.arm_ids(m0)
    d0.qpos[qidx0] = q
    mujoco.mj_forward(m0, d0)
    G = m0.body("arm/gripper").id
    Rg, pg = d0.xmat[G].reshape(3, 3), d0.xpos[G]
    # The stalk crosses the jaw along gripper y at the tip inserts, 1 mm off the fixed pad's face.
    pad = m0.geom("arm/pad_fixed").id
    pad_face_x = ((d0.geom_xpos[pad] - pg) @ Rg)[0] + m0.geom_size[pad][0]
    blade = m0.geom("arm/blade_fixed").id
    yb, zb = ((d0.geom_xpos[blade] - pg) @ Rg)[1:]
    cross = pg + Rg @ [pad_face_x + r + 0.001, yb, zb]
    y = Rg[:, 1]
    cut_at = np.mean(tr.CUT_ZONE_M)
    k0 = cross - y * (STEM_R + cut_at) + knuckle_shift
    rng = np.random.default_rng(seed)
    out = tr.truss_xml("tt", k0, y, rng, "ready", stem_r=STEM_R, n=7, peduncle_len=tr.PEDUNCLE_LEN[0])
    knuckle, body, fruit_xml, eq, excl, info = out
    stem = (f'<geom name="stand_in_stem" type="capsule" fromto="{k0[0] - y[0]*STEM_R:.5f} {k0[1] - y[1]*STEM_R:.5f} '
            f'{k0[2] + 0.06:.5f} {k0[0] - y[0]*STEM_R:.5f} {k0[1] - y[1]*STEM_R:.5f} {k0[2] - 0.06:.5f}" '
            f'size="{STEM_R}" rgba="{tm.STEM_RGBA}" contype="0" conaffinity="0"/>')
    # ⚠️ r is forced: truss_xml draws its own radius; overwrite the two stalk geoms' sizes in the XML text
    body = body.replace(f'size="{info["r"]:.5f}"', f'size="{r:.5f}"').replace(
        f'size="{info["r"] * tr.RACHIS_R_FRACTION:.5f}"', f'size="{r * tr.RACHIS_R_FRACTION:.5f}"')
    info["r"] = r
    m, d = scene.compose(stem + knuckle + body, fruit_xml, eq, arm=True, scenery=False, cut_hold=True, excludes=excl)
    qidx, uidx = scene.arm_ids(m)
    d.qpos[qidx] = q
    d.ctrl[uidx] = q
    mujoco.mj_forward(m, d)
    return m, d, info, cross


def stalk_point_in_gripper(m, d, info, y_g):
    """Where the peduncle axis crosses the gripper's y = y_g plane, in the gripper frame."""
    G = m.body("arm/gripper").id
    Rg, pg = d.xmat[G].reshape(3, 3), d.xpos[G]
    g = m.geom(info["geoms"][0]).id
    a = d.geom_xpos[g] - d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
    b = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
    A, B = (a - pg) @ Rg, (b - pg) @ Rg
    t = (y_g - A[1]) / (B[1] - A[1])
    return A + t * (B - A)


def run(q, r, watch=False):
    jaw_open = np.radians(25.0)
    q = q.copy()
    q[5] = jaw_open
    # Pass 1: let the truss sag on its knuckle with the jaw open, see where the stalk ends up, shift the
    # knuckle so the settled stalk sits where pass 1 aimed it.
    m, d, info, cross = build(q, r)
    G = m.body("arm/gripper").id
    want = (cross - d.xpos[G]) @ d.xmat[G].reshape(3, 3)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    got = stalk_point_in_gripper(m, d, info, want[1])
    shift = d.xmat[G].reshape(3, 3) @ (want - got)
    m, d, info, _ = build(q, r, knuckle_shift=shift)
    qidx, uidx = scene.arm_ids(m)
    G = m.body("arm/gripper").id
    cut = tr.TrussCutter(m, d, info)
    det = tm.Detacher(m, d, info["fruit"])
    v = None
    if watch:
        v = cg.open_viewer(m, d, d.xpos[G])
    T = m.body(info["body"]).id

    trace = []

    def step(n=1):
        for _ in range(n):
            mujoco.mj_step(m, d)
            cut.step()
            det.step()
            if round(d.time / m.opt.timestep) % 10 == 0:
                trace.append((round(d.time, 3), round(cut.squeeze, 1), round(cut.pad_force, 1),
                              round(float(np.degrees(d.qpos[qidx[5]])), 1),
                              np.round(stalk_point_in_gripper(m, d, info, want[1]) * 1000, 1).tolist()))
            if v is not None:
                v.sync()
                time.sleep(m.opt.timestep / 0.5)

    step(int(1.0 / m.opt.timestep))                       # settle, jaw open
    settled = stalk_point_in_gripper(m, d, info, want[1])
    lo, hi = scene.GRIPPER_RANGE_RAD
    n_close = int(CLOSE_S / m.opt.timestep)
    for s in expert.min_jerk(n_close):
        d.ctrl[uidx[5]] = jaw_open + (lo - jaw_open) * s
        step()
    step(int(HOLD_S / m.opt.timestep))
    cut_t = None if cut.cut_t is None else cut.cut_t - (d.time - HOLD_S - CLOSE_S)
    Rg = d.xmat[G].reshape(3, 3)
    in_g0 = (d.xpos[T] - d.xpos[G]) @ Rg
    drift = 0.0
    q0 = d.ctrl[uidx].copy()
    for joint, amp, hz in cg.SHAKE:
        k = cg.JOINT_IDX[joint]
        n = int(cg.SHAKE_S / m.opt.timestep)
        for i in range(n):
            d.ctrl[uidx[k]] = q0[k] + SHAKE_SCALE * amp * np.sin(2 * np.pi * hz * i * m.opt.timestep)
            step()
            drift = max(drift, np.linalg.norm((d.xpos[T] - d.xpos[G]) @ d.xmat[G].reshape(3, 3) - in_g0))
        d.ctrl[uidx] = q0
    if v is not None:
        cg.close_viewer(v)
    dropped = int(np.sum(~np.isnan(det.detached_at)))
    return dict(trace=trace, cut_t=cut_t, squeeze=cut.peak["squeeze"], cut_force=cut.cut_force, pad=cut.pad_force,
                pad_peak=cut.peak["pad_force"], drift=drift, dropped=dropped, settled_mm=np.round(settled * 1000, 1),
                want_mm=np.round(want * 1000, 1), jaw_deg=float(np.degrees(d.qpos[qidx[5]])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--radius", type=float, default=None, help="one stalk radius (m) instead of RADII")
    a = ap.parse_args()
    q, _ = cg.find_pose("down")
    for r in ([a.radius] if a.radius else RADII):
        o = run(q, r, a.windowed)
        tag = f"stalk {r * 2000:.1f} mm"
        print(f"     {tag}: stalk at {o['settled_mm']} mm (aimed {o['want_mm']}), blade squeeze peak "
              f"{o['squeeze']:.1f} N (cut at {o['cut_force']:.1f} N), pads {o['pad']:.1f} N after "
              f"(peak {o['pad_peak']:.1f}), jaw at {o['jaw_deg']:.1f} deg, drift {o['drift'] * 1000:.1f} mm, "
              f"{o['dropped']} fruit off", flush=True)
        report(o["cut_t"] is not None and o["cut_t"] <= CLOSE_S + 0.5,
               f"{tag}: blades cut it" + ("" if o["cut_t"] is None else f" {o['cut_t']:.2f} s into the close"))
        report(o["drift"] <= HOLD_DRIFT_M, f"{tag}: pads hold it while shaking ({o['drift'] * 1000:.1f} mm)")
        report(o["dropped"] == 0, f"{tag}: no fruit came off ({o['dropped']})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
