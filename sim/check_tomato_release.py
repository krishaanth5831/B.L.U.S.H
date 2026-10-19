#!/usr/bin/env python3
"""Step 1 of the sim plan: does the tomato tether hang, release above threshold, and fall free?

  python check_tomato_release.py --windowed   window with a truss of 6 tomatoes. Double-click a
                                              tomato, then Ctrl + right-drag to pull it. The terminal
                                              prints the tether force; a plucked tomato drops.
  python check_tomato_release.py              headless checks, PASS/FAIL per check, exit 1 on a FAIL

Writes nothing. Run in the lerobot env (it has mujoco).
"""
import argparse
import os
import sys
import time

import numpy as np

import tomato as tm

TIMESTEP = 0.002   # 500 Hz, same as the Vinea FR5 sim (unverified for the SO-101 scene yet)
G = 9.81


def scene(n=1, seed=0, thresholds=None):
    rng = np.random.default_rng(seed)
    rachis, tomatoes, eq, fruit = tm.truss_xml("tr", (-0.06, 0, 0.40), (0.06, 0, 0.40), n, rng)
    body = rachis + tomatoes
    if thresholds is not None:
        for f, t in zip(fruit, thresholds):
            f["threshold"] = t
    xml = f"""<mujoco model="tomato_release">
  <option timestep="{TIMESTEP}" gravity="0 0 -{G}"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -0.5 1" dir="0 0.5 -1"/>
    <geom type="plane" size="0.5 0.5 0.01" rgba="0.8 0.8 0.75 1"/>
    {body}
  </worldbody>
  <equality>{eq}</equality>
</mujoco>"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, fruit


def bid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def run_checks():
    import mujoco
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg)

    # 1. Hanging: 3 s under gravity only. No release, force ~ weight, fruit settles.
    m, d, fruit = scene(n=1)
    det = tm.Detacher(m, d, fruit)
    b = bid(m, fruit[0]["tomato"])
    mass = m.body_mass[b]
    for _ in range(int(3.0 / TIMESTEP)):
        mujoco.mj_step(m, d); det.step()
    p0 = d.xpos[b].copy()
    for _ in range(int(0.5 / TIMESTEP)):
        mujoco.mj_step(m, d); det.step()
    drift = np.linalg.norm(d.xpos[b] - p0) * 1000
    report(np.isnan(det.detached_at[0]), f"hangs 3.5 s without releasing (threshold {fruit[0]['threshold']:.2f} N)")
    report(abs(det.force[0] - mass * G) < 0.1 * mass * G,
           f"tether force at rest {det.force[0]:.3f} N vs weight {mass*G:.3f} N ({mass*1000:.1f} g)")
    report(drift < 0.5, f"settled: moves {drift:.3f} mm over the last 0.5 s")
    report(det.force[0] < tm.DETACH_FORCE_N[0], "fruit weight is below the lowest break force")

    # 2. Ramp pull in several directions and thresholds. Release must land just above threshold,
    #    then the fruit must fall like a free body.
    ramp = 2.0   # N/s
    for direction, label in (((1, 0, 0), "sideways"), ((0, 0, -1), "down"), ((0.3, -0.7, 0.65), "up-out")):
        for thr in (tm.DETACH_FORCE_N[0], 1.5, tm.DETACH_FORCE_N[1]):
            m, d, fruit = scene(n=1, thresholds=[thr])
            det = tm.Detacher(m, d, fruit)
            b = bid(m, fruit[0]["tomato"])
            u = np.array(direction, float); u /= np.linalg.norm(u)
            for _ in range(int(1.0 / TIMESTEP)):         # settle first
                mujoco.mj_step(m, d); det.step()
            t0 = d.time
            released = False
            while d.time - t0 < 3.0 and not released:
                d.xfrc_applied[b, :3] = u * ramp * (d.time - t0)
                mujoco.mj_step(m, d)
                released = len(det.step()) > 0
            d.xfrc_applied[b] = 0
            if not released:
                report(False, f"{label} pull, threshold {thr:.2f} N: never released")
                continue
            # Force at release vs threshold: allow the hold time's worth of ramp plus the weight
            # component and solver lag.
            slack = ramp * tm.DETACH_HOLD_S + m.body_mass[b] * G + 0.05
            f_rel = det.force_at[0]
            report(thr < f_rel < thr + slack,
                   f"{label} pull, threshold {thr:.2f} N: released at {f_rel:.2f} N (t={det.detached_at[0]-t0:.2f} s)")
            # Free fall check: 0.1 s after release the fruit has dropped about g t^2 / 2 relative
            # to where it would be with its release velocity. The floor is 0.36 m below, so no hit yet.
            # Free joint dofs = world linear velocity (3), then angular (3).
            z0, vz0 = d.xpos[b, 2], d.qvel[m.jnt_dofadr[m.body_jntadr[b]] + 2]
            for _ in range(int(0.1 / TIMESTEP)):
                mujoco.mj_step(m, d); det.step()
            drop = z0 + vz0 * 0.1 - d.xpos[b, 2]
            report(abs(drop - 0.5 * G * 0.01) < 0.01,
                   f"{label} pull, threshold {thr:.2f} N: falls free ({drop*1000:.1f} mm vs {0.5*G*0.01*1000:.1f} mm)")

    # 3. A spike over threshold shorter than the hold time must not pluck.
    m, d, fruit = scene(n=1, thresholds=[1.0])
    det = tm.Detacher(m, d, fruit)
    b = bid(m, fruit[0]["tomato"])
    for _ in range(int(1.0 / TIMESTEP)):
        mujoco.mj_step(m, d); det.step()
    spike_steps = max(1, det.hold_steps // 3)
    for k in range(int(0.5 / TIMESTEP)):
        d.xfrc_applied[b, :3] = (3.0, 0, 0) if k < spike_steps else (0, 0, 0)
        mujoco.mj_step(m, d); det.step()
    report(np.isnan(det.detached_at[0]),
           f"a {spike_steps*TIMESTEP*1000:.0f} ms 3 N spike (hold {det.hold_steps*TIMESTEP*1000:.0f} ms) does not pluck")

    # 4. Neighbours: plucking one tomato off a 6-tomato truss leaves the other five on.
    m, d, fruit = scene(n=6, seed=1, thresholds=[1.0] * 6)
    det = tm.Detacher(m, d, fruit)
    b = bid(m, fruit[2]["tomato"])
    for _ in range(int(1.0 / TIMESTEP)):
        mujoco.mj_step(m, d); det.step()
    for _ in range(int(2.0 / TIMESTEP)):
        d.xfrc_applied[b, :3] = (0, 2.0, 0) if np.isnan(det.detached_at[2]) else (0, 0, 0)
        mujoco.mj_step(m, d); det.step()
    others = [k for k in range(6) if k != 2]
    report(not np.isnan(det.detached_at[2]) and np.all(np.isnan(det.detached_at[others])),
           "plucking tomato 2 of 6 leaves the other five hanging")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def windowed(seed):
    import mujoco
    import mujoco.viewer
    m, d, fruit = scene(n=6, seed=seed)
    det = tm.Detacher(m, d, fruit)
    for k, f in enumerate(fruit):
        print(f"tomato {k}: {f['ripeness']:<8} r={f['radius']*1000:.1f} mm  breaks at {f['threshold']:.2f} N")
    print("\nDouble-click a tomato, then Ctrl + right-drag to pull it.\n")
    last_print = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (0, 0, 0.37); v.cam.distance = 0.35; v.cam.elevation = -10; v.cam.azimuth = 90
        while v.is_running():
            t_wall = time.time()
            with v.lock():
                mujoco.mj_step(m, d)
                for k in det.step():
                    print(f"\n>>> tomato {k} plucked at {det.force_at[k]:.2f} N (threshold {det.thresholds[k]:.2f} N)")
            if t_wall - last_print > 0.2:
                last_print = t_wall
                live = ["--" if not d.eq_active[e] else f"{f:.2f}" for e, f in zip(det.eq_ids, det.force)]
                print("\rtether force (N): " + "  ".join(live) + "   ", end="", flush=True)
            v.sync()
            dt = TIMESTEP - (time.time() - t_wall)
            if dt > 0:
                time.sleep(dt)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true", help="interactive window, pluck with the mouse")
    ap.add_argument("--seed", type=int, default=0, help="truss randomisation seed (windowed)")
    a = ap.parse_args()
    if a.windowed:
        windowed(a.seed)
    else:
        # ⚠️ Only force offscreen GL when not watching: setting it unconditionally kills the viewer.
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.exit(1 if run_checks() else 0)


if __name__ == "__main__":
    main()
