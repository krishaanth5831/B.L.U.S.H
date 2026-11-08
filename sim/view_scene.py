#!/usr/bin/env python3
"""Watch the sim greenhouse (a truss scene); optionally drive the sim arm from the real leader arm.

  python view_scene.py --cams              3D window + a second window with the policy's two cameras exactly as
                                           they get recorded: lens model + the target truss's green box and
                                           cut-point cross (+ every YOLO detection in red with --overlay yolo)
  python view_scene.py --cams --rig base   the rig "base" cameras (base | claw) and the base camera's bracket
  python view_scene.py --cams --overlay none   raw frames
  python view_scene.py --leader --cams     the real leader arm drives the sim follower (no recording)
  python view_scene.py --seed 7600000      another random truss layout

Without --leader, drag the actuator sliders in the viewer's right panel (Control) to move the arm.
Ctrl + right-drag a tomato to pull it by hand. Plucks and stalk cuts are printed in the terminal.
The target is picked on the first camera frame, as the plugin does (truss_target.start_frame): the truss YOLO
(--yolo-weights, default the generator's runs/yolo_truss/truss_v1 if trained) with the true box filling in, or
the true box alone (--overlay true_box, the default when no truss YOLO is trained).

--leader is the sign check from arm_map.py: move each leader joint on its own and check the sim joint
turns the same way and lands where the real follower would. The leader is read only (no torque,
no calibration written). Writes nothing. Run in the lerobot env.
"""
import argparse
import os
import pathlib
import time

# ⚠️ Camera renders through EGL, not GLFW: a GLFW renderer starting while the viewer's thread starts
# GLFW can abort the process (see the plugin). Before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import mujoco.viewer
import numpy as np

import arm_map
import scene
import tomato as tm
import truss as tr
import truss_scene as ts
import truss_target as ttg  # also puts the repo root on sys.path for harvest/ (keep it before that import)
from cameras import RIG_CAMERAS, SimCameras
from generate_truss import YOLO_TRUSS
from harvest import yolo_overlay as ov

HERE = pathlib.Path(__file__).resolve().parent
HW_ENV = HERE.parent / "config" / "hardware.env"


def hardware_env():
    env = {}
    for line in HW_ENV.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def connect_leader():
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig
    hw = hardware_env()
    port = hw["LEADER_PORT"]
    if not pathlib.Path(port).exists():
        raise SystemExit(f"leader not found at {port} (by-id path from config/hardware.env). Plugged in?")
    # use_degrees=false: the same range-normalised units the real follower gets (CLAUDE.md rule).
    leader = SO101Leader(SOLeaderTeleopConfig(port=port, id=hw["LEADER_ID"], use_degrees=False))
    # ⚠️ calibrate=False: never start lerobot's interactive calibration from here, it overwrites
    # leader_black.json without asking twice.
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        leader.disconnect()
        raise SystemExit(f"leader EEPROM doesn't match {hw['LEADER_ID']}.json. Run ~/teleop.sh once, then retry.")
    return leader


def draw_cross(img, cut):
    """The cut point as the plugin draws it."""
    import cv2
    if cut is not None:
        cv2.drawMarker(img, tuple(int(round(c)) for c in cut), ov.TARGET_COLOUR, cv2.MARKER_CROSS, 10, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="truss layout seed (truss_scene.build)")
    ap.add_argument("--cams", action="store_true", help="also show the policy's two camera images")
    ap.add_argument("--rig", choices=tuple(RIG_CAMERAS), default="birdseye",
                    help="cameras shown (SO101MujocoConfig.rig); base also builds the base camera's bracket")
    ap.add_argument("--leader", action="store_true", help="drive the sim arm from the real leader arm")
    ap.add_argument("--overlay", choices=("yolo", "true_box", "none"), default=None,
                    help="what --cams draws (default: yolo if a truss YOLO is trained, else true_box)")
    ap.add_argument("--yolo-weights", type=pathlib.Path, default=YOLO_TRUSS if YOLO_TRUSS.exists() else None,
                    help="truss YOLO for --overlay yolo (default: runs/yolo_truss/truss_v1 if trained)")
    a = ap.parse_args()
    overlay = a.overlay or ("yolo" if a.yolo_weights else "true_box")
    if overlay == "yolo" and not a.yolo_weights:
        ap.error("--overlay yolo needs --yolo-weights (no truss YOLO at runs/yolo_truss/truss_v1)")

    m, d, info = ts.build(a.seed, base_mount=a.rig == "base")
    qidx, uidx = scene.arm_ids(m)
    det = tm.Detacher(m, d, info["fruit"])
    cutters = [tr.TrussCutter(m, d, t) for t in info["trusses"]]
    fm = arm_map.FollowerMap()
    d.ctrl[uidx] = d.qpos[qidx]
    T = info["trusses"]
    tgt, picked = None, False          # picked on the first camera frame, like the plugin does
    print(f"seed {a.seed}: {len(T)} trusses ({len(info['fruit'])} tomatoes), ready: "
          f"{[T[k]['name'] for k in info['ready']] or 'none'}")
    print(f"follower calibration {fm.path.name} sha256 {fm.sha256[:12]}")

    leader = connect_leader() if a.leader else None
    if leader:
        # Start the sim arm where the leader already is. Otherwise it swings from the start pose
        # to the leader's pose at full gain and knocks fruit off before the first check.
        act = leader.get_action()
        d.qpos[qidx] = fm.norm_to_rad([act[f"{j}.pos"] for j in arm_map.JOINTS])
        d.ctrl[uidx] = d.qpos[qidx]
        mujoco.mj_forward(m, d)
        print("leader connected: move it, the sim arm follows. Close the window to stop.")

    cams = cv2 = yolo = None
    names_ = RIG_CAMERAS[a.rig]        # [0] carries the target box
    box = cut = None
    by_yolo = False
    if a.cams:
        import cv2
        cams = SimCameras(m)
        if overlay == "yolo":
            from yolo_client import YoloClient
            yolo = YoloClient(a.yolo_weights)
    cam_dt, next_cam, next_print = 1.0 / 15, 0.0, 0.0   # preview at 15 fps; recording uses 30

    try:
        with mujoco.viewer.launch_passive(m, d) as v:
            # From behind the robot, looking at the vines over its shoulder: its left and right are yours
            # (crate on the right). Through the gap between row 0's rails.
            v.cam.lookat[:] = (0.42, 0.30, 0.18)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.75, 90, -35
            t0_wall, t0_sim = time.time(), d.time
            while v.is_running():
                if leader:
                    act = leader.get_action()
                    norm = np.array([act[f"{j}.pos"] for j in arm_map.JOINTS])
                    with v.lock():
                        d.ctrl[uidx] = fm.norm_to_rad(norm)
                # Step the sim until it catches up with the wall clock.
                with v.lock():
                    while d.time - t0_sim < time.time() - t0_wall:
                        mujoco.mj_step(m, d)
                        for k in det.step():
                            f = info["fruit"][k]
                            print(f">>> plucked {f['tomato']} ({f['ripeness']}) at {det.force_at[k]:.2f} N")
                        for k, c in enumerate(cutters):
                            if c.step():
                                print(f">>> cut {T[k]['name']} ({'ready' if T[k]['ready'] else 'NOT ready'}) at "
                                      f"{c.peak['squeeze']:.1f} N{'  <- the target' if k == tgt else ''}")
                v.sync()
                now = time.time()
                if leader and now > next_print:
                    next_print = now + 0.5
                    with v.lock():
                        sim = fm.rad_to_norm(d.qpos[qidx])
                    print("leader " + " ".join(f"{x:+6.1f}" for x in norm) + "  | sim " +
                          " ".join(f"{x:+6.1f}" for x in sim), flush=True)
                if cams and now > next_cam:
                    next_cam = now + cam_dt
                    start = not picked and overlay != "none"
                    with v.lock():
                        imgs = {c: cams.rgb(d, c) for c in names_}
                        bird = imgs["birdseye"] if "birdseye" in imgs else cams.rgb(d, "birdseye") if start else None
                        ids = cams.ids(d, "birdseye") if start else None
                    dets = {c: [] for c in ("birdseye", *names_)}
                    run = list(names_) if a.rig == "birdseye" else ["birdseye"]   # rig base: YOLO only picks
                    if yolo is not None and (start or a.rig == "birdseye"):
                        dets.update(zip(run, yolo.many([bird if c == "birdseye" else imgs[c] for c in run])))
                    if start:                    # the start frame: pick the target, hold it from here on
                        with v.lock():
                            st = ttg.start_frame(cams, d, m, info, dets["birdseye"], ids=ids)
                            if overlay == "true_box":
                                st["reason"] = "overlay=true_box"
                            tgt, picked, by_yolo = st["target"], True, st["source"] == "yolo"
                            box, cut = st["box"], st["cut"]
                            if a.rig != "birdseye" and tgt is not None:   # rig base: where the base camera sees it
                                box, cut = ttg.view_target(cams, d, m, T[tgt], names_[0])
                        print(f"target: {st['target_name']} ({st['source']} {st['reason']}), cut point from "
                              f"{st['cut_source']}")
                    if overlay != "none":
                        c0 = names_[0]
                        if a.rig == "birdseye" and yolo is not None:
                            names = yolo.ready()
                            imgs[c0] = draw_cross(ov.draw_overlay(imgs[c0], dets[c0], names, target=box,
                                                                  label_target=by_yolo), cut)
                            imgs["claw"] = ov.draw_overlay(imgs["claw"], dets["claw"], names)
                        elif box is not None:
                            imgs[c0] = draw_cross(ov.draw_overlay(imgs[c0], [], {}, target=box), cut)
                    legend = {"yolo": " (red = YOLO, green = target)", "true_box": " (green = target)", "none": ""}
                    title = " | ".join(names_) + legend[overlay]
                    cv2.imshow(title, np.hstack([imgs[c] for c in names_])[..., ::-1])
                    cv2.waitKey(1)
                time.sleep(0.002)
    finally:
        if yolo:
            yolo.close()   # also on Ctrl+C: the worker + its /dev/shm frames file go away
    if leader:
        leader.disconnect()


if __name__ == "__main__":
    main()
