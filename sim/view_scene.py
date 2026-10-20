#!/usr/bin/env python3
"""Watch the sim greenhouse; optionally drive the sim arm from the real leader arm.

  python view_scene.py --cams              3D window + a second window with the claw | birdseye
                                           images exactly as they get recorded: lens model + the
                                           YOLO view (red = every detection, green = the target)
  python view_scene.py --cams --overlay true_box   no YOLO, just the green target box from the sim's truth
  python view_scene.py --leader --cams     the real leader arm drives the sim follower (no recording)
  python view_scene.py --seed 7            another random vine layout

Without --leader, drag the actuator sliders in the viewer's right panel (Control) to move the arm.
Ctrl + right-drag a tomato to pull it by hand. Plucks are printed in the terminal.

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
import target  # noqa: F401  (puts the repo root on sys.path for harvest/)
import tomato as tm
from cameras import SimCameras
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="vine layout seed")
    ap.add_argument("--cams", action="store_true", help="also show the claw + birdseye images")
    ap.add_argument("--leader", action="store_true", help="drive the sim arm from the real leader arm")
    ap.add_argument("--overlay", choices=("yolo", "true_box", "none"), default="yolo",
                    help="what --cams draws (default yolo: the recorded view)")
    a = ap.parse_args()

    m, d, info = scene.build(a.seed)
    qidx, uidx = scene.arm_ids(m)
    det = tm.Detacher(m, d, info["fruit"])
    fm = arm_map.FollowerMap()
    d.ctrl[uidx] = d.qpos[qidx]
    tgt, picked, by_yolo = None, False, False   # picked on the first camera frame, like the plugin does
    print(f"seed {a.seed}: {len(info['fruit'])} tomatoes, ripe: " +
          ", ".join(f"{info['fruit'][k]['tomato']} (breaks at {info['fruit'][k]['threshold']:.2f} N)" for k in info["ripe"]))
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
    box = None
    if a.cams:
        import cv2
        cams = SimCameras(m)
        if a.overlay == "yolo":
            from yolo_client import YoloClient
            yolo = YoloClient()
    cam_dt, next_cam, next_print = 1.0 / 15, 0.0, 0.0   # preview at 15 fps; recording will use 30

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
                            print(f">>> plucked {f['tomato']} ({f['ripeness']}) at {det.force_at[k]:.2f} N"
                                  f"{'  <- the target' if k == tgt else ''}")
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
                    with v.lock():
                        claw, bird = cams.rgb(d, "claw"), cams.rgb(d, "birdseye")
                        ids = cams.ids(d, "birdseye") if not picked and a.overlay != "none" else None
                    dets_b, dets_c = yolo.many([bird, claw]) if yolo else ([], [])
                    if ids is not None:          # the start frame: pick the target, hold it from here on
                        with v.lock():
                            st = target.start_frame(cams, d, m, info, dets_b, ids=ids)
                        tgt, box, picked, by_yolo = st["target"], st["box"], True, st["source"] == "yolo"
                        print(f"target: {st['target_name']} ({st['source']} {st['reason']})")
                    if a.overlay != "none":
                        names = yolo.ready() if yolo else {}
                        bird = ov.draw_overlay(bird, dets_b, names, target=box, label_target=by_yolo)
                        claw = ov.draw_overlay(claw, dets_c, names)
                    cv2.imshow("claw | birdseye (red = YOLO, green = target)", np.hstack([claw, bird])[..., ::-1])
                    cv2.waitKey(1)
                time.sleep(0.002)
    finally:
        if yolo:
            yolo.close()   # also on Ctrl+C: the worker + its /dev/shm frames file go away
    if leader:
        leader.disconnect()


if __name__ == "__main__":
    main()
