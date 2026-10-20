"""The sim follower. It behaves like the real so101_follower towards lerobot.

- Same features: shoulder_pan.pos ... gripper.pos in use_degrees=false units, plus the cameras
  `birdseye` and `claw` at 640x480 (config/hardware.env names and size).
- send_action takes the leader's normalised values and converts them through follower_white.json,
  as the real follower does (sim/arm_map.py).
- Physics runs in its own thread at wall-clock speed, like a real arm moving between calls.
- Cameras go through the InnoMaker lens model (sim/lens.py). overlay="yolo" runs the ripeness YOLO
  on both cameras every frame and draws its view into the frames (harvest/yolo_overlay.py): red
  boxes + class + confidence on every detection, and on the birdseye a green box on the target, which
  is picked among the scene's 2-3 ripe fruits on its first frame and then held (sim/target.py).

MuJoCo window keys: N = new vine layout (next seed). Press it during lerobot-record's reset period.
Ctrl + right-drag pulls a tomato by hand. Plucks and crate drops are printed in the terminal.
"""
import json
import logging
import os
import pathlib
import sys
import threading
import time

# ⚠️ Camera renders go through EGL (headless), never GLFW: the MuJoCo window's thread and a GLFW
# camera renderer starting at the same moment tripped a GLFW assertion and aborted the process
# (1 run in 2, 2026-10-20). Must be set before mujoco is first imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from lerobot.robots.robot import Robot  # noqa: E402

from .config_so101_mujoco import SO101MujocoConfig  # noqa: E402

SIM_DIR = pathlib.Path(__file__).resolve().parents[2]   # B.L.U.S.H/sim
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import arm_map  # noqa: E402
import scene  # noqa: E402
import target as tg  # noqa: E402
import tomato as tm  # noqa: E402
from cameras import CAMERAS, SimCameras  # noqa: E402
from harvest import yolo_overlay as ov  # noqa: E402  (target.py put the repo root on sys.path)

logger = logging.getLogger(__name__)
OVERLAYS = ("none", "true_box", "yolo")
KEY_N = 78                       # GLFW key code
MAX_CATCH_UP_S = 0.1             # after a stall (a slow policy call), skip ahead instead of fast-forwarding


class SO101Mujoco(Robot):
    config_class = SO101MujocoConfig
    # ⚠️ The real follower's robot_type. lerobot-edit-dataset merge refuses datasets with different
    # robot_types, and sim + real must merge for co-training. Sim vs real is kept in the dataset name.
    name = "so_follower"

    def __init__(self, config: SO101MujocoConfig):
        super().__init__(config)
        if config.use_degrees:
            raise ValueError("so101_mujoco only speaks use_degrees=false units (see sim/arm_map.py)")
        if config.overlay not in OVERLAYS:
            raise ValueError(f"overlay={config.overlay!r}: one of {OVERLAYS}")
        self.config = config
        self.cameras = dict.fromkeys(CAMERAS)   # names only; lerobot-rollout counts them
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()
        self._want_new = threading.Event()
        self._connected = False
        self._cams = None
        self._cams_key = None
        self._yolo = None
        self._gen = 0
        self._session = time.strftime("%Y%m%d_%H%M%S")

    # --- features (callable while disconnected) ---
    @property
    def observation_features(self) -> dict:
        ft = {f"{j}.pos": float for j in arm_map.JOINTS}
        ft.update({c: (scene.CAM_H, scene.CAM_W, 3) for c in CAMERAS})
        return ft

    @property
    def action_features(self) -> dict:
        return {f"{j}.pos": float for j in arm_map.JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True                          # nothing to calibrate: the units come from follower_white.json

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # --- scene ---
    def _build(self, seed):
        self.m, self.d, self.info = scene.build(seed)
        self.qidx, self.uidx = scene.arm_ids(self.m)
        self.det = tm.Detacher(self.m, self.d, self.info["fruit"])
        self.d.ctrl[self.uidx] = self.d.qpos[self.qidx]
        self.seed = seed
        self._gen += 1
        self._snap = self.config.snap_to_first_action
        self._box = None
        self._box_info = None
        self._tgt_k = None                   # the target fruit, picked on the scene's first frame
        self._tgt_body = None
        self._dropped = False
        self._log = []
        breaks = ", ".join(f"{self.info['fruit'][k]['threshold']:.2f}" for k in self.info["ripe"])
        print(f"[so101_mujoco] scene {self._gen}: seed {seed}, {len(self.info['fruit'])} tomatoes, "
              f"{len(self.info['ripe'])} ripe (break at {breaks} N). "
              f"Press N in the MuJoCo window for a new layout", flush=True)

    def _save_log(self):
        if not self.config.state_log_dir or not self._log:
            return
        out = pathlib.Path(self.config.state_log_dir).expanduser() / self._session
        out.mkdir(parents=True, exist_ok=True)
        rows = self._log
        np.savez_compressed(out / f"scene{self._gen:03d}_seed{self.seed}.npz",
                            t_sim=np.array([r[0] for r in rows]), t_wall=np.array([r[1] for r in rows]),
                            qpos=np.array([r[2] for r in rows]), eq_active=np.array([r[3] for r in rows]),
                            state=np.array([r[4] for r in rows]),
                            # Every frame's raw YOLO output per camera (JSON), so the drawn view can be
                            # audited or redrawn exactly; empty when overlay != "yolo".
                            dets_json=np.array([r[5] for r in rows]))
        meta = out / "session.json"
        info = json.loads(meta.read_text()) if meta.exists() else {"scenes": []}
        info.update(self.fm.meta())
        # The target + box per scene, so a re-render (step 6) redraws exactly the same one.
        info["scenes"].append({"gen": self._gen, "seed": self.seed, "frames": len(rows),
                               "target": self._box_info and self._box_info["target_name"],
                               "dropped_in_crate": self._dropped, "overlay": self.config.overlay,
                               "display_conf": ov.DISPLAY_CONF, "box": self._box_info,
                               "yolo": self._yolo_meta})
        meta.write_text(json.dumps(info, indent=2))
        self._log = []

    # --- physics thread ---
    def _launch_viewer(self):
        if not self.config.viewer:
            return None
        import mujoco.viewer
        before = set(threading.enumerate())
        v = mujoco.viewer.launch_passive(self.m, self.d, key_callback=self._key)
        # The viewer's render-loop thread (launch_passive doesn't hand it out), for _close_viewer.
        v._blush_threads = [t for t in set(threading.enumerate()) - before if "_launch_internal" in t.name]
        v.cam.lookat[:] = (0.38, 0.22, 0.25)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.95, 215, -22
        return v

    @staticmethod
    def _close_viewer(v):
        # ⚠️ close() only asks the viewer's render loop to stop, and is_running() turns false before
        # it has. A new viewer opened next to a still-running loop, or the process exiting while it
        # runs (glfw.terminate at exit), hangs forever (seen 2026-10-20). Join the thread itself.
        v.close()
        for t in getattr(v, "_blush_threads", []):
            t.join(timeout=5)

    def _key(self, keycode):
        if keycode == KEY_N:
            self._want_new.set()

    def _run(self):
        import mujoco
        v = self._launch_viewer()
        t_wall, t_sim = time.time(), self.d.time
        next_sync = 0.0
        while not self._stop.is_set():
            if self._want_new.is_set():
                self._want_new.clear()
                if v is not None:
                    self._close_viewer(v)
                with self._lock:
                    self._save_log()
                    self._build(self.seed + 1)
                v = self._launch_viewer()
                t_wall, t_sim = time.time(), self.d.time
            now = time.time()
            with self._lock:
                if now - t_wall > (self.d.time - t_sim) + MAX_CATCH_UP_S:
                    t_sim = self.d.time - (now - t_wall) + MAX_CATCH_UP_S
                while self.d.time - t_sim < now - t_wall:
                    mujoco.mj_step(self.m, self.d)
                    for k in self.det.step():
                        f = self.info["fruit"][k]
                        print(f"[so101_mujoco] plucked {f['tomato']} ({f['ripeness']}) at {self.det.force_at[k]:.2f} N"
                              f"{'  <- the target' if k == self._tgt_k else ''}", flush=True)
                if (not self._dropped and self._tgt_body is not None
                        and scene.in_crate(self.d.xpos[self._tgt_body])):
                    self._dropped = True
                    print(f"[so101_mujoco] ✅ target in the crate (scene {self._gen})", flush=True)
            if v is not None and now > next_sync:
                next_sync = now + 1 / 60
                if v.is_running():
                    v.sync()
            time.sleep(0.001)
        if v is not None:
            self._close_viewer(v)

    # --- lerobot API ---
    def connect(self, calibrate: bool = True) -> None:
        self.fm = arm_map.FollowerMap()
        self._yolo_meta = None
        if self.config.overlay == "yolo":
            from yolo_client import YoloClient
            self._yolo = YoloClient()          # loads in the background; the first frame waits for it
            self._yolo_meta = dict(self._yolo.meta(), t_ripe_birdseye=tg.T_RIPE_BIRDSEYE)
        self._build(self.config.seed)
        if self.config.viewer:
            import glfw
            glfw.init()                        # once, on this thread, before the viewer thread starts GLFW
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="so101_mujoco_physics", daemon=True)
        self._thread.start()
        self._connected = True
        logger.info(f"{self} connected (follower calibration sha256 {self.fm.sha256[:12]}).")

    def _present_norm(self):
        return self.fm.rad_to_norm(self.d.qpos[self.qidx])

    def send_action(self, action):
        goal = np.array([action[f"{j}.pos"] for j in arm_map.JOINTS], float)
        with self._lock:
            if self.config.max_relative_target is not None and not self._snap:
                cur = self._present_norm()
                goal = cur + np.clip(goal - cur, -self.config.max_relative_target, self.config.max_relative_target)
            q = self.fm.norm_to_rad(goal)
            if self._snap:
                import mujoco
                self.d.qpos[self.qidx] = q
                self.d.qvel[:] = 0
                mujoco.mj_forward(self.m, self.d)
                self._snap = False
            self.d.ctrl[self.uidx] = q
        return {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, goal)}

    def _cameras(self):
        key = (self._gen, threading.get_ident())
        if self._cams_key != key:              # ⚠️ a renderer is bound to one model and one GL thread
            if self._cams is not None:
                self._cams.close()
            self._cams = SimCameras(self.m)
            self._cams_key = key
        return self._cams

    def _pick_target(self, dets_birdseye, ids, gen):
        """The scene's start frame: which ripe fruit is the target, and its box."""
        with self._lock:
            if gen != self._gen:               # N was pressed meanwhile: this frame belongs to the old scene
                return
            info = tg.start_frame(self._cameras(), self.d, self.m, self.info, dets_birdseye, ids=ids)
            if self.config.overlay == "true_box":
                info["reason"] = "overlay=true_box"
            self._box_info, self._box = info, info["box"]
            self._tgt_k = info["target"]
            self._tgt_body = None if self._tgt_k is None else self.m.body(info["target_name"]).id
        why = info["reason"].replace("_", " ")
        if info["source"] == "yolo":
            what = f"YOLO picked it (ripe {info['yolo_conf']:.0%}, IoU {info['yolo_iou']} with its true box)"
        elif info["source"] == "true_box":
            what = f"true box ({why})"
        elif info["source"] == "true_box_hidden":
            what = f"true box around a hidden ripe fruit ({why})"
        else:
            what = f"no box ({why})"
        print(f"[so101_mujoco] scene {gen} target: {info['target_name']} of {len(info['ripe'])} ripe, {what}",
              flush=True)

    def get_observation(self):
        with self._lock:
            state = self._present_norm()
            obs = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, state)}
            cams = self._cameras()
            for c in CAMERAS:
                obs[c] = cams.rgb(self.d, c)
            # The start frame: the first one after the arm reached the leader's pose (the snap).
            need_target = self.config.overlay != "none" and self._box_info is None and not self._snap
            ids = cams.ids(self.d, "birdseye") if need_target else None
            gen = self._gen
            # ⚠️ The sim state of THIS frame (YOLO below takes ~11 ms, the physics keeps running).
            row = (self.d.time, time.time(), self.d.qpos.copy(), self.d.eq_active.copy(), state)
        # YOLO runs outside the lock: it must not stall the physics. ~11 ms for both cameras.
        dets = self._yolo.many([obs[c] for c in CAMERAS]) if self._yolo is not None else [[] for _ in CAMERAS]
        dets = dict(zip(CAMERAS, dets))
        if need_target:
            self._pick_target(dets["birdseye"], ids, gen)
        if self.config.overlay == "yolo":
            names = self._yolo.ready()
            obs["birdseye"] = ov.draw_overlay(obs["birdseye"], dets["birdseye"], names, target=self._box,
                                              label_target=bool(self._box_info) and self._box_info["source"] == "yolo")
            obs["claw"] = ov.draw_overlay(obs["claw"], dets["claw"], names)
        elif self.config.overlay == "true_box" and self._box is not None:
            obs["birdseye"] = ov.draw_overlay(obs["birdseye"], [], {}, target=self._box)
        with self._lock:
            if self.config.state_log_dir and gen == self._gen:
                self._log.append(row + (json.dumps(dets) if self._yolo is not None else "",))
        return obs

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            self._save_log()
            if self._cams is not None:
                self._cams.close()
                self._cams = None
        if self._yolo is not None:
            self._yolo.close()
            self._yolo = None
        self._connected = False
        logger.info(f"{self} disconnected.")
