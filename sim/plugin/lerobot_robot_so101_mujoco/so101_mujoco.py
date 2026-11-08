"""The sim follower. It behaves like the real so101_follower towards lerobot.

- Same features: shoulder_pan.pos ... gripper.pos in use_degrees=false units, plus the cameras at 640x480
  (config/hardware.env size): `birdseye` and `claw` (rig="birdseye", the real rig's names), or `base` and `claw`
  (rig="base").
- send_action takes the leader's normalised values and converts them through follower_white.json,
  as the real follower does (sim/arm_map.py).
- Physics runs in its own thread at wall-clock speed, like a real arm moving between calls.
  realtime=False instead steps it exactly 1/fps inside each send_action (scripted episodes).
- Whole-truss harvesting (sim/truss_scene.py, the only task since 2026-11-08): the target is a ready truss, picked
  on the scene's first birdseye frame and then held (sim/truss_target.py), drawn as a green box with a cut-point
  cross; TrussCutter cuts a stalk the blades press hard enough; done = the whole truss in the crate.
- Cameras go through the InnoMaker lens model (sim/lens.py). overlay="yolo" runs the truss YOLO on both cameras
  every frame (rig "birdseye") and draws its view into the frames (harvest/yolo_overlay.py): red boxes + class +
  confidence on every detection. rig="base" draws the box + cross on the base camera from the sim's truth (the
  target is still picked on a birdseye frame that isn't returned).

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
import randomize  # noqa: E402
import scene  # noqa: E402
import tomato as tm  # noqa: E402
import truss as tr  # noqa: E402
import truss_scene as ts  # noqa: E402
import truss_target as ttg  # noqa: E402
from cameras import RIG_CAMERAS, SimCameras  # noqa: E402
from harvest import yolo_overlay as ov  # noqa: E402  (target.py, via truss_target, put the repo root on sys.path)

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
        if config.overlay == "yolo" and not config.yolo_weights:
            raise ValueError("overlay='yolo' needs yolo_weights: the truss YOLO, e.g. runs/yolo_truss/truss_v1/weights/"
                             "best.pt (sim/yolo_truss/train.sh); or overlay='true_box'")
        if config.rig not in RIG_CAMERAS:
            raise ValueError(f"rig={config.rig!r}: one of {tuple(RIG_CAMERAS)}")
        self.config = config
        self.cam_names = RIG_CAMERAS[config.rig]  # what the policy sees, in order; [0] carries the target box
        self.cameras = dict.fromkeys(self.cam_names)   # names only; lerobot-rollout counts them
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
        self._viewer = None
        self.extra = {}                       # per-scene notes a caller adds (scripted plan), into session.json

    # --- features (callable while disconnected) ---
    @property
    def observation_features(self) -> dict:
        ft = {f"{j}.pos": float for j in arm_map.JOINTS}
        ft.update({c: (scene.CAM_H, scene.CAM_W, 3) for c in self.cam_names})
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
    def _build(self, seed, hold=False):
        self.m, self.d, self.info = ts.build(seed, hold=hold, base_mount=self.config.rig == "base")
        self.info["dr"] = randomize.apply(self.m, seed, self.info["fruit"]) if self.config.domain_rand else None
        self.qidx, self.uidx = scene.arm_ids(self.m)
        if self.config.start_jitter_deg > 0:      # own stream: the scene's other draws stay as they were
            import mujoco
            j = np.random.default_rng([seed, 0x517]).uniform(-1, 1, 5) * np.radians(self.config.start_jitter_deg)
            self.d.qpos[self.qidx[:5]] += j
            mujoco.mj_forward(self.m, self.d)
            self.info["start_jitter_deg"] = np.round(np.degrees(j), 2).tolist()
        self.det = tm.Detacher(self.m, self.d, self.info["fruit"])
        # One cutter per truss: a cut on the wrong truss shows up as that truss's cut.
        self.cutters = [tr.TrussCutter(self.m, self.d, t) for t in self.info["trusses"]]
        self.d.ctrl[self.uidx] = self.d.qpos[self.qidx]
        self.extra = {}
        self.seed = seed
        self._gen += 1
        self._snap = self.config.snap_to_first_action
        self._box = None
        self._box_info = None
        self._view = (None, None)            # rig="base": the target's box + cut point on the base camera
        self._tgt_k = None                   # the target truss, picked on the scene's first frame
        self._dropped = False
        self._log = []
        if self.config.settle_s > 0:          # the vines swing for ~1 s after the build
            self._advance(self.d.time + self.config.settle_s, quiet=True)
        self._t_frame = self.d.time           # realtime=False: sim time at the last frame boundary
        T = self.info["trusses"]
        print(f"[so101_mujoco] scene {self._gen}: seed {seed}, {len(T)} trusses ({len(self.info['fruit'])} "
              f"tomatoes), ready: {[T[k]['name'] for k in self.info['ready']]}. "
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
                               "yolo": self._yolo_meta, "domain_rand": self.info.get("dr"),
                               "settle_s": self.config.settle_s,
                               "realtime": self.config.realtime, **self.extra})
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
        # From behind the robot, looking at the vines over its shoulder: its left and right are yours
        # (crate on the right). Through the gap between row 0's rails.
        v.cam.lookat[:] = (0.42, 0.30, 0.18)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.75, 90, -35
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

    def _advance(self, until, quiet=False):
        """Step physics up to sim time `until`: tethers checked after every step, plucks + crate printed.
        The one stepping path for both modes, so a scripted dry run and its recording match exactly."""
        import mujoco
        while self.d.time < until - 1e-9:
            mujoco.mj_step(self.m, self.d)
            for k in self.det.step():
                f = self.info["fruit"][k]
                if not quiet:
                    print(f"[so101_mujoco] plucked {f['tomato']} ({f['ripeness']}) at {self.det.force_at[k]:.2f} N",
                          flush=True)
            for k, c in enumerate(self.cutters):
                if c.step() and not quiet:
                    t = self.info["trusses"][k]
                    print(f"[so101_mujoco] cut {t['name']} ({'ready' if t['ready'] else 'NOT ready'}) at "
                          f"{c.peak['squeeze']:.1f} N{'  <- the target' if k == self._tgt_k else ''}", flush=True)
        if not self._dropped and self._tgt_k is not None and ts.in_crate(self.info, self.d, self.m, self._tgt_k):
            self._dropped = True
            if not quiet:
                print(f"[so101_mujoco] ✅ target truss in the crate (scene {self._gen})", flush=True)

    def new_scene(self, seed, log=True, hold=False):
        """realtime=False: next scene with this seed, now. log=False drops its state log (a dry run).
        hold: nothing ready to pick (truss_scene.build hold), for stay-home episodes."""
        with self._lock:
            self._save_log()
            if self._viewer is not None:
                self._close_viewer(self._viewer)
            self._build(seed, hold=hold)
            self._log_on = log
            if self.config.viewer:
                self._viewer = self._launch_viewer()

    def _run(self):
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
                self._advance(t_sim + (now - t_wall))
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
            # loads in the background; the first frame waits for it
            self._yolo = YoloClient(self.config.yolo_weights)
            self._yolo_meta = dict(self._yolo.meta(), t_ready_birdseye=ttg.T_READY_BIRDSEYE)
        self._log_on = True
        self._build(self.config.seed)
        if self.config.viewer:
            import glfw
            glfw.init()                        # once, on this thread, before the viewer thread starts GLFW
        if self.config.realtime:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="so101_mujoco_physics", daemon=True)
            self._thread.start()
        elif self.config.viewer:
            self._viewer = self._launch_viewer()
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
            if not self.config.realtime:
                self._t_frame += 1.0 / self.config.fps
                self._advance(self._t_frame)
                if self._viewer is not None and self._viewer.is_running():
                    self._viewer.sync()
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
        """The scene's start frame: which ready truss is the target, its box and its cut point."""
        with self._lock:
            if gen != self._gen:               # N was pressed meanwhile: this frame belongs to the old scene
                return
            info = ttg.start_frame(self._cameras(), self.d, self.m, self.info, dets_birdseye, ids=ids)
            if self.config.overlay == "true_box":
                info["reason"] = "overlay=true_box"
            self._box_info, self._box = info, info["box"]
            self._tgt_k = info["target"]
        what = {"yolo": f"YOLO picked it (ready {info['yolo_conf'] or 0:.0%})"}.get(
            info["source"], f"{info['source'].replace('_', ' ')} ({info['reason'].replace('_', ' ')})")
        print(f"[so101_mujoco] scene {gen} target: {info['target_name']} of {len(info['ready'])} ready, {what}, "
              f"cut point from {info['cut_source']}", flush=True)

    def get_observation(self):
        if self.config.rig == "base":
            return self._observation_base()
        with self._lock:
            state = self._present_norm()
            obs = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, state)}
            cams = self._cameras()
            for c in self.cam_names:
                obs[c] = cams.rgb(self.d, c)
            # The start frame: the first one after the arm reached the leader's pose (the snap).
            need_target = self.config.overlay != "none" and self._box_info is None and not self._snap
            ids = cams.ids(self.d, "birdseye") if need_target else None
            gen = self._gen
            # ⚠️ The sim state of THIS frame (YOLO below takes ~11 ms, the physics keeps running).
            row = (self.d.time, time.time(), self.d.qpos.copy(), self.d.eq_active.copy(), state)
        # YOLO runs outside the lock: it must not stall the physics. ~11 ms for both cameras.
        names_ = self.cam_names
        dets = self._yolo.many([obs[c] for c in names_]) if self._yolo is not None else [[] for _ in names_]
        dets = dict(zip(names_, dets))
        if need_target:
            self._pick_target(dets["birdseye"], ids, gen)
        if self.config.overlay == "yolo":
            names = self._yolo.ready()
            obs["birdseye"] = ov.draw_overlay(obs["birdseye"], dets["birdseye"], names, target=self._box,
                                              label_target=bool(self._box_info) and self._box_info["source"] == "yolo")
            obs["claw"] = ov.draw_overlay(obs["claw"], dets["claw"], names)
        elif self.config.overlay == "true_box" and self._box is not None:
            obs["birdseye"] = ov.draw_overlay(obs["birdseye"], [], {}, target=self._box)
        if self.config.overlay != "none" and self._box_info and self._box_info.get("cut"):
            import cv2
            u, v = (int(round(c)) for c in self._box_info["cut"])
            cv2.drawMarker(obs["birdseye"], (u, v), ov.TARGET_COLOUR, cv2.MARKER_CROSS, 10, 2)
        with self._lock:
            if self.config.state_log_dir and gen == self._gen and self._log_on:
                self._log.append(row + (json.dumps(dets) if self._yolo is not None else "",))
        return obs

    def _observation_base(self):
        """rig="base": base + claw frames. On the scene's first frame the birdseye is rendered once more (not returned)
        and the target picked there exactly as in the birdseye rig (truss YOLO, true-box fallback). The chosen truss's
        box + cut point as the BASE camera sees them (sim truth, truss_target.view_target) are fixed then and drawn on
        every base frame. The claw frame stays clean."""
        with self._lock:
            state = self._present_norm()
            obs = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, state)}
            cams = self._cameras()
            for c in self.cam_names:
                obs[c] = cams.rgb(self.d, c)
            need_target = self.config.overlay != "none" and self._box_info is None and not self._snap
            bird = cams.rgb(self.d, "birdseye") if need_target else None
            ids = cams.ids(self.d, "birdseye") if need_target else None
            gen = self._gen
            row = (self.d.time, time.time(), self.d.qpos.copy(), self.d.eq_active.copy(), state)
        dets = []
        if need_target:
            dets = self._yolo.many([bird])[0] if self._yolo is not None else []
            self._pick_target(dets, ids, gen)
            with self._lock:
                if gen == self._gen:
                    k = self._tgt_k
                    self._view = ttg.view_target(cams, self.d, self.m, self.info["trusses"][k], self.cam_names[0]) \
                        if k is not None else (None, None)
        box, cut = self._view if self._box_info is not None else (None, None)
        if self.config.overlay != "none" and box is not None:
            import cv2
            c0 = self.cam_names[0]
            obs[c0] = ov.draw_overlay(obs[c0], [], {}, target=box)
            if cut is not None:
                u, v = (int(round(c)) for c in cut)
                cv2.drawMarker(obs[c0], (u, v), ov.TARGET_COLOUR, cv2.MARKER_CROSS, 10, 2)
        with self._lock:
            if self.config.state_log_dir and gen == self._gen and self._log_on:
                self._log.append(row + (json.dumps({"birdseye": dets}) if need_target else "",))
        return obs

    def env_state(self):
        """The target truss's box + cut point as numbers (truss_target.ENV_NAMES); zeros before the start frame.
        What the generator records as observation.environment_state."""
        if self.config.rig == "base":
            box, cut = self._view if self._box_info is not None else (None, None)
            return ttg.env_vector(box, cut)
        info = self._box_info or {}
        return ttg.env_vector(info.get("box"), info.get("cut"))

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._viewer is not None:
            self._close_viewer(self._viewer)
            self._viewer = None
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
