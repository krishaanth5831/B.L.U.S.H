"""yolo_client.py: the ripeness YOLO from the lerobot env, via yolo_worker.py in the ripeness env.

    yolo = YoloClient()            # starts the worker; loading the weights takes a few seconds
    dets = yolo(rgb)               # 640x480 RGB uint8 -> [{"box", "cls", "conf"}, ...]
    d_bird, d_claw = yolo.many([bird, claw])   # both cameras as one batch
    yolo.close()

Frames go to the worker through a shared-memory file in /dev/shm (see yolo_worker.py for why),
which close() deletes again.
"""
import json
import os
import pathlib
import struct
import subprocess
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# The ripeness env's python. Override with BLUSH_RIPENESS_PYTHON if conda lives elsewhere.
RIPENESS_PYTHON = os.environ.get("BLUSH_RIPENESS_PYTHON", str(pathlib.Path.home() / "miniforge3/envs/ripeness/bin/python"))
# ⚠️ unverified for the birdseye: pretrain_v4 is the current default weights (ripeness/models/README),
# trained on claw-distance photos. gate.yaml has no released weights yet.
WEIGHTS = ROOT / "ripeness/models/pretrain_v4.pt"
# Lowest confidence returned. Drawing (harvest/yolo_overlay.DISPLAY_CONF) and the selection threshold
# (t_ripe) are applied later by the caller; the recall score also looks below them.
CONF_FLOOR = 0.05
MAX_FRAMES, MAX_H, MAX_W = 4, 480, 640     # what the shared frames file holds (config/hardware.env size)


def _sweep_stale(folder):
    """Delete frames files left by YOLO clients whose process is gone. ⚠️ A crash (an abort skips
    close() and __del__) left a 3.7 MB file in /dev/shm, which is RAM."""
    for p in pathlib.Path(folder).glob("blush_yolo_*_*.u8"):
        try:
            pid = int(p.name.split("_")[2])
        except (IndexError, ValueError):
            continue
        if not pathlib.Path(f"/proc/{pid}").exists():
            p.unlink(missing_ok=True)


class YoloClient:
    def __init__(self, weights=WEIGHTS, python=RIPENESS_PYTHON, conf=CONF_FLOOR):
        if not pathlib.Path(python).exists():
            raise FileNotFoundError(f"ripeness env python not found at {python} (set BLUSH_RIPENESS_PYTHON)")
        self.weights = pathlib.Path(weights)
        shm = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        _sweep_stale(shm)
        fd, self.frames_path = tempfile.mkstemp(prefix=f"blush_yolo_{os.getpid()}_", suffix=".u8", dir=shm)
        os.close(fd)
        self.shared = np.memmap(self.frames_path, dtype=np.uint8, mode="w+", shape=(MAX_FRAMES * MAX_H * MAX_W * 3,))
        self.p = subprocess.Popen([python, str(HERE / "yolo_worker.py"), str(self.weights), str(conf), self.frames_path],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(ROOT))
        self.names = None

    def _recv(self):
        head = self.p.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(f"yolo_worker exited (code {self.p.poll()}); its error is above in stderr")
        (n,) = struct.unpack("<I", head)
        return json.loads(self.p.stdout.read(n))

    def ready(self):
        if self.names is None:
            msg = self._recv()
            self.names = {int(k): v for k, v in msg["names"].items()}
        return self.names

    def __call__(self, rgb):
        return self.many([rgb])[0]

    def many(self, frames):
        """Detections for several frames, run as one batch (both cameras in one forward pass)."""
        assert len(frames) <= MAX_FRAMES
        self.ready()
        head, at = [struct.pack("<I", len(frames))], 0
        for rgb in frames:
            h, w = rgb.shape[:2]
            assert rgb.dtype == np.uint8 and h <= MAX_H and w <= MAX_W, (rgb.shape, rgb.dtype)
            self.shared[at:at + h * w * 3] = np.ascontiguousarray(rgb).reshape(-1)
            at += h * w * 3
            head.append(struct.pack("<II", h, w))
        self.p.stdin.write(b"".join(head))   # the frames are in place before the worker hears about them
        self.p.stdin.flush()
        return self._recv()

    def meta(self):
        """What made the boxes: logged per scene, since new weights change the training images."""
        import hashlib
        return dict(weights=self.weights.name, weights_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest())

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        del self.shared
        try:
            os.unlink(self.frames_path)
        except FileNotFoundError:
            pass

    def __del__(self):
        # A crashed lerobot command never calls close(): don't leave the frames file in /dev/shm.
        try:
            if os.path.exists(self.frames_path):
                os.unlink(self.frames_path)
        except Exception:  # noqa: BLE001
            pass
