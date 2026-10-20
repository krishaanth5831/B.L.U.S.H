"""yolo_client.py: the ripeness YOLO from the lerobot env, via yolo_worker.py in the ripeness env.

    yolo = YoloClient()            # starts the worker; loading the weights takes a few seconds
    dets = yolo(rgb)               # 640x480 RGB uint8 -> [{"box", "cls", "conf"}, ...]
    yolo.close()
"""
import json
import os
import pathlib
import struct
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# The ripeness env's python. Override with BLUSH_RIPENESS_PYTHON if conda lives elsewhere.
RIPENESS_PYTHON = os.environ.get("BLUSH_RIPENESS_PYTHON", str(pathlib.Path.home() / "miniforge3/envs/ripeness/bin/python"))
# ⚠️ unverified for the birdseye: pretrain_v4 is the current default weights (ripeness/models/README),
# trained on claw-distance photos. gate.yaml has no released weights yet.
WEIGHTS = ROOT / "ripeness/models/pretrain_v4.pt"
# Lowest confidence returned. The selection threshold (t_ripe) is applied later, by the caller.
CONF_FLOOR = 0.05


class YoloClient:
    def __init__(self, weights=WEIGHTS, python=RIPENESS_PYTHON, conf=CONF_FLOOR):
        if not pathlib.Path(python).exists():
            raise FileNotFoundError(f"ripeness env python not found at {python} (set BLUSH_RIPENESS_PYTHON)")
        self.weights = pathlib.Path(weights)
        self.p = subprocess.Popen([python, str(HERE / "yolo_worker.py"), str(self.weights), str(conf)],
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
        self.ready()
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        self.p.stdin.write(struct.pack("<II", h, w) + rgb.tobytes())
        self.p.stdin.flush()
        return self._recv()

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()
