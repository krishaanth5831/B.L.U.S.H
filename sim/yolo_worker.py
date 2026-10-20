"""yolo_worker.py: runs the ripeness YOLO for the sim, in the `ripeness` env.

The sim runs in the `lerobot` env, which has no ultralytics, and the two envs pin different
torch/opencv (CLAUDE.md). So yolo_client.py starts this script with the ripeness env's python and
talks to it over pipes. Not meant to be run by hand.

Protocol, one request at a time:
  in:  8 bytes (height, width as little-endian uint32) + height*width*3 bytes of RGB
  out: 4 bytes (length, uint32) + that many bytes of JSON: [{"box": [x0,y0,x1,y1], "cls", "conf"}, ...]
The first message out is {"ready": true, "names": {...}} once the weights are loaded.
"""
import json
import os
import struct
import sys

PROTO = os.fdopen(os.dup(1), "wb")   # the pipe back to the client
os.dup2(2, 1)                        # ⚠️ anything ultralytics prints goes to stderr, not into the protocol

import numpy as np  # noqa: E402


def send(obj):
    b = json.dumps(obj).encode()
    PROTO.write(struct.pack("<I", len(b)) + b)
    PROTO.flush()


def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    weights, conf = sys.argv[1], float(sys.argv[2])
    from ultralytics import YOLO
    model = YOLO(weights)
    send({"ready": True, "names": model.names})
    while True:
        head = read_exact(8)
        if head is None:
            return
        h, w = struct.unpack("<II", head)
        rgb = np.frombuffer(read_exact(h * w * 3), np.uint8).reshape(h, w, 3)
        # ultralytics takes numpy frames as BGR, like cv2 camera frames (live_view.py / pick_target.py).
        # ⚠️ save=False: the global ultralytics settings point runs_dir at a stale folder.
        r = model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=conf, verbose=False, save=False)[0]
        send([{"box": b, "cls": int(c), "conf": float(p)} for b, c, p in
              zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())])


if __name__ == "__main__":
    main()
