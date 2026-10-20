"""yolo_worker.py: runs the ripeness YOLO for the sim, in the `ripeness` env.

The sim runs in the `lerobot` env, which has no ultralytics, and the two envs pin different
torch/opencv (CLAUDE.md). So yolo_client.py starts this script with the ripeness env's python and
talks to it over pipes. Not meant to be run by hand.

  python yolo_worker.py <weights> <conf floor> <frames file>

The frames themselves go through a shared-memory file (/dev/shm, made by the client): ⚠️ pushing
2 x 900 KB through the pipe cost ~4 ms a frame, which with rendering and drawing pushed a frame
past 20 ms, and the record loop has to fit in 33 ms. Protocol, one request at a time:
  in (pipe):  4 bytes n (little-endian uint32), then n x 8 bytes (height, width as uint32);
              the n RGB frames sit back to back at the start of the frames file
  out (pipe): 4 bytes (length, uint32) + JSON: per frame [{"box": [x0,y0,x1,y1], "cls", "conf"}, ...]
The first message out is {"ready": true, "names": {...}} once the weights are loaded. The n frames
go through the network as one batch: both cameras in one forward pass.
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
    buf = sys.stdin.buffer.read(n)       # BufferedReader.read(n) blocks until n bytes or EOF
    return buf if buf is not None and len(buf) == n else None


def main():
    weights, conf, frames_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    from ultralytics import YOLO
    model = YOLO(weights)
    shared = np.memmap(frames_path, dtype=np.uint8, mode="r")
    send({"ready": True, "names": model.names})
    while True:
        head = read_exact(4)
        if head is None:
            return
        (n,) = struct.unpack("<I", head)
        sizes = [struct.unpack("<II", read_exact(8)) for _ in range(n)]
        frames, at = [], 0
        for h, w in sizes:
            rgb = shared[at:at + h * w * 3].reshape(h, w, 3)
            at += h * w * 3
            # ultralytics takes numpy frames as BGR, like cv2 camera frames (live_view.py / pick_target.py).
            frames.append(np.ascontiguousarray(rgb[..., ::-1]))
        # ⚠️ save=False: the global ultralytics settings point runs_dir at a stale folder.
        results = model.predict(frames, conf=conf, verbose=False, save=False)
        send([[{"box": b, "cls": int(c), "conf": float(p)} for b, c, p in
               zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())] for r in results])


if __name__ == "__main__":
    main()
