#!/usr/bin/env python3
"""How a dataset's camera pictures are stored, and the same step for a live picture: ONE place for both, so that
training and every test see the same pictures (v5 post-mortem fix 1; vault note 4.5, 5.6 and 7.1).

    python picture_store.py selfcheck             # the encoder's property on raw test pictures (no simulator)
    python picture_store.py gatecheck             # the gate itself, on two throwaway datasets of those pictures
    python picture_store.py gate <repo_id>        # a recorded dataset against the recorder's fingerprints

What went wrong in v5: the policy was trained on pictures decoded from the dataset's AV1 video (LeRobot's default,
a key frame every 2 frames), every test fed raw renders. A stored frame was not a function of its own picture: it
lost detail AND carried a trace of the frame after it. At the start pose that texture was the whole claw picture,
and the raw picture threw the first plan off (6.8 deg short).

The rule (note 5.6): a stored frame may depend on its own picture only, and a test passes every live picture through
the same step. So:
    encoder()            the dataset's video setting (STORE). Every recorder passes it to LeRobotDataset.create.
    through_encoder(img) a live picture -> what the dataset would have stored for it (a one-frame clip, decoded).
                         eval_truss.py --through-encoder and the real arm's test loop call this.
    Gate                 the recorder notes a fingerprint of through_encoder(raw picture) for GATE_FRAMES of every
                         episode and camera (meta/picture_gate.jsonl in the dataset).
    gate(repo_id)        reads those stored frames back the way training does and compares: every one must match to
                         the grey level. Run it on every new dataset BEFORE training.

⚠️ Traps (all measured, selfcheck prints them):
  - LeRobot's default (libsvtav1, g 2) fails the rule.
  - AV1 needs g 1 AND `svtav1-params pred-struct=1`; RGBEncoderConfig drops an extra `svtav1-params` silently
    unless fast_decode is None, and drops it again as soon as encoder_threads is set (configs/video.py
    get_codec_options: "extra_options ... never override a structured field"). So AV1 is one forgotten argument
    away from the v5 defect, and H.264 with a constant quantiser is used: its options survive both cases.
  - H.264 in its usual crf mode lets earlier frames change a stored frame; constant qp does not.
lerobot-mj4.
"""
import argparse
import hashlib
import itertools
import json
import os
import pathlib
import tempfile

import numpy as np

# The dataset's picture storage: H.264 (libx264), every frame a key frame (g 1), a constant quantiser. No trace of
# other frames, and a live picture sent through alone gives exactly the stored frame (post-mortem encoder test
# 2026-11-20 at qp 18; selfcheck re-measures it through LeRobot's own encode call).
# selfcheck 2026-11-26, 6 raw start pictures (stored = alone | grey levels lost | kB per frame):
#     AV1, LeRobot's default (v5)                       NO  | 1.94 | 10.0
#     AV1 g 1 + pred-struct=1 as an extra option        NO  | 1.78 | 12.4   (the option is dropped silently)
#     the same with fast_decode=None                    yes | 1.61 | 20.8
#     the same with fast_decode=None and 4 threads      NO  | 1.78 | 12.4   (dropped again)
#     H.264 g 1 crf 23                                  NO  | 2.13 | 10.8
#     H.264 g 1 qp 18 (also with 4 threads)             yes | 1.61 | 25.0
#     H.264 g 1 qp 23                                   yes | 1.70 | 18.1
#     H.264 g 1 qp 26                                   yes | 1.78 | 14.6
# (about 1.6 of the loss is the colour format, yuv420p, which every row shares.) pyav and torchcodec decode it to the
# same grey levels (largest difference 0).
# qp 23: any constant qp keeps the rule, so the choice is size against detail. 1,400 episodes x 3 cameras x ~480
# frames is ~2.0 M frames: ~36 GB at qp 23 on these start pictures (frames with the arm in view ran ~1.3x bigger in
# the post-mortem's test), ~50 GB at qp 18, ~29 GB at qp 26; v5 was 6.6 GB for 2 cameras. ⚠️ Change it BEFORE a
# recording starts, never between the parts of one dataset (the gate would still pass; the pictures would differ).
STORE = dict(vcodec="h264", g=1, crf=None, preset="medium", extra_options={"qp": 23})
GATE_FRAMES = (0, 30)     # per episode and camera: the start frame (where v5 broke) and one in motion (1 s in)
GATE_FILE = "meta/picture_gate.jsonl"
FPS = 30                  # config/hardware.env; only the clip's time base, a one-frame clip has no motion


def encoder():
    """The RGBEncoderConfig every recorder uses. Refuses to return a setting whose options did not arrive."""
    from lerobot.configs.video import RGBEncoderConfig
    cfg = RGBEncoderConfig(**STORE)
    for threads in (None, 4):
        o = cfg.get_codec_options(encoder_threads=threads, as_strings=True)
        want = {"g": str(STORE["g"]), **{k: str(v) for k, v in STORE["extra_options"].items()}}
        bad = {k: (o.get(k), v) for k, v in want.items() if o.get(k) != v}
        if bad or "crf" in o:
            raise RuntimeError(f"picture_store: the encoder options did not arrive (threads {threads}): {bad}, {o}")
    return cfg


_N = itertools.count()      # unique video file names (see through_encoder)


def backend():
    """The decoder training reads the dataset with (lerobot's default), so a test picture goes the same way."""
    from lerobot.utils.import_utils import get_safe_default_video_backend
    return get_safe_default_video_backend()


def _encode(frames, path, cfg=None, threads=None):
    """RGB uint8 pictures -> one video file, through LeRobot's own encode call (PNG files, as the recorder)."""
    import cv2
    from lerobot.datasets.video_utils import encode_video_frames
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        d = pathlib.Path(td)
        for i, img in enumerate(frames):
            cv2.imwrite(str(d / f"frame-{i:06d}.png"), cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        encode_video_frames(d, path, FPS, video_encoder=cfg or encoder(), encoder_threads=threads, overwrite=True,
                            log_level=None)


def _decode(path, n, how=None):
    from lerobot.datasets.video_utils import decode_video_frames
    fr = decode_video_frames(str(path), [k / FPS for k in range(n)], 1e-3, backend=how or backend(),
                             return_uint8=True)
    return [f.permute(1, 2, 0).numpy() for f in fr]


def through_encoder(img, cfg=None, how=None, threads=None):
    """A live RGB uint8 picture -> the picture the dataset stores for it (encoded alone, decoded as training does).
    ⚠️ A new file name every call: lerobot's torchcodec decoder is cached by PATH, so a second video written to the
    same path is answered with the first one's frames (selfcheck's first version compared nonsense that way)."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        v = pathlib.Path(td) / f"one_{next(_N)}.mp4"
        _encode([img], v, cfg, threads)
        return _decode(v, 1, how)[0]


def fingerprint(img):
    return hashlib.sha1(np.ascontiguousarray(img, dtype=np.uint8).tobytes()).hexdigest()


class Gate:
    """The recorder's side: note(obs) once per recorded frame, then keep(root, episode) or drop()."""

    def __init__(self, cams):
        self.cams, self.k, self.rows = tuple(cams), 0, {}

    def note(self, obs):
        if self.k in GATE_FRAMES:
            self.rows[self.k] = {c: fingerprint(through_encoder(np.asarray(obs[c]))) for c in self.cams}
        self.k += 1

    def drop(self):
        self.k, self.rows = 0, {}

    def keep(self, root, episode):
        p = pathlib.Path(root) / GATE_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(dict(episode=int(episode), frames={str(k): v for k, v in self.rows.items()})) + "\n")
        self.drop()


def gate(repo_id, root=None, limit=None):
    """Every fingerprinted frame of a dataset, read back as training reads it, against its fingerprint.
    Returns (checked, failed)."""
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    root = pathlib.Path(root) if root else HF_LEROBOT_HOME / repo_id
    ds = LeRobotDataset(repo_id, root=root)
    rows = [json.loads(x) for x in open(root / GATE_FILE) if x.strip()]
    checked, failed = 0, []
    for r in rows[:limit]:
        e = ds.meta.episodes[r["episode"]]
        i0, n = int(e["dataset_from_index"]), int(e["dataset_to_index"]) - int(e["dataset_from_index"])
        for k, cams in r["frames"].items():
            if int(k) >= n:
                continue
            item = ds[i0 + int(k)]
            for c, want in cams.items():
                img = (item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                checked += 1
                if fingerprint(img) != want:
                    failed.append((r["episode"], int(k), c))
    print(f"{repo_id}: {checked} stored frames checked against the recorder's fingerprints, {len(failed)} differ"
          + (f": {failed[:10]}" if failed else " -> PASS"))
    return checked, failed


def test_pictures():
    """Raw simulator pictures for selfcheck: the middle third of the post-mortem's first-frame triptychs
    (recorded | simulator | difference), 3 scenes x base and claw camera."""
    import cv2
    here = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "notes" / "2026-11-19_v5_postmortem"
    out = []
    for f in sorted(here.glob("first_frame_*_*.png")):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        out.append(np.ascontiguousarray(im[:, 640:1280]))
    return out


def selfcheck():
    """For each candidate setting, through LeRobot's own encode and decode: is a stored frame of a clip the same as
    its picture sent through alone (the rule), what does storing cost in grey levels, and how big is a frame."""
    from lerobot.configs.video import RGBEncoderConfig
    P = test_pictures()
    if len(P) < 3:
        raise SystemExit("no raw test pictures (outputs/notes/2026-11-19_v5_postmortem/first_frame_*.png)")
    P = P[0::2] + P[1::2]                                   # 3 base pictures, then 3 claw pictures: 6 different frames
    cand = [
        ("AV1, LeRobot's default (v5)", dict(vcodec="libsvtav1"), None),
        ("AV1 g 1 + pred-struct=1 as an extra option", dict(vcodec="libsvtav1", g=1, extra_options={
            "svtav1-params": "fast-decode=0:pred-struct=1"}), None),
        ("  the same with fast_decode=None", dict(vcodec="libsvtav1", g=1, fast_decode=None, extra_options={
            "svtav1-params": "fast-decode=0:pred-struct=1"}), None),
        ("  the same with fast_decode=None, 4 threads", dict(vcodec="libsvtav1", g=1, fast_decode=None,
                                                             extra_options={"svtav1-params":
                                                                            "fast-decode=0:pred-struct=1"}), 4),
        ("H.264 g 1, crf 23 (its usual mode)", dict(vcodec="h264", g=1, crf=23, preset="medium"), None),
        ("H.264 g 1, qp 18", dict(STORE, extra_options={"qp": 18}), None),
        ("  the same, 4 threads", dict(STORE, extra_options={"qp": 18}), 4),
        ("H.264 g 1, qp 23 = STORE", dict(STORE, extra_options={"qp": 23}), None),
        ("H.264 g 1, qp 26", dict(STORE, extra_options={"qp": 26}), None),
    ]
    print(f"{len(P)} raw 640 x 480 simulator pictures as one clip; decoder: {backend()}")
    print(f"{'setting':46s} | options passed to the encoder{'':36s} | stored = alone | loss | kB/frame")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for name, kw, threads in cand:
            cfg = RGBEncoderConfig(**kw)
            opts = cfg.get_codec_options(encoder_threads=threads)
            v = pathlib.Path(td) / f"clip_{next(_N)}.mp4"
            _encode(P, v, cfg, threads)
            size = v.stat().st_size / len(P) / 1e3
            clip = _decode(v, len(P))
            worst, loss = 0, []
            for k, img in enumerate(P):
                alone = through_encoder(img, cfg, threads=threads)
                worst = max(worst, int(np.abs(clip[k].astype(np.int16) - alone.astype(np.int16)).max()))
                loss.append(float(np.abs(clip[k].astype(np.int16) - img.astype(np.int16)).mean()))
            print(f"{name:46s} | {str(opts):64s} | {'yes' if worst == 0 else f'NO (max {worst})':14s} | "
                  f"{np.mean(loss):4.2f} | {size:5.1f}")
        # the decoders against each other on STORE (training may use either)
        v = pathlib.Path(td) / f"clip_{next(_N)}.mp4"
        _encode(P, v)
        try:
            a, b = _decode(v, len(P), "pyav"), _decode(v, len(P), "torchcodec")
            d = max(int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) for x, y in zip(a, b))
            print(f"STORE decoded by pyav against torchcodec: largest difference {d} grey levels")
        except Exception as e:      # noqa: BLE001
            print(f"STORE decoded by pyav against torchcodec: could not compare ({type(e).__name__}: {e})")
        # the recorder's fingerprint path on the same pictures
        ok = all(fingerprint(through_encoder(img)) == fingerprint(c) for img, c in zip(P, _decode(v, len(P))))
        print("through_encoder(picture) = the clip's stored frame, by fingerprint:", "yes" if ok else "NO")


def gatecheck():
    """The gate end to end on two throwaway datasets made of the test pictures (TMPDIR, nothing kept, no simulator):
    one stored with STORE must PASS, one stored with LeRobot's default must FAIL."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset
    P = test_pictures()
    cams = ("base", "claw")
    feats = {f"observation.images.{c}": {"dtype": "video", "shape": (480, 640, 3),
                                         "names": ["height", "width", "channels"]} for c in cams}
    feats["observation.state"] = {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}
    feats["action"] = {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}
    verdict = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for name, cfg in (("stored with STORE", encoder()), ("stored as v5", RGBEncoderConfig(vcodec="libsvtav1"))):
            root = pathlib.Path(td) / name.replace(" ", "_")
            ds = LeRobotDataset.create("local/gatecheck", FPS, root=root, robot_type="none", features=feats,
                                       use_videos=True, rgb_encoder=cfg, image_writer_threads=2)
            g = Gate(cams)
            for ep in range(2):
                for k in range(GATE_FRAMES[-1] + 6):
                    obs = {c: P[(k + ep + i) % len(P)] for i, c in enumerate(cams)}
                    g.note(obs)
                    ds.add_frame({**{f"observation.images.{c}": obs[c] for c in cams},
                                  "observation.state": np.zeros(2, np.float32), "action": np.zeros(2, np.float32),
                                  "task": "gatecheck"})
                ds.save_episode(parallel_encoding=False)
                g.keep(ds.root, ds.meta.total_episodes - 1)
            ds.finalize()
            verdict[name] = gate("local/gatecheck", root)
    ok = not verdict["stored with STORE"][1] and bool(verdict["stored as v5"][1])
    print("gatecheck:", "PASS (the gate passes STORE and catches the v5 storage)" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("selfcheck")
    sub.add_parser("gatecheck")
    g = sub.add_parser("gate")
    g.add_argument("repo_id")
    g.add_argument("--root")
    g.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.mode == "selfcheck":
        selfcheck()
    elif a.mode == "gatecheck":
        raise SystemExit(0 if gatecheck() else 1)
    else:
        raise SystemExit(1 if gate(a.repo_id, a.root, a.limit)[1] else 0)
