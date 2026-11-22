#!/usr/bin/env python3
"""Post-mortem 2026-11-20: which video-encoder setting writes the NEXT frame into a stored frame, and which do not?
Pictures in, pictures out: no simulator, no policy, no training.

    conda activate lerobot && python encoder_leak_test.py [--episodes 4]

Why: the v5 dataset stores its camera pictures as AV1 video (crf 30, a key frame every 2 frames, LeRobot's default).
replay_raw_vs_recorded.py showed that a stored frame is not a function of its own picture alone: frame 0 of a clip
comes out different when frame 1 is there. At the start pose the claw camera sees the same wall in every scene, so
that trace of the next frame was the only thing in the picture that differed between episodes, and the policy
learned to read it (claw_frame_leak.py). A test gives the policy the raw picture, which has no such trace.

This measures the trace for the dataset's setting and for candidate settings of the NEXT dataset. For each setting:

    next-frame trace   frame k of a clip against frame k of the same clip with every LATER frame replaced by a copy of
                       picture k (same length, only the future differs). 0.000 = a stored frame does not depend on
                       what comes after it. Mean |difference| in grey levels of 255; and the share of clips with any.
    earlier-frame trace  the same with every EARLIER frame replaced: 0.000 = it does not depend on what came before
    alone              frame k of a clip against picture k sent through the encoder ALONE, as a one-frame clip. That
                       is what a test can do with a live picture. 0.000 for every frame = passing the live picture
                       through the encoder reproduces a stored frame exactly (the stored frame is a function of its
                       own picture and nothing else).
    loss               stored frame against its own source picture (what compression costs), same unit
    size               bytes per frame on 30 consecutive frames of a moving arm (the v5 dataset: 8.6 kB per frame,
                       6.6 GB for 2 x 383,877 frames)

⚠️ The source pictures are decoded frames of the dataset (pictures like any other): what is measured is a property of
the encoder, not of those pictures.
The switches are from the SVT-AV1 parameter documentation (Docs/Parameters.md, read 2026-11-20): enable-kf-tf
"Enable MCTF for key frames" (default on), enable-tf "Enable ALT-REF (temporally filtered) frames" (default on),
lookahead "Number of frames in the future to look ahead, beyond minigop, temporal filtering, and rate control",
pred-struct "0: all intra, 1: low delay, 2: random access" (default 2).
"""
import argparse
import glob
import json
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "notes" / "2026-11-19_v5_postmortem"   # generated files (git-ignored), not the repo tree

SNAP = sorted(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/lerobot/hub/datasets--krish5831--"
                            "blush_sim_teacher_v5/snapshots/*")))[-1]
LOGS = sorted(glob.glob(str(pathlib.Path.home() / ".cache/blush_sim/truss/blush_sim_teacher_v5_p*.jsonl")))
FPS = 30            # the dataset's frame rate (meta/info.json)
N = 9               # frames per test clip: frame 0, a later key frame (4) and a later in-between frame (5) fit
LOOK = (0, 4, 5)    # frames whose dependence on the future is measured
STARTS = (0, 100)   # where in the episode the clip starts: the start pose (the arm about to move), and mid-motion
SIZE_FROM, SIZE_N = 60, 30   # the moving clip used for the size: frames 60-89 (the arm swings toward the row)
AV1 = "fast-decode=0"        # what LeRobot's encoder passes for its default fast_decode=0 (configs/video.py)

# name -> (encoder, pixel format, options). The first row is the dataset's setting: LeRobot's RGBEncoderConfig
# defaults (libsvtav1, yuv420p, g 2, crf 30, preset 12).
SETTINGS = [
    ("AV1 as the v5 dataset (crf 30, key frame every 2)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1})),
    ("  + key-frame temporal filter off (enable-kf-tf=0)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":enable-kf-tf=0"})),
    ("  + temporal filter off (enable-tf=0)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":enable-tf=0"})),
    ("  + both filters off", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":enable-tf=0:enable-kf-tf=0"})),
    ("  + look-ahead 0 (lookahead=0)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":lookahead=0"})),
    ("  + both filters off, look-ahead 0", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":enable-tf=0:enable-kf-tf=0:lookahead=0"})),
    ("  + low delay (pred-struct=1)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":pred-struct=1"})),
    ("  + low delay, both filters off", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":pred-struct=1:enable-tf=0:enable-kf-tf=0"})),
    ("  + all intra (pred-struct=0)", "libsvtav1", "yuv420p", dict(g="2", crf="30", preset="12", **{"svtav1-params": AV1 + ":pred-struct=0"})),
    ("AV1, every frame a key frame (g 1)", "libsvtav1", "yuv420p", dict(g="1", crf="30", preset="12", **{"svtav1-params": AV1})),
    ("AV1 low delay, every frame a key frame", "libsvtav1", "yuv420p", dict(g="1", crf="30", preset="12", **{"svtav1-params": AV1 + ":pred-struct=1"})),
    ("AV1 low delay, every frame a key frame, crf 20", "libsvtav1", "yuv420p", dict(g="1", crf="20", preset="12", **{"svtav1-params": AV1 + ":pred-struct=1"})),
    ("AV1, every frame a key frame, crf 20", "libsvtav1", "yuv420p", dict(g="1", crf="20", preset="12", **{"svtav1-params": AV1})),
    ("AV1, key frame every 30 (ordinary video)", "libsvtav1", "yuv420p", dict(g="30", crf="30", preset="12", **{"svtav1-params": AV1})),
    ("H.264, crf 23, key frame every 2", "libx264", "yuv420p", dict(g="2", crf="23", preset="medium")),
    ("H.264, crf 23, key frame every 2, no look-ahead (zerolatency)", "libx264", "yuv420p", dict(g="2", crf="23", preset="medium", tune="zerolatency")),
    ("H.264, crf 23, every frame a key frame (g 1)", "libx264", "yuv420p", dict(g="1", crf="23", preset="medium")),
    ("H.264, crf 18, every frame a key frame", "libx264", "yuv420p", dict(g="1", crf="18", preset="medium")),
    ("H.264, crf 12, every frame a key frame", "libx264", "yuv420p", dict(g="1", crf="12", preset="medium")),
    ("H.264, constant quantiser 18, every frame a key frame", "libx264", "yuv420p", dict(g="1", qp="18", preset="medium")),
    ("H.264 lossless RGB, every frame a key frame", "libx264rgb", "rgb24", dict(g="1", crf="0", preset="veryfast")),
]


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    import av
    from PIL import Image
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames

    em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/meta/episodes/*/*.parquet"))])
    em = em.set_index("episode_index")
    kept = [r for f in LOGS for r in (json.loads(ln) for ln in open(f) if ln.strip()) if r.get("kept") is True]
    by_seed = {r["seed"]: i for i, r in enumerate(kept)}
    eps = [by_seed[json.loads(ln)["seed"]] for ln in open(HERE / "train_clean_50.jsonl") if ln.strip()][:a.episodes]

    def source(ep, cam, ks):
        key = f"videos/observation.images.{cam}"
        path = (f"{SNAP}/videos/observation.images.{cam}/chunk-{int(em.loc[ep, key + '/chunk_index']):03d}/"
                f"file-{int(em.loc[ep, key + '/file_index']):03d}.mp4")
        t0 = float(em.loc[ep, key + "/from_timestamp"])
        fr = decode_video_frames(path, [t0 + k / FPS for k in ks], 1e-3, backend="pyav", return_uint8=True)
        return [f.permute(1, 2, 0).numpy() for f in fr]

    tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
    td = pathlib.Path(tmp.name)

    def encode(frames, enc, pix, opts, name="clip.mp4"):
        """As lerobot's encode_video_frames (datasets/video_utils.py) does it, with any options. -> (frames, bytes)"""
        path = td / name
        with av.open(str(path), "w") as out:
            st = out.add_stream(enc, FPS, options=dict(opts))
            st.pix_fmt, st.width, st.height = pix, frames[0].shape[1], frames[0].shape[0]
            for img in frames:
                pkt = st.encode(av.VideoFrame.from_image(Image.fromarray(img)))
                if pkt:
                    out.mux(pkt)
            pkt = st.encode()
            if pkt:
                out.mux(pkt)
        size = path.stat().st_size
        with av.open(str(path)) as c:
            dec = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        return dec, size

    def diff(x, y):
        return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

    # 0. this file's encoder call and decoder against the library's own, on the dataset's setting
    src = source(eps[0], "claw", range(N))
    d = td / "imgs"
    d.mkdir()
    for k, img in enumerate(src[:2]):
        Image.fromarray(img).save(d / f"frame-{k:06d}.png")
    encode_video_frames(d, td / "lib.mp4", FPS, video_encoder=RGBEncoderConfig(vcodec="libsvtav1"), overwrite=True)
    lib = decode_video_frames(str(td / "lib.mp4"), [0.0, 1 / FPS], 1e-3, backend="pyav", return_uint8=True)
    mine, _ = encode(src[:2], *SETTINGS[0][1:])
    print(f"check: this file's encoder call + decoder against the library's on a 2-frame clip: largest difference "
          f"{max(int(np.abs(lib[k].permute(1, 2, 0).numpy().astype(np.int16) - mine[k].astype(np.int16)).max()) for k in (0, 1))} "
          "grey levels (0 = the rows below describe what the dataset's recording did)")
    print(f"{len(eps)} picking episodes x 2 cameras x clips starting at frames {STARTS}; {N} frames per clip\n")

    S = {}                                               # (episode, camera, start) -> source frames
    for ep in eps:
        for cam in ("base", "claw"):
            for s in STARTS:
                S[(ep, cam, s)] = source(ep, cam, range(s, s + N))
    M = {(ep, cam): source(ep, cam, range(SIZE_FROM, SIZE_FROM + SIZE_N)) for ep in eps for cam in ("base", "claw")}

    print(f"{'setting':62s} | next-frame trace: frame 0, claw at the start | frame 0, all clips | frame 4 | frame 5 "
          "| earlier-frame trace: frame 4 | frame 5 | alone: frame 0 | frame 4 | frame 5 | loss | kB per frame")
    res = []
    for name, enc, pix, opts in SETTINGS:
        try:
            t0c, t0, t4, t5, loss = [], [], [], [], []
            past, alone = {4: [], 5: []}, {0: [], 4: [], 5: []}
            for (_ep, cam, s), P in S.items():
                full, _ = encode(P, enc, pix, opts)
                for k, bucket in zip(LOOK, (t0, t4, t5)):
                    cut, _ = encode(P[:k + 1] + [P[k]] * (N - 1 - k), enc, pix, opts)
                    bucket.append(diff(full[k], cut[k]))
                    if k == 0 and cam == "claw" and s == 0:
                        t0c.append(bucket[-1])
                    alone[k].append(diff(full[k], encode([P[k]], enc, pix, opts)[0][0]))
                    if k:
                        before, _ = encode([P[k]] * k + P[k:], enc, pix, opts)
                        past[k].append(diff(full[k], before[k]))
                loss.append(np.mean([diff(full[k], P[k]) for k in range(N)]))
            size = np.mean([encode(P, enc, pix, opts)[1] / SIZE_N for P in M.values()]) / 1000
        except Exception as e:                           # a switch this build of the encoder does not know
            print(f"{name:62s} | not accepted by this encoder build ({type(e).__name__}: {str(e)[:60]})")
            continue
        res.append(dict(setting=name, encoder=enc, options=opts, trace0_claw_start=med(t0c), trace0=med(t0), trace0_any=float(np.mean(np.array(t0) > 0)),
                        trace_key=med(t4), trace_between=med(t5), trace_between_any=float(np.mean(np.array(t5) > 0)),
                        past_key=med(past[4]), past_between=med(past[5]), alone0=med(alone[0]), alone_key=med(alone[4]), alone_between=med(alone[5]),
                        alone_any=float(np.mean(np.array(alone[0] + alone[4] + alone[5]) > 0)), loss=med(loss), kb_per_frame=float(size)))
        r = res[-1]
        print(f"{name:62s} | {r['trace0_claw_start']:26.3f}                   | {r['trace0']:6.3f} ({r['trace0_any']:4.0%} any) | {r['trace_key']:7.3f} "
              f"| {r['trace_between']:7.3f} | {r['past_key']:27.3f} | {r['past_between']:7.3f} | {r['alone0']:14.3f} | {r['alone_key']:7.3f} | {r['alone_between']:7.3f} "
              f"| {r['loss']:4.2f} | {r['kb_per_frame']:6.1f}", flush=True)
    (OUT / "encoder_leak_test.json").write_text(json.dumps(res, indent=1))
    tmp.cleanup()


if __name__ == "__main__":
    main()
