#!/usr/bin/env python3
"""Does the training notebook's own picture gate (colab_train_act_v6.ipynb, cell 4) agree with picture_store.gate?

    python scripts/check_colab_gate.py            # from sim/, env lerobot-mj4, ~1 min, nothing kept

The notebook cannot import picture_store (Colab has no checkout of this repo), so cell 4 carries its own copy of the
gate. This runs that cell's code, as it stands in the notebook file, on two throwaway datasets made of the test
pictures (TMPDIR, no simulator): one stored by the picture rule, which it must PASS, and one stored as v5 was, which
it must refuse. Only the download and Colab's disk are stubbed out.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import picture_store as ps  # noqa: E402

NOTEBOOK = HERE.parent / "colab_train_act_v6.ipynb"
CELL = 4


def tiny(root, cfg, cams=("high", "low", "claw")):
    """A two-episode dataset of the test pictures with the recorder's gate file, as picture_store.gatecheck's."""
    from lerobot.datasets import LeRobotDataset
    P = ps.test_pictures()
    feats = {f"observation.images.{c}": {"dtype": "video", "shape": (480, 640, 3),
                                         "names": ["height", "width", "channels"]} for c in cams}
    feats["observation.state"] = {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}
    feats["action"] = {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}
    ds = LeRobotDataset.create("local/gate", ps.FPS, root=root, robot_type="none", features=feats, use_videos=True,
                               rgb_encoder=cfg, image_writer_threads=2)
    g = ps.Gate(cams)
    for ep in range(2):
        for k in range(ps.GATE_FRAMES[-1] + 6):
            obs = {c: P[(k + ep + i) % len(P)] for i, c in enumerate(cams)}
            g.note(obs)
            ds.add_frame({**{f"observation.images.{c}": obs[c] for c in cams},
                          "observation.state": np.zeros(2, np.float32), "action": np.zeros(2, np.float32),
                          "task": "gate"})
        ds.save_episode(parallel_encoding=False)
        g.keep(ds.root, ds.meta.total_episodes - 1)
    ds.finalize()


def run_cell(home, repo):
    """Cell 4's code on <home>/<repo>. Returns None when it passes, else the reason it stopped."""
    src = "".join(json.load(open(NOTEBOOK))["cells"][CELL]["source"])
    assert "picture_gate.jsonl" in src and "snapshot_download" in src, "cell 4 is not the gate cell any more"
    src = src.replace("from huggingface_hub import snapshot_download", "")      # the dataset is here already
    fake_shutil = types.SimpleNamespace(disk_usage=lambda _: types.SimpleNamespace(free=0.0))
    os.environ["HF_LEROBOT_HOME"] = str(home)
    ns = dict(os=os, shutil=fake_shutil, DATASET=repo, snapshot_download=lambda *a, **k: None, GATE_EVERY=1)
    src = src.replace("GATE_EVERY = 10", "GATE_EVERY = 1")                        # 2 episodes: check both
    try:
        exec(compile(src, "cell4", "exec"), ns)
    except AssertionError as e:
        return str(e)
    return None


def main():
    from lerobot.configs.video import RGBEncoderConfig
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        good, old = pathlib.Path(td) / "good" / "local" / "gate", pathlib.Path(td) / "old" / "local" / "gate"
        tiny(good, ps.encoder())
        tiny(old, RGBEncoderConfig(vcodec="libsvtav1"))
        mine = ps.gate("local/gate", good)
        a = run_cell(good.parents[1], "local/gate")
        b = run_cell(old.parents[1], "local/gate")
        # a damaged gate file: one fingerprint changed must fail the cell
        rows = [json.loads(x) for x in open(good / ps.GATE_FILE)]
        k0 = next(iter(rows[0]["frames"]))
        c0 = next(iter(rows[0]["frames"][k0]))
        rows[0]["frames"][k0][c0] = "0" * 40
        open(good / ps.GATE_FILE, "w").write("".join(json.dumps(r) + "\n" for r in rows))
        c = run_cell(good.parents[1], "local/gate")
        shutil.rmtree(td, ignore_errors=True)
    ok = a is None and not mine[1] and mine[0] > 0 and b is not None and c is not None
    print(f"\npicture_store.gate on the good dataset: {mine[0]} frames, {len(mine[1])} differ")
    said = lambda r: "PASSED (wrong)" if r is None else "refused: " + r[:100]   # noqa: E731
    print(f"notebook cell {CELL} on the good dataset: {'PASS' if a is None else 'stopped: ' + a}")
    print(f"notebook cell {CELL} on a dataset stored as v5: {said(b)}")
    print(f"notebook cell {CELL} with one fingerprint changed: {said(c)}")
    print("check_colab_gate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
