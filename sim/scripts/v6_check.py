#!/usr/bin/env python3
"""Is a finished part of dataset v6 right? "The worker ended" does not say so: remake_parts.sh and record_parts.sh
exit 0 whatever the part did, and a part that stopped early is still finalized and reads as a dataset.

    python scripts/v6_check.py parts krish5831/<name>_old krish5831/<name>_new        # every finalized part
    python scripts/v6_check.py merged krish5831/<name>_trial krish5831/<name>_old krish5831/<name>_new
    python scripts/v6_check.py sheet krish5831/<name>_trial krish5831/<name>_old krish5831/<name>_new \
        --out ../outputs/v6/trial                                                     # pictures as training reads them
    python scripts/v6_check.py videos krish5831/<name> krish5831/<name>_old krish5831/<name>_new \
        --out ../outputs/videos/review_<name>                                         # five videos for K7's look

(from sim/, env lerobot-mj4; reads only, apart from the sheets and videos it is asked for. The prefixes are
merge_parts.py's. First run 2026-11-27 on dataset v6's trial: both parts and the merge pass, notes section 28.
The whole dataset the same evening: all 57 parts and the merge pass.)
⚠️ `videos` calls ffmpeg: activate the env first. The env's python called by its path alone does not find it
(FileNotFoundError: 'ffmpeg', 2026-11-27).

parts, per part (its kind comes from its attempt log in ~/.cache/blush_sim/truss: re-made or recorded):
  storage   the three cameras of rig trio, stored with picture_store's setting (encoder_of = encoder()), the gate
            file there, and the gate itself: every fingerprinted frame read back as training reads it (2 frames x 3
            cameras per episode)
  re-made   25 episodes (24 in the part of v5 episode 353); joint readings and commands equal v5's bit for bit;
            no cut episode with an all-zero box in the cue numbers
  recorded  kept episodes per zone against the plan's quota (a part whose scenes ran out is "short": input for
            v6_seed_plan.py --topup, not a redo); the cue numbers in the dataset = the log's; refusals by kind,
            with the two only a recording pass has ("recording differs from the dry run", "recording picked
            another truss") named
A part that is not finalized (still being written, or killed before its finalize) is skipped with a line.
  size      MB of video per episode, and what that makes for the whole dataset (799 + 600 episodes)
merged: the storage checks on the merge, its episode and frame counts against the parts', and every part's numbers
  (joint readings, commands, cue numbers) found again at its place in the merge, bit for bit.
sheet: for a re-made cut, a recorded cut, a recorded noisy cut and a nothing-ripe scene of each kind, the three
  cameras at the start frame and mid-episode, read through LeRobotDataset (the decoded video, not a render).
videos: the five review videos K7 looks at before a training (same arguments as sheet; replaces review_videos.py,
  which is dataset v5's: two cameras, every part 25 episodes).
"""
import argparse
import collections
import glob
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "whatif"))

CAMS = ("high", "low", "claw")      # cameras.RIG_CAMERAS["trio"]
LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "truss"     # generate_truss.LOG_DIR
FPS = 30                            # the dataset's frame rate (expert.FPS, config/hardware.env)
WHOLE = 799 + 600                   # episodes of the whole dataset v6: v5's 800 less episode 353, + 600 new
#                                     (K7 2026-11-26)
ONLY_RECORDING = ("recording differs from the dry run", "recording picked another truss")   # generate_truss.py's
#                                     two refusals that a dry rehearsal cannot show


def home():
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return HF_LEROBOT_HOME


def parts_of(prefixes):
    """The parts merge_parts.py merges, in its order: prefix by prefix, by name."""
    out = []
    for pre in prefixes:
        user, name = pre.split("/")
        out += [f"{user}/{d.name}" for d in sorted((home() / user).glob(name + "*")) if re.search(r"_p\d{2}$", d.name)]
    return out


def info(root):
    return json.load(open(root / "meta" / "info.json"))


def numbers(root):
    """The dataset's own numbers, one row per frame, in episode and frame order."""
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet"))))
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def col(g, name):
    return np.stack(g[name].to_numpy()).astype(np.float32)


def log_rows(pid):
    """The part's latest attempt log (a part redone after a kill has several), without the summary row."""
    files = sorted(f for f in LOG_DIR.glob(pid.split("/")[-1] + "_2*.jsonl"))
    if not files:
        return None, []
    return files[-1], [r for r in (json.loads(x) for x in open(files[-1]) if x.strip()) if not r.get("summary")]


def storage(repo_id, root, say):
    """The picture rule on one dataset. Returns (episodes, frames, video bytes)."""
    import picture_store
    from lerobot.datasets import LeRobotDatasetMetadata
    i = info(root)
    feats = LeRobotDatasetMetadata(repo_id, root=root).features
    cams = tuple(k.split(".")[-1] for k, v in feats.items() if v.get("dtype") == "video")
    say(set(cams) == set(CAMS), f"cameras {cams}")
    try:
        same = picture_store.encoder_of(feats) == picture_store.encoder()
        v = (feats[f"observation.images.{CAMS[0]}"].get("info") or {})
        say(same, f"stored as {v.get('video.codec')} g {v.get('video.g')} crf {v.get('video.crf')} "
                  f"{v.get('video.extra_options')} (picture_store.STORE: {'the same' if same else 'DIFFERENT'})")
        say(picture_store.stored_with_gate(root, feats), "gate file there")
    except RuntimeError as e:
        say(False, str(e))
    rows = [json.loads(x) for x in open(root / picture_store.GATE_FILE) if x.strip()]
    checked, failed = picture_store.gate(repo_id, root)
    want = i["total_episodes"] * len(picture_store.GATE_FRAMES) * len(CAMS)
    say(len(rows) == i["total_episodes"] and checked == want and not failed,
        f"gate: {checked} frames read back (want {want} = {i['total_episodes']} episodes x "
        f"{len(picture_store.GATE_FRAMES)} frames x {len(CAMS)} cameras), {len(failed)} differ")
    mp4 = sum(f.stat().st_size for f in root.rglob("*.mp4"))
    return i["total_episodes"], i["total_frames"], mp4


def check_remade(root, rows, J, say):
    kept = [r for r in rows if r.get("kept")]
    left = [r for r in rows if not r.get("kept")]
    df = numbers(root)
    say(len(kept) + len(left) == 25 and df["episode_index"].nunique() == len(kept),      # remake_pictures.PART_EPISODES
        f"re-made {len(kept)} of 25" + (f", left out by name: {[r['from_merged'] for r in left]}" if left else ""))
    bad, no_cue, frames = [], [], 0
    for r in kept:
        g = df[df["episode_index"] == r["episode_index"]]
        st, act, _ = J[r["from_merged"]]
        frames += len(g)
        if col(g, "observation.state").tobytes() != st.tobytes() or col(g, "action").tobytes() != act.tobytes():
            bad.append(r["from_merged"])
        env = col(g, "observation.environment_state")
        if not (env == env[0]).all() or not np.allclose(env[0], r["env"], atol=1e-4):
            bad.append(("cue", r["from_merged"]))
        if not r["hold"] and not env[0][:4].any():
            no_cue.append(r["from_merged"])
    say(not bad, f"joint readings and commands of {len(kept)} episodes ({frames} frames) equal v5's bit for bit"
        + (f": NOT for v5 episodes {bad}" if bad else ""))
    say(not no_cue, f"cut episodes with an all-zero box in the cue numbers: {len(no_cue)} {no_cue or ''}")
    return dict(kept=len(kept), hold=sum(bool(r["hold"]) for r in kept), seconds=sum(r["wall_s"] for r in kept))


def check_recorded(root, rows, plan, say):
    kept = [r for r in rows if r.get("kept") and "episode_index" in r]
    refused = [r for r in rows if not r.get("kept")]
    quota = json.loads(open(plan).readline())
    df = numbers(root)
    cells = collections.Counter("hold" if r.get("hold") else r["cell"] for r in kept)
    say(df["episode_index"].nunique() == len(kept),
        f"kept {len(kept)} of {quota['episodes']} ({cells['hold']} nothing-ripe), {len(rows)} scenes built; the "
        f"dataset has {df['episode_index'].nunique()} episodes")
    short = {c: n - cells[c] for c, n in quota["quota"].items() if cells[c] < n}
    over = {c: n - quota["quota"].get(c, 0) for c, n in cells.items() if n > quota["quota"].get(c, 0)}
    say(not over, "no zone above the plan's quota" + (f": NOT, {over} too many" if over else ""))
    if short:       # the plan's scenes ran out: not a redo, v6_seed_plan.py --topup plans these as an extra part
        print(f"   short  {sum(short.values())} episodes missing: {short} (input for the top-up, not a redo)")
    else:
        print("   ok  every zone at the plan's quota")
    bad = []
    for r in kept:
        env = col(df[df["episode_index"] == r["episode_index"]], "observation.environment_state")
        if not (env == env[0]).all() or not np.allclose(env[0], r["env"], atol=1e-4):
            bad.append(r["seed"])
        if not r.get("hold") and not env[0][:4].any():
            bad.append(("no box", r["seed"]))
    say(not bad, "cue numbers in the dataset = the log's, no cut episode without a box"
        + (f": NOT {bad}" if bad else ""))
    kinds = collections.Counter(re.split(r"[:(]", r["reason"])[0].strip() for r in refused)
    only = sum(n for k, n in kinds.items() if k.startswith(ONLY_RECORDING))
    print(f"      refused {len(refused)}: {dict(kinds)}; of these {only} are a recording pass's own")
    cut = [r for r in kept if not r.get("hold")]
    print(f"      noisy {sum('noise' in r for r in cut)} of {len(cut)} cut episodes; start pose: first draw clear in "
          f"{sum(r.get('start_draws') == 1 for r in rows)} of {len(rows)} scenes, "
          f"{sum(not any(r.get('start_drawn_deg', [1])) for r in rows)} at the exact pose")
    return dict(kept=len(kept), hold=cells["hold"], cells=cells, only=only, short=sum(short.values()),
                seconds=sum(r.get("wall_s", 0) for r in rows))


def one_part(pid, plan_dir, J):
    root = home() / pid
    bad = []

    def say(ok, text):
        print(f"   {'ok ' if ok else '⚠️ '} {text}")
        if not ok:
            bad.append(text)

    f, rows = log_rows(pid)
    kind = "re-made" if rows and "from_merged" in rows[0] else "recorded"
    print(f"{pid}  ({kind}; log {f.name if f else 'MISSING'})")
    eps, frames, mp4 = storage(pid, root, say)
    if not rows:
        say(False, "no attempt log: the part's episodes cannot be checked against it")
        facts = dict(kept=eps, hold=0, seconds=0)
    elif kind == "re-made":
        facts = check_remade(root, rows, J, say)
    else:
        facts = check_recorded(root, rows, pathlib.Path(plan_dir) / f"p{pid[-2:]}.jsonl", say)
    print(f"      {frames} frames, {mp4 / 1e6:.0f} MB of video = {mp4 / 1e6 / max(eps, 1):.1f} MB per episode, "
          f"{mp4 / 1e3 / max(frames, 1) / len(CAMS):.1f} kB per picture; "
          f"{facts['seconds'] / 60:.0f} min of work by its log")
    return dict(pid=pid, kind=kind, bad=bad, eps=eps, frames=frames, mp4=mp4, **facts)


def parts(a):
    import remake_pictures as rp
    P = parts_of(a.prefixes)
    if not P:
        sys.exit("no parts found")
    from merge_parts import finalized
    J = rp.joint_data()
    skipped = [p for p in P if not finalized(home() / p)]
    for p in skipped:       # a part still being written (or killed before its finalize) has no parquet footer
        print(f"{p}: not finalized (still being written, or killed before its finalize): skipped")
    R = [one_part(p, a.plan_dir, J) for p in P if p not in skipped]
    print()
    for kind, n_all in (("re-made", 799), ("recorded", 600)):       # WHOLE's two halves
        K = [r for r in R if r["kind"] == kind]
        if K:
            eps, mp4 = sum(r["eps"] for r in K), sum(r["mp4"] for r in K)
            print(f"{kind}: {len(K)} parts, {eps} episodes ({sum(r['hold'] for r in K)} nothing-ripe), "
                  f"{mp4 / 1e9:.2f} GB of video -> {mp4 / eps * n_all / 1e9:.1f} GB for all {n_all}")
    if {r["kind"] for r in R} == {"re-made", "recorded"}:
        per = {k: sum(r["mp4"] for r in R if r["kind"] == k) / sum(r["eps"] for r in R if r["kind"] == k)
               for k in ("re-made", "recorded")}
        print(f"whole dataset at these sizes: {(per['re-made'] * 799 + per['recorded'] * 600) / 1e9:.1f} GB of video "
              f"for {WHOLE} episodes")
    cells = sum((r["cells"] for r in R if "cells" in r), collections.Counter())
    if cells:
        print("recorded per zone:", dict(sorted(cells.items())))
    short = {r["pid"][-3:]: r["short"] for r in R if r.get("short")}
    if short:
        print(f"recorded parts that ended short (top-up, not a redo): {short} = {sum(short.values())} episodes")
    wrong = [r["pid"] for r in R if r["bad"]]
    print(("⚠️ parts with a failed check (redo): " + ", ".join(wrong) if wrong else f"all {len(R)} checked parts pass")
          + (f"; {len(skipped)} not finalized and skipped" if skipped else ""))
    return 1 if wrong else 0


def merged(a):
    root, P = home() / a.out, parts_of(a.prefixes)
    bad = []

    def say(ok, text):
        print(f"   {'ok ' if ok else '⚠️ '} {text}")
        if not ok:
            bad.append(text)

    print(f"{a.out}  (merge of {len(P)} parts)")
    eps, frames, mp4 = storage(a.out, root, say)
    n = [(info(home() / p)["total_episodes"], info(home() / p)["total_frames"]) for p in P]
    say((eps, frames) == (sum(x[0] for x in n), sum(x[1] for x in n)),
        f"{eps} episodes, {frames} frames (the parts: {sum(x[0] for x in n)}, {sum(x[1] for x in n)})")
    M, off, wrong = numbers(root), 0, []
    for p, (e, _) in zip(P, n):
        D = numbers(home() / p)
        for k in range(e):
            g, h = M[M["episode_index"] == off + k], D[D["episode_index"] == k]
            if len(g) != len(h) or any(col(g, c).tobytes() != col(h, c).tobytes() for c in
                                       ("observation.state", "action", "observation.environment_state")):
                wrong.append((p.split("_")[-2] + "_" + p[-3:], k))
        off += e
    say(not wrong, "every part's joint readings, commands and cue numbers are at their place in the merge, bit for bit"
        + (f": NOT {wrong[:10]}" if wrong else ""))
    print(f"      {mp4 / 1e9:.2f} GB of video")
    print("⚠️ the merge fails a check" if bad else "the merge passes")
    return 1 if bad else 0


def episodes_of(prefixes):
    """Every episode of a merge of these parts: its number in the merge and what it is, from the parts' logs.
    ⚠️ Not 25 x part + local number (review_videos.py, dataset v5): a part can be short (the re-made part of v5
    episode 353 has 24, a recorded part can end short and is topped up by a later one)."""
    off, E = 0, []
    for p in parts_of(prefixes):
        _, rows = log_rows(p)
        for r in rows:
            if r.get("kept") and "episode_index" in r:
                E.append(dict(e=off + r["episode_index"], seed=r["seed"], hold=bool(r.get("hold")),
                              old="from_merged" in r, noisy="noise" in r and not r.get("hold"),
                              what=f"v5 episode {r['from_merged']}" if "from_merged" in r else "new"))
        off += info(home() / p)["total_episodes"]
    return E


def sheet(a):
    import cv2
    from lerobot.datasets import LeRobotDataset
    root, E = home() / a.out, episodes_of(a.prefixes)
    want = [("re-made cut", lambda x: x["old"] and not x["hold"]),
            ("re-made nothing-ripe", lambda x: x["old"] and x["hold"]),
            ("recorded cut", lambda x: not x["old"] and not x["hold"] and not x["noisy"]),
            ("recorded noisy cut", lambda x: not x["old"] and x["noisy"]),
            ("recorded nothing-ripe", lambda x: not x["old"] and x["hold"])]
    ds = LeRobotDataset(a.out, root=root)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, f in want:
        pick = next((x for x in E if f(x)), None)
        if pick is None:
            print(f"no {name} in {a.out}")
            continue
        ep = ds.meta.episodes[pick["e"]]
        i0, n = int(ep["dataset_from_index"]), int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        rows = []
        for k in (0, n // 2):
            item = ds[i0 + k]
            tiles = []
            for c in CAMS:
                im = (item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.putText(im, f"{c}, frame {k} of {n}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                tiles.append(im)
            rows.append(np.hstack(tiles))
        png = out / f"v6_check_{name.replace(' ', '_').replace('-', '_')}_ep{pick['e']:04d}.png"
        cv2.imwrite(str(png), np.vstack(rows))
        print(f"{png}: {name}, episode {pick['e']} of the merge ({pick['what']}, seed {pick['seed']}), {n} frames")


def videos(a):
    """Five review videos, the three cameras side by side (K7 approves five per dataset before training,
    2026-11-19): a re-made cut, a recorded clean cut (short), the longest recorded clean cut, a recorded noisy cut
    and a recorded nothing-ripe scene. Cut out of the dataset's own video files (what training decodes).
    ⚠️ One look only (K7 2026-11-19): delete the folder once the videos are approved or rejected."""
    import subprocess
    root, E = home() / a.out, episodes_of(a.prefixes)
    meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet"))))
    meta = meta.set_index("episode_index")
    for x in E:
        x["length"] = int(meta.loc[x["e"], "length"])
    E.sort(key=lambda x: x["length"])

    def of(f, where):
        K = [x for x in E if f(x)]
        return K[min(int(len(K) * where), len(K) - 1)] if K else None

    picks = {"remade_cut": of(lambda x: x["old"] and not x["hold"], 0.5),
             "new_clean_short": of(lambda x: not x["old"] and not x["hold"] and not x["noisy"], 0.1),
             "new_clean_longest": of(lambda x: not x["old"] and not x["hold"] and not x["noisy"], 1.0),
             "new_noisy": of(lambda x: not x["old"] and x["noisy"], 0.5),
             "new_hold": of(lambda x: not x["old"] and x["hold"], 0.5)}
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for kind, x in picks.items():
        if x is None:
            print(f"no {kind} in {a.out}")
            continue
        m, src = meta.loc[x["e"]], []
        for c in CAMS:
            k = f"videos/observation.images.{c}"
            f = root / k / f"chunk-{int(m[k + '/chunk_index']):03d}" / f"file-{int(m[k + '/file_index']):03d}.mp4"
            # ⚠️ start half a frame early and stop by frame count: cutting from/to the rounded timestamps took the
            # next episode's start frame along as a last frame (the trial's first five videos, 2026-11-27)
            src += ["-ss", f"{max(0.0, m[k + '/from_timestamp'] - 0.5 / FPS):.4f}", "-i", str(f)]
        mp4 = out / f"v6_check_ep{x['e']:04d}_{kind}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3",
                        "-frames:v", str(x["length"]), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        str(mp4)], check=True)
        print(f"{mp4}: {kind}, episode {x['e']} of the merge ({x['what']}, seed {x['seed']}), "
              f"{x['length']} frames = {x['length'] / FPS:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("parts")
    s.add_argument("prefixes", nargs="+", help="repo id prefixes of the parts, as for merge_parts.py")
    s.add_argument("--plan-dir", default=str(HERE / "seed_plan_v6"), help="the recorded parts' seed plans (pNN.jsonl)")
    for mode in ("merged", "sheet", "videos"):
        s = sub.add_parser(mode)
        s.add_argument("out", help="the merged repo id")
        s.add_argument("prefixes", nargs="+", help="the prefixes it was merged from, in merge_parts.py's order")
        if mode != "merged":
            s.add_argument("--out", dest="out_dir", required=True, help="a folder for the PNG sheets / the mp4 files")
    a = ap.parse_args()
    sys.exit({"parts": parts, "merged": merged, "sheet": sheet, "videos": videos}[a.mode](a))
