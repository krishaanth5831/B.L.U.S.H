#!/usr/bin/env python3
"""Re-make dataset v5's pictures from the simulator's state logs: other cameras, and stored by the picture rule.

K7 2026-11-26: v6 keeps the old teacher, gripper and base, records 600 more episodes and puts them in the same
dataset as v5's 800. The 800 have two cameras at the old spots and pictures stored the faulty way (AV1 with
look-ahead, post-mortem 4.5). The recorder logged the simulator's full state at every recorded frame
(~/.cache/blush_sim/states, 800 scenes), so their pictures can be rendered again with rig "trio" and stored through
picture_store.py, while every joint reading and every command stays exactly the recorded one.

    python remake_pictures.py align                       # read-only: logs against the dataset's joint data, all 800
    python remake_pictures.py check --n 3                  # read-only: re-renders the OLD rig, compares with the
                                                           #   old videos (downloads those episodes' video files)
    python remake_pictures.py dry --shots DIR              # read-only: the re-make's own path with the NEW rig on a
                                                           #   few episodes, frames built and validated, no dataset
    python remake_pictures.py dry --start-only --shard 0/4 # read-only: every episode's start frame with the new rig:
                                                           #   same target as recorded? does each camera get a cue?
    python remake_pictures.py part --part 0 --repo-id krish5831/blush_sim_teacher_v6r
                                                           # ⚠️ WRITES a dataset part: only with K7's go

⚠️ `part` has not been run (2026-11-26): no re-recording without K7's go. First step of that go: one part, then
`picture_store.py gate` on it and a look at its pictures, before the other 31.

How an episode is re-made (frames()):
  1. The scene is rebuilt from its seed in v5's world (truss_scene.set_world("crate23")) and decided with the bracket
     that stood there when it was recorded (layout_rig="base"): the same plants, the same ripe truss. Checked
     against the log: number of joints and constraints, the target's name.
  2. The logged state of frame 0 is set and the start frame taken: the target's box + cut point are fixed on each
     fixed camera of the new rig, as the recorder does (the cue at the teacher's cut_u).
  3. Per recorded frame k: the logged state k is set, the pictures rendered. The joint readings of the re-made
     frame must equal the dataset's (STATE_TOL), frame by frame, or the episode is refused.
  4. Stored with the dataset's own joint readings and commands (bit for bit), the new rig's cue numbers, the new
     pictures through picture_store.encoder(), and the picture gate's fingerprints.
What differs from the recording: the pictures show the state at the END of the simulator step (the recorder
rendered the geometry of one 2 ms physics step earlier; sub-pixel), MuJoCo 4.0.1 renders them (v5: 3.8.1), and
the brackets shown are the new rig's.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "whatif"))
import numpy as np  # noqa: E402

import rebuild_v5_check as rc  # noqa: E402

OLD_REPO = "krish5831/blush_sim_teacher_v5"
PART_EPISODES = 25             # sim/scripts/record_parts.sh: v5 was recorded in 32 parts of 25
STATE_TOL = 1e-3               # norm units: a re-made frame's joint readings against the dataset's (float32 rounding
#                                is 4e-6, measured over all 383,877 frames 2026-11-26; a wrong frame is off by ~1)
LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "truss"     # as generate_truss.LOG_DIR
# Old episodes a re-make leaves out, by seed. ⚠️ ONLY these: any other episode whose scene does not come back stops
# the part, because a wrong world, another MuJoCo or missing logs would otherwise give a quiet, nearly empty part.
LEFT_OUT = {
    # v5 episode 353 (part 14): its seed gives a scene of 4 trusses today (nq 399, neq 45), the log has nq 498,
    # neq 54; the same with the old bracket's rig, twice in a row. The other 799 rebuild as recorded
    # (dry --start-only over all 800, 2026-11-27). Cause not chased (v5 was recorded on MuJoCo 3.8.1).
    10014003: "its seed no longer gives the recorded scene",
}


class Mismatch(Exception):
    pass


def nums(v, n=3):
    return [round(float(x), n) for x in v]


def old_episodes():
    """Every episode of dataset v5 in the merged dataset's order: rebuild_v5_check.v5_scenes() rows + 'part', 'e'."""
    S = rc.v5_scenes()
    for x in S:
        x["part"] = int(x["dataset"].split("_p")[-1])
        x["e"] = x["part"] * PART_EPISODES + int(x["episode"])
    S.sort(key=lambda x: x["e"])
    assert [x["e"] for x in S] == list(range(len(S))), "the logs do not cover v5's episodes one to one"
    return S


def joint_data():
    """{episode: (state (N, 6), action (N, 6), cue (6,))} float32, from the dataset's own data file (30 MB; fetched
    into the Hugging Face cache on first use, no pictures)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    f = hf_hub_download(OLD_REPO, "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(f).sort_values(["episode_index", "frame_index"])
    out = {}
    for e, g in df.groupby("episode_index"):
        assert (g["frame_index"].to_numpy() == np.arange(len(g))).all()
        out[int(e)] = (np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                       np.stack(g["action"].to_numpy()).astype(np.float32),
                       np.asarray(g["observation.environment_state"].iloc[0], np.float32))
    return out


def make_robot(rig, layout_rig, seed):
    import expert
    import truss_scene as ts
    from lerobot.utils.import_utils import register_third_party_plugins
    ts.set_world("crate23")
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=seed, viewer=False, overlay="true_box", realtime=False, fps=expert.FPS, domain_rand=False,
        settle_s=1.0, snap_to_first_action=False, yolo_weights=None, state_log_dir=None, rig=rig,
        layout_rig=layout_rig))
    robot.connect()
    return robot


def frames(robot, x, state):
    """Per recorded frame k of episode x: (k, observation) rendered from the logged state, the cue drawn as the
    recorder draws it. Raises Mismatch when the rebuilt scene or a frame's joint readings are not the recorded ones."""
    import arm_map
    import mujoco
    z = np.load(x["npz"], allow_pickle=True)
    if len(z["qpos"]) != len(state) + 1:
        raise Mismatch(f"{len(z['qpos'])} logged states for {len(state)} frames (want one more)")
    robot.new_scene(x["seed"], log=False, hold=x["hold"], cut_u=0.5 if x["hold"] else x["cut_u"])
    m, d = robot.m, robot.d
    if (m.nq, m.neq) != (z["qpos"].shape[1], z["eq_active"].shape[1]):
        raise Mismatch(f"rebuilt scene has nq {m.nq}, neq {m.neq}; the log {z['qpos'].shape[1]}, "
                       f"{z['eq_active'].shape[1]}")

    def put(k):
        d.qpos[:] = z["qpos"][k]
        d.eq_active[:] = z["eq_active"][k]
        mujoco.mj_forward(m, d)

    put(0)
    o = robot.get_observation()                 # the start frame: picks the target, fixes the cue on every camera
    tgt = None if robot._tgt_k is None else robot.info["trusses"][robot._tgt_k]["name"]
    if tgt != x["target"]:
        raise Mismatch(f"rebuilt scene's target is {tgt}, recorded {x['target']}")
    for k in range(len(state)):
        if k:
            put(k)
            o = robot.get_observation()
        off = float(np.abs(np.array([o[f"{j}.pos"] for j in arm_map.JOINTS], np.float64) - state[k]).max())
        if off > STATE_TOL:
            raise Mismatch(f"frame {k}: joint readings {off:.4f} off the dataset's")
        yield k, o


def dataset_features(robot):
    """(features, tap, rop): the dataset's features exactly as generate_truss.py builds them for this robot's rig."""
    import truss_target as ttg
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import OBS_ENV_STATE
    from lerobot.utils.feature_utils import combine_feature_dicts
    tap, _, rop = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=tap, initial_features=create_initial_features(
            action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=rop, initial_features=create_initial_features(
            observation=robot.observation_features), use_videos=True))
    features[OBS_ENV_STATE] = {"dtype": "float32", "shape": [len(ttg.ENV_NAMES)], "names": ttg.ENV_NAMES}
    return features, tap, rop


def frame_of(features, tap, rop, o, st_k, act_k, env):
    """One dataset frame as generate_truss.py's recording callback builds it, then the joint readings and the command
    replaced by the dataset's own (bit for bit): the new pictures, the old numbers, this rig's cue."""
    import arm_map
    import generate_truss as gt
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    obs_feats = {k: v for k, v in features.items() if k != OBS_ENV_STATE}
    sent = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, act_k)}
    fr = {**build_dataset_frame(obs_feats, rop(o), prefix=OBS_STR),
          **build_dataset_frame(features, tap((sent, o)), prefix=ACTION), "task": gt.TASK}
    assert fr["observation.state"].shape == st_k.shape and fr[ACTION].shape == act_k.shape
    fr["observation.state"], fr[ACTION] = st_k, act_k
    fr[OBS_ENV_STATE] = env
    return fr


def cue_facts(robot):
    """Per fixed camera of the rig, at the start frame: (has a box, cut pixel inside the frame)."""
    import truss_target as ttg
    out = {}
    for c in robot.fixed:
        box, cut = robot._views.get(c, (None, None))
        out[c] = (box is not None, cut is not None and 0 <= cut[0] < ttg.W and 0 <= cut[1] < ttg.H)
    return out


def align(a):
    """Read-only, no rendering: per episode the logged joint readings against the dataset's, frame by frame."""
    S, J = old_episodes(), joint_data()
    worst, bad, cue_bad = 0.0, [], 0
    for x in S:
        st, _, cue = J[x["e"]]
        z = np.load(x["npz"], allow_pickle=True)
        ls = np.asarray(z["state"], np.float64)
        ok = len(ls) == len(st) + 1
        off = float(np.abs(ls[:len(st)] - st).max()) if ok else np.inf
        worst = max(worst, off if ok else 0.0)
        if not ok or off > STATE_TOL:
            bad.append((x["e"], x["seed"], len(st), len(ls), off))
        want = np.asarray(x["env_state"] if x["env_state"] is not None else np.zeros(6), np.float32)
        cue_bad += bool(np.abs(cue - want).max() > 1e-4)
    print(f"{len(S)} episodes, {sum(len(J[x['e']][0]) for x in S)} frames: {len(S) - len(bad)} line up frame for "
          f"frame with the logs (largest difference of a joint reading {worst:.2e} norm units), {len(bad)} do not"
          f"{': ' + str(bad[:5]) if bad else ''}; the dataset's cue differs from the log's in {cue_bad} episodes")
    print("align:", "PASS" if not bad and not cue_bad else "FAIL")


def check(a):
    """Read-only: re-render --n old episodes with the OLD rig and compare each frame with the old dataset's stored
    frame. They cannot be equal (v5's video storage loses detail, another MuJoCo version rendered it); what must
    hold is that every frame shows the same thing: a small difference that stays small through the cut and carry."""
    import torch
    from lerobot.datasets import LeRobotDataset
    S, J = old_episodes(), joint_data()
    cuts = [x for x in S if not x["hold"]]
    pick = [cuts[i] for i in np.linspace(0, len(cuts) - 1, a.n).astype(int)] if a.episodes is None else \
        [S[int(e)] for e in a.episodes.split(",")]
    ds = LeRobotDataset(OLD_REPO, episodes=[x["e"] for x in pick], video_backend="torchcodec")
    print("old videos at", ds.root, flush=True)
    robot = make_robot("base", None, pick[0]["seed"])
    row0 = {}
    for i in range(len(ds.hf_dataset)):          # dataset row of each picked episode's frame 0
        e = int(ds.hf_dataset[i]["episode_index"])
        row0.setdefault(e, i)
    ok_all = True
    try:
        for x in pick:
            st = J[x["e"]][0]
            diffs = {c: [] for c in robot.cam_names}
            try:
                for k, o in frames(robot, x, st):
                    if k % a.every:
                        continue
                    item = ds[row0[x["e"]] + k]
                    assert int(item["frame_index"]) == k and int(item["episode_index"]) == x["e"]
                    for c in robot.cam_names:
                        old = (item[f"observation.images.{c}"] * 255.0).round().to(torch.float32)
                        new = torch.from_numpy(np.ascontiguousarray(o[c])).permute(2, 0, 1).to(torch.float32)
                        diffs[c].append(float((old - new).abs().mean()))
            except Mismatch as err:
                print(f"DIFF episode {x['e']} (seed {x['seed']}): {err}", flush=True)
                ok_all = False
                continue
            line = ", ".join(f"{c}: mean {np.mean(v):.2f}, worst frame {np.max(v):.2f} (frame "
                             f"{int(np.argmax(v)) * a.every})" for c, v in diffs.items())
            good = all(np.max(v) < a.max_diff for v in diffs.values())
            ok_all = ok_all and good
            print(f"{'OK  ' if good else 'DIFF'} episode {x['e']} (seed {x['seed']}, {x['target']}, {len(st)} "
                  f"frames, every {a.every}.): grey levels off, {line}", flush=True)
    finally:
        robot.disconnect()
    print(f"\ncheck: {'PASS' if ok_all else 'FAIL'} (every compared frame under {a.max_diff} grey levels mean "
          f"difference)\n⚠️ the old videos of these episodes are now at {ds.root / 'videos'}: delete them after the "
          f"check (K7 2026-11-16: datasets live on the Hub, no local copies)")


def part(a):
    """⚠️ WRITES a dataset part (K7's go first): the old episodes part*25 .. part*25+24, re-made."""
    import cameras
    import expert
    import picture_store
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    S, J = old_episodes(), joint_data()
    mine = [x for x in S if x["part"] == a.part]
    pid = f"{a.repo_id}_p{a.part:02d}"
    if (HF_LEROBOT_HOME / pid).exists():
        sys.exit(f"⚠️ {HF_LEROBOT_HOME / pid} exists already: nothing was written")
    robot = make_robot(a.rig, "base", mine[0]["seed"])
    features, tap, rop = dataset_features(robot)
    ds = LeRobotDataset.create(pid, expert.FPS, root=HF_LEROBOT_HOME / pid, robot_type=robot.name, features=features,
                               use_videos=True, rgb_encoder=picture_store.encoder(), image_writer_threads=4)
    pg = picture_store.Gate(cameras.RIG_CAMERAS[a.rig])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"{pid.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl").open("w")
    done, no_cue, left_out = 0, 0, []
    try:
        for x in mine:
            t0 = time.time()
            if x["seed"] in LEFT_OUT:                # by name only (LEFT_OUT's comment)
                left_out.append((x["e"], x["seed"], LEFT_OUT[x["seed"]]))
                log.write(json.dumps(dict(seed=x["seed"], hold=x["hold"], target=x["target"], kept=False,
                                          reason=f"not re-made: {LEFT_OUT[x['seed']]}", part=pid,
                                          from_dataset=x["dataset"], from_episode=int(x["episode"]),
                                          from_merged=x["e"])) + "\n")
                log.flush()
                print(f"⚠️ v5 episode {x['e']} (seed {x['seed']}) is left out: {LEFT_OUT[x['seed']]}", flush=True)
                continue
            st, act, _ = J[x["e"]]
            pg.drop()
            env, cue = None, None
            for k, o in frames(robot, x, st):       # a Mismatch stops the part: nothing half-right is kept
                if env is None:
                    env, cue = robot.env_state(), cue_facts(robot)   # this rig's cue, fixed at the start frame
                pg.note(o)
                ds.add_frame(frame_of(features, tap, rop, o, st[k], act[k], env))
            ds.save_episode(parallel_encoding=False)
            pg.keep(ds.root, ds.meta.total_episodes - 1)
            # ⚠️ an all-zero box means "nothing ripe" to the policy and to eval_truss --start-guard
            no_cue += bool(not x["hold"] and not np.any(env[:4]))
            log.write(json.dumps(dict(seed=x["seed"], hold=x["hold"], target=x["target"], env=nums(env, 4),
                                      cue={c: list(map(bool, v)) for c, v in cue.items()},
                                      kept=True, reason="re-made", episode_index=done, part=pid,
                                      from_dataset=x["dataset"], from_episode=int(x["episode"]), from_merged=x["e"],
                                      frames=len(st), wall_s=round(time.time() - t0, 1))) + "\n")
            log.flush()
            done += 1
            print(f"✅ episode {done}/{len(mine)}: v5 episode {x['e']} (seed {x['seed']}, "
                  f"{'HOLD' if x['hold'] else x['target']}), {len(st)} frames  [{time.time() - t0:.0f} s]", flush=True)
    finally:
        robot.disconnect()
        ds.finalize()
        log.close()
    print(f"💾 part {pid} finalized: {done} of {len(mine)} episodes"
          + (f"\n⚠️ left out by name (LEFT_OUT): {left_out}" if left_out else "")
          + (f"\n⚠️ {no_cue} cut episodes have an all-zero box in the cue numbers: the first camera of rig {a.rig} "
             f"does not see their truss at the start (they would read as 'nothing ripe')" if no_cue else ""))


def dry(a):
    """Read-only, no dataset is created: the re-make's own path with the NEW rig.
    Default: a few episodes (a hold, a far stem, a near one, a high truss, the last one) frame by frame; every
    --every-th frame is built into the dataset frame `part` would add and checked by LeRobot's own validate_frame;
    seconds per frame; --shots DIR writes each episode's start / before the cut / cut / carry frames, all cameras as
    the policy gets them (the cue drawn, the new brackets in the scene).
    --start-only [--shard i/n]: only the start frame, of every episode: is the rebuilt scene's target the recorded
    one, and does each fixed camera get a box and a cut pixel inside its frame."""
    from lerobot.datasets.feature_utils import validate_frame
    S, J = old_episodes(), joint_data()
    if a.start_only:
        i, n = (int(v) for v in a.shard.split("/"))
        pick = S[i::n]
    elif a.episodes:
        pick = [S[int(e)] for e in a.episodes.split(",")]
    else:
        cuts = [x for x in S if not x["hold"]]
        first = lambda f: next((x for x in cuts if f(x["target"])), None)   # noqa: E731
        pick = [next(x for x in S if x["hold"]), first(lambda t: t.startswith("s4")) or
                first(lambda t: t.startswith("s3")), first(lambda t: t == "s0t0"), first(lambda t: t.endswith("t2")),
                cuts[-1]]
        pick = list({x["e"]: x for x in pick if x is not None}.values())
    robot = make_robot(a.rig, "base", pick[0]["seed"])
    features, tap, rop = dataset_features(robot)
    # what a dataset checks its frames against: its info turns the shapes into tuples when it is created (lerobot
    # DatasetInfo.__post_init__), and validate_frame compares a frame's shape with that. ⚠️ Against the raw dict the
    # cue feature's list shape [6] fails every frame (the first dry run, 2026-11-26).
    checked = {k: {**v, "shape": tuple(v["shape"])} for k, v in features.items()}
    cams = list(robot.cam_names)
    bad, known, n_frames, t_frames, tally = [], [], 0, 0.0, {c: [0, 0] for c in robot.fixed}
    no_cue, n_cut = [], 0
    if a.shots:
        import cv2
        os.makedirs(a.shots, exist_ok=True)
    try:
        for x in pick:
            st, act, _ = J[x["e"]]
            z = np.load(x["npz"], allow_pickle=True)
            ch = np.flatnonzero((np.asarray(z["eq_active"])[1:] != np.asarray(z["eq_active"])[:-1]).any(1))
            cut_k = int(ch[0]) + 1 if len(ch) else None          # the first frame after a constraint let go: the cut
            want = {0, len(st) // 2} if cut_k is None else {0, max(0, cut_k - 15), min(cut_k, len(st) - 1),
                                                            min(len(st) - 1, cut_k + (len(st) - cut_k) // 2)}
            env, cue, shots, t0 = None, None, {}, time.time()
            try:
                for k, o in frames(robot, x, st):
                    if env is None:
                        env, cue = robot.env_state(), cue_facts(robot)
                    if a.start_only:
                        break
                    if k % a.every == 0:
                        validate_frame(frame_of(features, tap, rop, o, st[k], act[k], env), checked)
                    if a.shots and k in want:
                        shots[k] = np.hstack([o[c] for c in cams])
            except Mismatch as err:
                (known if x["seed"] in LEFT_OUT else bad).append((x["e"], x["seed"], str(err)))
                print(f"{'LEFT OUT (known)' if x['seed'] in LEFT_OUT else 'DIFF'} v5 episode {x['e']} "
                      f"(seed {x['seed']}): {err}", flush=True)
                continue
            if x["seed"] in LEFT_OUT:
                print(f"⚠️ v5 episode {x['e']} (seed {x['seed']}) is on LEFT_OUT but rebuilt as recorded", flush=True)
            if not x["hold"]:
                n_cut += 1
                for c, (b, u) in cue.items():
                    tally[c][0] += b
                    tally[c][1] += u
                if not np.any(env[:4]):
                    no_cue.append(x["e"])
            if a.start_only:
                print(f"ok {x['e']} seed {x['seed']} {'HOLD' if x['hold'] else x['target']} cue "
                      f"{ {c: list(map(int, v)) for c, v in cue.items()} } env {nums(env)}", flush=True)
                continue
            n_frames += len(st)
            t_frames += time.time() - t0
            if a.shots:
                import cv2
                rows = []
                for k in sorted(shots):
                    im = np.ascontiguousarray(shots[k][..., ::-1])
                    cv2.putText(im, f"v5 episode {x['e']} seed {x['seed']} {'HOLD' if x['hold'] else x['target']} "
                                f"frame {k}" + (" (cut)" if k == cut_k else ""), (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    rows.append(im)
                f = os.path.join(a.shots, f"remake_{a.rig}_e{x['e']:03d}.png")
                cv2.imwrite(f, np.vstack(rows))
            print(f"OK   v5 episode {x['e']} (seed {x['seed']}, {'HOLD' if x['hold'] else x['target']}): {len(st)} "
                  f"frames in {time.time() - t0:.0f} s, cue {nums(env)}, per camera (box, cut pixel in "
                  f"frame) {cue}" + (f", pictures {f}" if a.shots else ""), flush=True)
    finally:
        robot.disconnect()
    print(f"\ndry ({'start frames' if a.start_only else 'whole episodes'}, rig {a.rig}, cameras {cams}): "
          f"{len(pick) - len(bad) - len(known)} of {len(pick)} episodes rebuilt as recorded"
          + (f", {len(known)} left out by name (LEFT_OUT): {known}" if known else "")
          + (f", {len(bad)} NOT: {bad}" if bad else ""))
    print(f"cut episodes: {n_cut}; per fixed camera, a box / the cut pixel inside the frame at the start: "
          + ", ".join(f"{c} {v[0]} / {v[1]}" for c, v in tally.items())
          + f"; all-zero box in the cue numbers (reads as 'nothing ripe'): {len(no_cue)} {no_cue[:20]}")
    if n_frames:
        print(f"{n_frames} frames at {t_frames / n_frames:.3f} s each (scene build included): all 383,877 frames "
              f"would take {383877 * t_frames / n_frames / 3600:.1f} h on one process, before storing")
    print("dry:", "PASS" if not bad else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("align")
    s = sub.add_parser("check")
    s.add_argument("--n", type=int, default=3, help="cut episodes spread over the recording")
    s.add_argument("--episodes", help="or: these merged v5 episode numbers, comma list")
    s.add_argument("--every", type=int, default=10, help="compare every k-th frame")
    s.add_argument("--max-diff", type=float, default=8.0,
                   help="largest mean difference of a frame, grey levels of 255 (guess: v5's AV1 storage alone is "
                        "about 2; a truss or the arm in another place is far more)")
    s = sub.add_parser("dry")
    s.add_argument("--rig", default="trio", choices=("trio", "base"))
    s.add_argument("--episodes", help="these merged v5 episode numbers, comma list (default: five chosen ones)")
    s.add_argument("--every", type=int, default=5, help="build and validate every k-th frame")
    s.add_argument("--shots", help="a folder for one picture sheet per episode")
    s.add_argument("--start-only", action="store_true", help="only the start frame, of every episode (see --shard)")
    s.add_argument("--shard", default="0/1", help="i/n: every n-th episode from the i-th (parallel start-only runs)")
    s = sub.add_parser("part")
    s.add_argument("--part", type=int, required=True, help=f"0-31: the old episodes part*{PART_EPISODES} and up")
    s.add_argument("--repo-id", required=True, help="the parts are <repo-id>_pNN (merge_parts.py merges them)")
    s.add_argument("--rig", default="trio", choices=("trio", "base"))
    a = ap.parse_args()
    dict(align=align, check=check, part=part, dry=dry)[a.mode](a)
