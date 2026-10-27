#!/usr/bin/env python3
"""Scripted sim episodes: the picker in sim/expert.py instead of a person on the leader (K7, 2026-10-27).

    python generate_scripted.py --pilot --watch          # 5 episodes into ..._pilot, MuJoCo window at real time
    python generate_scripted.py                          # 200 episodes into krish5831/blush_sim_scripted_v0
    lerobot-dataset-viz --repo-id krish5831/blush_sim_scripted_v0_pilot --episode-index 0   # what got recorded

Per scene seed: build the scene (domain randomisation on, vines settled 1 s), take the first frame (the
birdseye YOLO picks the target, exactly as in teleop), plan the pick, then
  1. dry run: play the plan with physics only (no frames), and judge it;
  2. if it passed: rebuild the same seed and play the same plan again, now recording every frame like
     lerobot-record does (same features, units, YOLO view, AV1). Judged again; a mismatch is discarded.
Kept if, at the LAST frame: the target is in the crate, no other fruit came off, no physics warnings, and
the arm is back at the start pose. Every attempt (kept or not, and why) goes to a JSONL log.

Mix of the KEPT episodes (K7, 2026-10-27), steered while generating, so failures don't skew it:
35 % same-truss scenes (two ripe fruits on one truss), 30 % twist-then-pull (70 % plain pull), 10 % near
miss + correction. Length is set by the motion: ~20-24 s in the dry runs (2026-10-27), varying with the
drawn speed and the route through the tucks.

Writes: the dataset (~/.cache/huggingface/lerobot/<repo-id>), the sim state log (~/.cache/blush_sim/states/),
the attempt log (~/.cache/blush_sim/scripted/<name>_<stamp>.jsonl). Never pushes to the Hub.
Run in the lerobot env, under the memory cap:  systemd-run --user --scope -p MemoryMax=3G python ...
⚠️ Yield is ~20-25 % of seeds (2026-10-27): most misses are a green neighbour on the same truss knocked
off as the jaw closes or pulls. Cap a run with --max-attempts.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert  # noqa: E402
import scene  # noqa: E402

TASK_ENV = HERE.parent / "harvest" / "harvest.env"
SAME_TRUSS_SHARE, TWIST_SHARE, NEAR_MISS_SHARE = 0.35, 0.30, 0.10   # K7, 2026-10-27
# Neighbour press limits per plan try on one scene (expert.Planner.plan strict=). One try: retrying
# the same scene, at the same or stricter limits, kept 2 of 12 seeds vs 6 of 30 with one try, at ~3x
# the time per failing seed (2026-10-27). Add entries here to retry.
PLAN_STRICT = (1.0,)
START_TOL_NORM = 3.0               # arm "back at the start pose" within this (norm units, every arm joint)
LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "scripted"


def task_string():
    """TASK from harvest/harvest.env: the identical string the real recordings use."""
    for line in TASK_ENV.read_text().splitlines():
        if line.startswith("TASK="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"no TASK= in {TASK_ENV}")


def phase_at(plan, t):
    """Which plan phase sim time t (s since the first frame) falls in."""
    k = int(t * expert.FPS)
    for name, n in plan.phases:
        if k < n:
            return name
        k -= n
    return "after"


def judge(robot, plan, t0):
    """(ok, reason) at the episode's last frame. t0: sim time of the first frame."""
    det, d, k = robot.det, robot.d, robot._tgt_k
    others = [f"{robot.info['fruit'][j]['tomato']} in {phase_at(plan, det.detached_at[j] - t0)}"
              for j in range(len(det.detached_at)) if j != k and not np.isnan(det.detached_at[j])]
    bad = sum(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                            mujoco.mjtWarning.mjWARN_BADQVEL))
    off = float(np.abs(robot._present_norm()[:5] - np.asarray(scene.START_NORM)[:5]).max())
    if bad:
        return False, f"{bad} physics warnings"
    if np.isnan(det.detached_at[k]):
        return False, "target never came off"
    if others:
        return False, f"knocked off {others}"
    if not scene.in_crate(d.xpos[robot._tgt_body]):
        return False, f"target not in the crate (at {np.round(d.xpos[robot._tgt_body], 3).tolist()})"
    if off > START_TOL_NORM:
        return False, f"arm {off:.1f} off the start pose"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--pilot", action="store_true", help="5 episodes into <repo-id>_pilot")
    ap.add_argument("--repo-id", default="krish5831/blush_sim_scripted_v0")
    ap.add_argument("--seed-start", type=int, default=100000, help="first scene seed (attempts count up)")
    ap.add_argument("--watch", action="store_true", help="MuJoCo window on the recording pass, at real time")
    ap.add_argument("--dry", action="store_true", help="dry runs only: plan + judge, write no dataset")
    ap.add_argument("--max-attempts", type=int, default=None)
    a = ap.parse_args()
    n_eps = 5 if a.pilot else a.episodes
    repo_id = a.repo_id + ("_pilot" if a.pilot else "")
    task = task_string()

    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    cfg = SO101MujocoConfig(id="sim", seed=a.seed_start, viewer=False, overlay="yolo", realtime=False,
                            fps=expert.FPS, domain_rand=True, settle_s=1.0, snap_to_first_action=False)
    robot = SO101Mujoco(cfg)

    ds = None
    if not a.dry:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.utils.constants import ACTION, OBS_STR, HF_LEROBOT_HOME
        from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
        root = HF_LEROBOT_HOME / repo_id
        if root.exists():
            sys.exit(f"⚠️ {root} exists already. Delete it or pick another --repo-id (nothing was written).")
        tap, _, rop = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(pipeline=tap, initial_features=create_initial_features(
                action=robot.action_features), use_videos=True),
            aggregate_pipeline_dataset_features(pipeline=rop, initial_features=create_initial_features(
                observation=robot.observation_features), use_videos=True))
        ds = LeRobotDataset.create(repo_id, expert.FPS, root=root, robot_type=robot.name, features=features,
                                   use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1"),
                                   image_writer_threads=4)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{repo_id.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    print(f"dataset {repo_id if ds else '(dry, none)'}  task {task!r}\nattempt log {log_path}", flush=True)

    robot.connect()
    kept = dict(n=0, same=0, twist=0, miss=0)
    want_same = round(SAME_TRUSS_SHARE * n_eps)
    seed = a.seed_start - 1
    attempts, t_start = 0, time.time()
    try:
        while kept["n"] < n_eps and (a.max_attempts is None or attempts < a.max_attempts):
            seed += 1
            attempts += 1
            rec = dict(seed=seed)

            def skip(why, **kw):
                rec.update(kept=False, reason=why, **kw)
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  seed {seed}: skip, {why}", flush=True)

            # --- dry run ---
            robot.config.viewer = False
            robot.new_scene(seed, log=False)
            same = bool(robot.info["same_truss"])
            rec["same_truss"] = same
            if same and kept["same"] >= want_same:
                skip("same-truss quota full"); continue
            if not same and kept["n"] - kept["same"] >= n_eps - want_same:
                skip("separate-truss quota full"); continue
            if not np.all(np.isnan(robot.det.detached_at)):
                skip("a fruit fell off while the vines settled"); continue
            robot.get_observation()
            if robot._tgt_k is None:
                skip(f"no target ({robot._box_info['reason']})"); continue
            left = n_eps - kept["n"]
            rng = np.random.default_rng([seed, 0x5C])
            twist = bool(rng.random() < max(0, round(TWIST_SHARE * n_eps) - kept["twist"]) / left)
            near_miss = bool(rng.random() < max(0, round(NEAR_MISS_SHARE * n_eps) - kept["miss"]) / left)
            rec.update(target=robot._box_info["target_name"], target_source=robot._box_info["source"],
                       twist=twist, near_miss=near_miss)
            fails, ok = [], False
            for k, strict in enumerate(PLAN_STRICT):
                if k:                                      # a fresh copy of the scene for the next plan
                    robot.new_scene(seed, log=False)
                    robot.get_observation()
                try:
                    planner = expert.Planner(robot.m, robot.d, robot.info, robot._tgt_k)
                    plan = planner.plan(rng, twist=twist, near_miss=near_miss, strict=strict)
                except expert.PlanError as e:
                    fails.append(f"no plan: {e}")
                    if k == 0:
                        break                              # stricter won't find what looser couldn't
                    continue
                rec["plan"] = plan.meta
                t0 = robot.d.time
                while not plan.done:
                    robot.send_action(plan.act(robot._present_norm()))
                ok, why = judge(robot, plan, t0)
                if ok:
                    break
                fails.append(f"dry run: {why}")
            rec["plan_tries"] = k + 1
            if not ok:
                skip(" | ".join(fails)); continue
            if a.dry:
                rec.update(kept=True, reason="dry ok")
                log.write(json.dumps(rec) + "\n"); log.flush()
                kept["n"] += 1; kept["same"] += same; kept["twist"] += twist; kept["miss"] += near_miss
                print(f"  seed {seed}: dry ok ({kept['n']}/{n_eps}) {plan.meta['seconds']} s", flush=True)
                continue

            # --- recording pass: same seed, same plan ---
            robot.config.viewer = a.watch
            robot.new_scene(seed, log=True)
            plan.reset()
            t_ep = time.time()
            obs = robot.get_observation()
            t0 = robot.d.time
            if robot._box_info["target_name"] != rec["target"]:
                skip(f"recording picked another target ({robot._box_info['target_name']})"); continue
            while not plan.done:
                t_f = time.time()
                act = plan.act([obs[f"{j}.pos"] for j in expert.arm_map.JOINTS])
                sent = robot.send_action(act)
                ds.add_frame({**build_dataset_frame(ds.features, rop(obs), prefix=OBS_STR),
                              **build_dataset_frame(ds.features, tap((sent, obs)), prefix=ACTION),
                              "task": task})
                obs = robot.get_observation()
                if a.watch:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
            ok, why = judge(robot, plan, t0)
            if not ok:
                ds.clear_episode_buffer()
                skip(f"recording differs from the dry run: {why}"); continue
            ds.save_episode()
            ep = kept["n"]
            kept["n"] += 1; kept["same"] += same; kept["twist"] += twist; kept["miss"] += near_miss
            robot.extra = dict(episode_index=ep, dataset=repo_id, scripted=plan.meta)
            rec.update(kept=True, reason="ok", episode_index=ep, wall_s=round(time.time() - t_ep, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            m = plan.meta
            print(f"✅ episode {ep} ({kept['n']}/{n_eps}): seed {seed}, {m['seconds']} s, {m['detach']}"
                  f"{', near miss' if m['near_miss'] else ''}{', same truss' if same else ''}, "
                  f"pitch {m['approach_pitch_deg']} tilt {m['jaw_tilt_deg']}  [{time.time() - t_ep:.0f} s wall]",
                  flush=True)
    finally:
        robot.disconnect()                 # writes the last scene's state log
        if ds is not None:
            ds.finalize()
        n = max(kept["n"], 1)
        summary = dict(summary=True, kept=kept["n"], attempts=attempts, same_truss=kept["same"],
                       twist=kept["twist"], near_miss=kept["miss"], minutes=round((time.time() - t_start) / 60, 1))
        log.write(json.dumps(summary) + "\n"); log.close()
        print(f"\nkept {kept['n']} of {attempts} attempts in {summary['minutes']} min: same-truss "
              f"{kept['same'] / n:.0%}, twist {kept['twist'] / n:.0%}, near miss {kept['miss'] / n:.0%}"
              f"\nlog {log_path}", flush=True)


if __name__ == "__main__":
    main()
