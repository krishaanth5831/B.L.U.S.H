#!/usr/bin/env python3
"""Scripted whole-truss harvesting episodes: truss_expert.py instead of a person on the leader (K7, 2026-11-03).

    python generate_truss.py --pilot --watch            # 5 episodes into ..._pilot, MuJoCo window at real time
    python generate_truss.py --dry --episodes 20        # plan + dry run + judge only: keep rate and time, no dataset
    python generate_truss.py                            # 200 episodes into krish5831/blush_sim_truss_v1

Per scene seed (as generate_scripted.py): build the truss scene (domain randomisation on, settle 1 s), take
the first frame (the truss YOLO, or the true box, picks the ready truss and its cut point), plan, then
  1. dry run: play the plan with physics only, judge it;
  2. if it passed: rebuild the same seed, play the same plan again and record every frame like lerobot-record
     does, plus observation.environment_state = the target truss's box + cut point (truss_target.ENV_NAMES).
     Judged again; a mismatch is discarded.
Kept if, at the last frame (judge()): the target truss was cut and lies whole in the crate; no other truss was
cut; no fruit came off anywhere (the target's included); no other truss or its fruit moved more than
NEIGHBOUR_MOVE_M at any frame (K7: "doesn't affect other tomatoes"); no physics warnings; the arm is back at
the start pose. A scene that fails is planned again (PLAN_TRIES), with a fresh draw of cut point and routes.

Mix: MISS_SHARE of the kept episodes first close SHORT of the stalk, open, back off and go in again (recovery
from a missed close; act_box_v1 never saw one, 2026-11-03 analysis), steered while generating.

Writes: the dataset (~/.cache/huggingface/lerobot/<repo-id>), the sim state log (~/.cache/blush_sim/states/),
the attempt log (~/.cache/blush_sim/truss/<name>_<stamp>.jsonl). Never pushes to the Hub.
Run in the lerobot env under the memory cap (README / CLAUDE.md):
  BLUSH_RIPENESS_PYTHON=$PWD/scripts/capped_ripeness_python.sh systemd-run --user --scope -q -p MemoryMax=9G \\
      -p MemorySwapMax=0 python generate_truss.py ...
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
import truss_expert as tx  # noqa: E402
import truss_scene as ts  # noqa: E402
import truss_target as ttg  # noqa: E402

# ⚠️ Not harvest/harvest.env's TASK: that string is shared with the real single-fruit recordings. Truss harvesting
# is a different task; this string goes with it (K7 decides the real rig's wording when it records trusses).
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
MISS_SHARE = 0.25                # K7 2026-11-03 (plan point 4: ~25 % recovery episodes)
PLAN_TRIES = 3                   # plans per seed before it's skipped (retry instead of discarding, plan point 2)
NEIGHBOUR_MOVE_M = 0.015         # any other truss's fruit may swing this far at most (plan point 2, guess)
START_TOL_NORM = 3.0             # generate_scripted.START_TOL_NORM
LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "truss"
DEFAULT_REPO = "krish5831/blush_sim_truss_v1"
YOLO_TRUSS = HERE.parent / "runs" / "yolo_truss" / "truss_v1" / "weights" / "best.pt"


class Watch:
    """Per frame: how far every other truss's fruit has moved from where it hung at the first frame."""

    def __init__(self, robot):
        m, d, k = robot.m, robot.d, robot._tgt_k
        self.ids = np.array([m.body(f["tomato"]).id for j, t in enumerate(robot.info["trusses"]) if j != k
                             for f in t["fruit"]], int)
        self.names = [f["tomato"] for j, t in enumerate(robot.info["trusses"]) if j != k for f in t["fruit"]]
        self.p0 = robot.d.xpos[self.ids].copy() if len(self.ids) else np.zeros((0, 3))
        self.max = 0.0
        self.worst = None
        self.d = d

    def step(self):
        if not len(self.ids):
            return
        dist = np.linalg.norm(self.d.xpos[self.ids] - self.p0, axis=1)
        i = int(np.argmax(dist))
        if dist[i] > self.max:
            self.max, self.worst = float(dist[i]), self.names[i]


def phase_at(plan, t):
    k = int(t * expert.FPS)
    for name, n in plan.phases:
        if k < n:
            return name
        k -= n
    return "after"


def judge(robot, plan, t0, watch):
    d, k, info = robot.d, robot._tgt_k, robot.info
    bad = sum(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                            mujoco.mjtWarning.mjWARN_BADQVEL))
    if bad:
        return False, f"{bad} physics warnings"
    cut = robot.cutters[k]
    if cut.cut_t is None:
        return False, f"target never cut (blade squeeze peak {cut.peak['squeeze']:.1f} N of {cut.cut_force:.1f})"
    wrong = [info["trusses"][j]["name"] for j, c in enumerate(robot.cutters) if j != k and c.cut_t is not None]
    if wrong:
        return False, f"cut the wrong truss {wrong}"
    off = [f"{info['fruit'][j]['tomato']} in {phase_at(plan, robot.det.detached_at[j] - t0)}"
           for j in range(len(info["fruit"])) if not np.isnan(robot.det.detached_at[j])]
    if off:
        return False, f"fruit came off {off}"
    if watch.max > NEIGHBOUR_MOVE_M:
        return False, f"moved a neighbour {watch.max * 1000:.0f} mm ({watch.worst})"
    if not ts.in_crate(info, d, robot.m, k):
        return False, "target truss not (all) in the crate"
    o = float(np.abs(robot._present_norm()[:5] - np.asarray(scene.START_NORM)[:5]).max())
    if o > START_TOL_NORM:
        return False, f"arm {o:.1f} off the start pose"
    return True, "ok"


def play(robot, plan, on_frame=None):
    """Plays a plan from the scene's start frame; returns (t0, Watch)."""
    plan.reset()
    watch = Watch(robot)
    t0 = robot.d.time
    obs = None
    while not plan.done:
        state = robot._present_norm() if obs is None else [obs[f"{j}.pos"] for j in expert.arm_map.JOINTS]
        act = plan.act(state)
        sent = robot.send_action(act)
        if on_frame is not None:
            obs = on_frame(sent)
        watch.step()
    return t0, watch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--pilot", action="store_true", help="5 episodes into <repo-id>_pilot")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--seed-start", type=int, default=200000)
    ap.add_argument("--watch", action="store_true", help="MuJoCo window on the recording pass, at real time")
    ap.add_argument("--dry", action="store_true", help="dry runs only: plan + judge, write no dataset")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--yolo-weights", type=pathlib.Path, default=YOLO_TRUSS if YOLO_TRUSS.exists() else None,
                    help="truss YOLO for the start frame + overlay (default: runs/yolo_truss/truss_v1 if trained, "
                         "else the true box, no YOLO)")
    ap.add_argument("--no-yolo", action="store_true", help="true box at the start frame even if the truss YOLO exists")
    a = ap.parse_args()
    if a.no_yolo:
        a.yolo_weights = None
    n_eps = 5 if a.pilot else a.episodes
    repo_id = a.repo_id + ("_pilot" if a.pilot else "")

    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    overlay = "yolo" if a.yolo_weights else "true_box"
    cfg = SO101MujocoConfig(id="sim", seed=a.seed_start, viewer=False, overlay=overlay, realtime=False,
                            fps=expert.FPS, domain_rand=True, settle_s=1.0, snap_to_first_action=False, task="truss",
                            yolo_weights=str(a.yolo_weights) if a.yolo_weights else None)
    robot = SO101Mujoco(cfg)

    ds = None
    if not a.dry:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
        root = HF_LEROBOT_HOME / repo_id
        if root.exists():
            sys.exit(f"⚠️ {root} exists already: pick another --repo-id or move it (nothing was written).")
        tap, _, rop = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(pipeline=tap, initial_features=create_initial_features(
                action=robot.action_features), use_videos=True),
            aggregate_pipeline_dataset_features(pipeline=rop, initial_features=create_initial_features(
                observation=robot.observation_features), use_videos=True))
        features[OBS_ENV_STATE] = {"dtype": "float32", "shape": [len(ttg.ENV_NAMES)], "names": ttg.ENV_NAMES}
        ds = LeRobotDataset.create(repo_id, expert.FPS, root=root, robot_type=robot.name, features=features,
                                   use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1"),
                                   image_writer_threads=4)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{repo_id.split('/')[-1]}{'_dry' if a.dry else ''}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    print(f"dataset {repo_id if ds is not None else '(dry, none)'}  task {TASK!r}\n"
          f"start frame: {'truss YOLO ' + str(a.yolo_weights) if a.yolo_weights else 'true box (no truss YOLO yet)'}\n"
          f"attempt log {log_path}", flush=True)
    robot.connect()
    kept = dict(n=0, miss=0)
    seed, attempts, t_start = a.seed_start - 1, 0, time.time()
    try:
        while kept["n"] < n_eps and (a.max_attempts is None or attempts < a.max_attempts):
            seed += 1
            attempts += 1
            rec = dict(seed=seed)
            t_try = time.time()

            def skip(why):
                rec.update(kept=False, reason=why, wall_s=round(time.time() - t_try, 1))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  seed {seed}: skip, {why}", flush=True)

            robot.config.viewer = False
            robot.new_scene(seed, log=False)
            if not np.all(np.isnan(robot.det.detached_at)):
                skip("a fruit fell off while the vines settled"); continue
            robot.get_observation()
            if robot._tgt_k is None:
                skip(f"no target ({robot._box_info['reason']})"); continue
            rng = np.random.default_rng([seed, 0x7255])
            left = n_eps - kept["n"]
            miss = bool(rng.random() < max(0, round(MISS_SHARE * n_eps) - kept["miss"]) / left)
            rec.update(target=robot._box_info["target_name"], target_source=robot._box_info["source"],
                       cut_source=robot._box_info["cut_source"], miss=miss, env=robot.env_state().round(4).tolist())
            fails, ok, plan = [], False, None
            for k in range(PLAN_TRIES):
                if k:
                    robot.new_scene(seed, log=False)
                    robot.get_observation()
                try:
                    plan = tx.TrussPlanner(robot.m, robot.d, robot.info, robot._tgt_k).plan(rng, miss=miss)
                except tx.PlanError as e:
                    fails.append(f"no plan: {e}")
                    break
                t0, watch = play(robot, plan)
                ok, why = judge(robot, plan, t0, watch)
                if ok:
                    break
                fails.append(f"dry run: {why}")
            rec["plan_tries"] = k + 1
            if not ok:
                skip(" | ".join(fails)); continue
            rec["plan"] = plan.meta
            if a.dry:
                kept["n"] += 1; kept["miss"] += miss
                rec.update(kept=True, reason="dry ok", wall_s=round(time.time() - t_try, 1))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  seed {seed}: dry ok ({kept['n']}/{n_eps}) {plan.meta['seconds']} s"
                      f"{', miss' if miss else ''}  [{time.time() - t_try:.0f} s wall]", flush=True)
                continue

            # --- recording pass: same seed, same plan ---
            robot.config.viewer = a.watch
            robot.new_scene(seed, log=True)
            t_ep = time.time()
            obs = robot.get_observation()
            if robot._box_info["target_name"] != rec["target"]:
                skip(f"recording picked another truss ({robot._box_info['target_name']})"); continue
            env = robot.env_state()
            cur = {"obs": obs}

            def frame(sent):
                t_f = time.time()
                o = cur["obs"]
                fr = {**build_dataset_frame(ds.features, rop(o), prefix=OBS_STR),
                      **build_dataset_frame(ds.features, tap((sent, o)), prefix=ACTION), "task": TASK}
                fr[OBS_ENV_STATE] = env
                ds.add_frame(fr)
                cur["obs"] = robot.get_observation()
                if a.watch:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
                return cur["obs"]

            t0, watch = play(robot, plan, on_frame=frame)
            ok, why = judge(robot, plan, t0, watch)
            if not ok:
                ds.clear_episode_buffer()
                skip(f"recording differs from the dry run: {why}"); continue
            ds.save_episode()
            ep = kept["n"]
            kept["n"] += 1; kept["miss"] += miss
            robot.extra = dict(episode_index=ep, dataset=repo_id, scripted=plan.meta, env_state=env.tolist())
            rec.update(kept=True, reason="ok", episode_index=ep, wall_s=round(time.time() - t_try, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            m = plan.meta
            print(f"✅ episode {ep} ({kept['n']}/{n_eps}): seed {seed}, {m['target']} ({m['n_fruit']} fruit, "
                  f"stalk {m['stalk_mm']} mm), {m['seconds']} s{', miss + recovery' if m['miss'] else ''}, "
                  f"neighbours moved {watch.max * 1000:.0f} mm  [{time.time() - t_ep:.0f} s wall]", flush=True)
    finally:
        robot.disconnect()
        if ds is not None:
            ds.finalize()
        n = max(kept["n"], 1)
        mins = round((time.time() - t_start) / 60, 1)
        log.write(json.dumps(dict(summary=True, kept=kept["n"], attempts=attempts, miss=kept["miss"], minutes=mins)) + "\n")
        log.close()
        print(f"\nkept {kept['n']} of {attempts} attempts in {mins} min ({mins / n:.1f} min per kept episode), "
              f"miss + recovery {kept['miss'] / n:.0%}\nlog {log_path}", flush=True)


if __name__ == "__main__":
    main()
