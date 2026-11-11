#!/usr/bin/env python3
"""Scripted whole-truss harvesting episodes: truss_expert.py instead of a person on the leader (K7, 2026-11-03).

    python generate_truss.py --pilot --watch            # 5 episodes into ..._pilot, MuJoCo window at real time
    python generate_truss.py --dry --episodes 20        # plan + dry run + judge only: keep rate and time, no dataset
    python generate_truss.py --dry --watch --seed-start 200007 --episodes 1   # watch one in the MuJoCo window
    python generate_truss.py                            # 200 episodes, rig "base", into krish5831/blush_sim_truss_v3
    python generate_truss.py --episodes 200 --repo-id krish5831/blush_sim_truss_v2 --rig base --start-jitter-deg 3 \
        --hold-share 0.10 --balance --seed-start 7600000   # how blush_sim_truss_v2 was recorded (rig "base")
    python generate_truss.py --teacher --one-layout --seed-start 9100009 --episodes 50 --no-dr --no-yolo \
        --repo-id krish5831/blush_sim_teacher_onelayout_v1    # post-mortem step 1: one layout, teacher + noise

Per scene seed: build the truss scene (domain randomisation on, settle 1 s), take
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
import truss_teacher as tt  # noqa: E402

# ⚠️ Not harvest/harvest.env's TASK: that string is the real rig's single-fruit placeholder. Truss harvesting
# is a different task; this string goes with it (K7 decides the real rig's wording when it records trusses).
TASK = "Cut the ready tomato truss in the green box at the cross and put it in the crate."
MISS_SHARE = 0.25                # K7 2026-11-03 (plan point 4: ~25 % recovery episodes)
PLAN_TRIES = 3                   # plans per seed before it's skipped (retry instead of discarding, plan point 2)
NEIGHBOUR_MOVE_M = 0.015         # any other truss's fruit may swing this far at most (plan point 2, guess)
# Rig "base" data mix (K7 2026-11-06, SockBot lessons): "nothing to pick" episodes, no truss ready, the arm holds its
# start pose for HOLD_S (his biggest lesson: the policy never saw what not reaching looks like). He capped them near
# 10 % of the set; the scripted episodes start with a 0.5-1.5 s hold of their own (design choice).
HOLD_SHARE = 0.10
HOLD_S = (4.0, 6.0)
# Targets spread along the row: blush_sim_truss_v1 had 98 of 200 targets on the left-most stem and 6 on the right-most
# (attempt log, 2026-11-06), as S.H.E.A.R's v0 had 180/34/0 cuts on its 3 plants. A seed whose target falls in a part
# of the row (BALANCE_BINS along truss_scene.STEM_X) already holding more than BALANCE_MAX_SHARE of the kept episodes
# is skipped before planning (cheap), once BALANCE_AFTER episodes are kept (design choice).
BALANCE_BINS, BALANCE_MAX_SHARE, BALANCE_AFTER = 4, 0.35, 8
START_TOL_NORM = 3.0             # max norm units off the start pose at the end (as the single-fruit generator, guess)
LOG_DIR = pathlib.Path.home() / ".cache" / "blush_sim" / "truss"
# The next dataset: v2 (rig "base") is the current one, v1 (birdseye) was deleted 2026-11-08 at K7's request.
DEFAULT_REPO = "krish5831/blush_sim_truss_v3"
YOLO_TRUSS = HERE.parent / "runs" / "yolo_truss" / "truss_v1" / "weights" / "best.pt"
# --teacher (post-mortem step 1+, 2026-11-11): the closed-loop teacher (truss_teacher.py) instead of the v2 planner,
# with DART-style noise (Laskey et al. 2017): the arm EXECUTES the teacher's command plus noise, the dataset records
# the teacher's clean command (the label) for the state the arm is really in, so the policy sees off-path states and
# what to do there. The noise: per arm joint an AR(1) process, stationary std NOISE_NORM (norm units, ~1.5-1.8 deg
# each), NOISE_RHO per frame (~0.3 s correlation at 30 fps); its seed per episode is logged, so the executed
# commands can be rebuilt (a replay of the recorded CLEAN actions won't reproduce a noisy episode).
NOISE_NORM = 1.0                 # guess, unverified: aims at ~1 cm at the jaw; the push test recovered from 1 cm
NOISE_RHO = 0.9                  # guess, unverified
# No noise at the stalk, at the crate or going home (the judge wants the arm within START_TOL_NORM of the start pose
# 0.3 s after the return); half on the slide (2 cm pushes there failed 4 of 18 in check_teacher.py push).
NOISE_OFF = {"settle_at_stalk", "close_cut", "squeeze", "lower", "release", "drop", "return", "hold_end"}
NOISE_SCALE = {"slide_in": 0.5}


class Noise:
    """The executed command = the teacher's label + an AR(1) noise per arm joint (gripper untouched)."""

    def __init__(self, sigma, rho, seed, episode):
        self.sigma, self.rho = sigma, rho
        self.rng = np.random.default_rng([seed, episode, 0xDA27])
        self.n = np.zeros(5)
        self.max = 0.0

    def __call__(self, leg, act):
        # the process steps every frame, also where it's switched off, so a leg's noise doesn't depend on the last
        self.n = self.rho * self.n + self.sigma * np.sqrt(1 - self.rho ** 2) * self.rng.normal(size=5)
        k = 0.0 if leg in NOISE_OFF else NOISE_SCALE.get(leg, 1.0)
        out = dict(act)
        for j, v in zip(expert.arm_map.JOINTS[:5], k * self.n):
            out[f"{j}.pos"] += float(v)
        self.max = max(self.max, float(np.abs(k * self.n).max()))
        return out


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


def judge_hold(robot, watch):
    """A stay-home episode: nothing cut, no fruit off, nothing moved past NEIGHBOUR_MOVE_M, no physics warnings."""
    d, info = robot.d, robot.info
    bad = sum(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                            mujoco.mjtWarning.mjWARN_BADQVEL))
    if bad:
        return False, f"{bad} physics warnings"
    cut = [info["trusses"][j]["name"] for j, c in enumerate(robot.cutters) if c.cut_t is not None]
    if cut:
        return False, f"cut {cut}"
    if not np.all(np.isnan(robot.det.detached_at)):
        return False, "a fruit came off"
    if watch.max > NEIGHBOUR_MOVE_M:
        return False, f"moved {watch.worst} {watch.max * 1000:.0f} mm"
    return True, "ok"


def hold_plan(robot, rng):
    """Stay where the scene started (the start pose, jittered or not) for HOLD_S."""
    cur = np.asarray(robot._present_norm(), float)
    n = int(round(rng.uniform(*HOLD_S) * expert.FPS))
    return expert.Plan(np.tile(cur, (n, 1)), np.zeros(n, bool), [("hold_home", n)],
                       dict(hold=True, frames=n, seconds=round(n / expert.FPS, 2)))


def play(robot, plan, on_frame=None, realtime=False, noise=None):
    """Plays a plan from the scene's start frame; returns (t0, Watch). realtime: paced at FPS (a window to watch).
    noise (--teacher): the arm executes noise(leg, label); on_frame gets the clean label, which the dataset records."""
    plan.reset()
    watch = Watch(robot)
    t0 = robot.d.time
    obs = None
    while not plan.done:
        t_f = time.time()
        state = robot._present_norm() if obs is None else [obs[f"{j}.pos"] for j in expert.arm_map.JOINTS]
        leg = getattr(plan, "leg", None)
        act = plan.act(state)
        sent = robot.send_action(act if noise is None else noise(leg, act))
        if noise is not None:
            sent = act                               # the label, not what was executed
        if on_frame is not None:
            obs = on_frame(sent)
        watch.step()
        if realtime:
            time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
    return t0, watch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--pilot", action="store_true", help="5 episodes into <repo-id>_pilot")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--seed-start", type=int, default=200000)
    ap.add_argument("--watch", action="store_true",
                    help="MuJoCo window at real time: on the recording pass, or with --dry on the dry run itself")
    ap.add_argument("--dry", action="store_true", help="dry runs only: plan + judge, write no dataset")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--yolo-weights", type=pathlib.Path, default=YOLO_TRUSS if YOLO_TRUSS.exists() else None,
                    help="truss YOLO for the start frame + overlay (default: runs/yolo_truss/truss_v1 if trained, "
                         "else the true box, no YOLO)")
    ap.add_argument("--no-yolo", action="store_true", help="true box at the start frame even if the truss YOLO exists")
    ap.add_argument("--rig", default="base", choices=("birdseye", "base"),
                    help="cameras recorded (SO101MujocoConfig.rig): birdseye + claw, or base + claw (2026-11-06)")
    ap.add_argument("--start-jitter-deg", type=float, default=0.0, help="start pose jitter per arm joint (SockBot: 3)")
    ap.add_argument("--hold-share", type=float, default=0.0,
                    help=f"share of stay-home episodes, nothing ready (rig base data mix: {HOLD_SHARE})")
    ap.add_argument("--balance", action="store_true", help="spread targets along the row (BALANCE_*)")
    ap.add_argument("--teacher", action="store_true",
                    help="teacher v3 (truss_teacher.py) + DART noise instead of the v2 planner; no misses")
    ap.add_argument("--noise", type=float, default=NOISE_NORM, help="--teacher: noise std per arm joint, norm units")
    ap.add_argument("--one-layout", action="store_true",
                    help="every episode the scene of --seed-start (the step-1 pipeline check); the noise differs")
    ap.add_argument("--no-dr", action="store_true", help="no domain randomisation (post-mortem: off until 50 %%)")
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
                            fps=expert.FPS, domain_rand=not a.no_dr, settle_s=1.0, snap_to_first_action=False,
                            rig=a.rig, start_jitter_deg=a.start_jitter_deg,
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
    kept = dict(n=0, miss=0, hold=0)
    n_hold = max(1, round(a.hold_share * n_eps)) if a.hold_share > 0 else 0
    x0, x1 = ts.STEM_X
    bins = np.zeros(BALANCE_BINS, int)
    bin_of = lambda x: int(np.clip((x - x0) / (x1 - x0) * BALANCE_BINS, 0, BALANCE_BINS - 1))   # noqa: E731
    seed, attempts, t_start = a.seed_start - 1, 0, time.time()
    try:
        while kept["n"] < n_eps and (a.max_attempts is None or attempts < a.max_attempts):
            seed = a.seed_start if a.one_layout else seed + 1
            attempts += 1
            rec = dict(seed=seed)
            noise = None
            if a.teacher:
                rec["noise"] = dict(std_norm=a.noise, rho=NOISE_RHO, seed=[seed, attempts, 0xDA27])
            t_try = time.time()

            def skip(why, rec=rec, t_try=t_try, seed=seed):   # bound now (B023); only called in this iteration anyway
                rec.update(kept=False, reason=why, wall_s=round(time.time() - t_try, 1))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  seed {seed}: skip, {why}", flush=True)

            robot.config.viewer = a.watch and a.dry
            rng = np.random.default_rng([seed, 0x7255])
            left = n_eps - kept["n"]
            hold = bool(np.random.default_rng([seed, 0x401D]).random() < max(0, n_hold - kept["hold"]) / left)
            robot.new_scene(seed, log=False, hold=hold)
            if not np.all(np.isnan(robot.det.detached_at)):
                skip("a fruit fell off while the vines settled"); continue
            robot.get_observation()
            if hold:
                rec.update(hold=True, target=None, env=robot.env_state().round(4).tolist())
                plan = hold_plan(robot, rng)
                t0, watch = play(robot, plan, realtime=a.watch and a.dry)
                ok, why = judge_hold(robot, watch)
                if not ok:
                    skip(f"hold dry run: {why}"); continue
                if a.dry:
                    kept["n"] += 1; kept["hold"] += 1
                    rec.update(kept=True, reason="dry ok (hold)", wall_s=round(time.time() - t_try, 1))
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"  seed {seed}: dry ok ({kept['n']}/{n_eps}) hold {plan.meta['seconds']} s", flush=True)
                    continue
            elif robot._tgt_k is None:
                skip(f"no target ({robot._box_info['reason']})"); continue
            miss = False
            if not hold:
                if a.balance and kept["n"] >= BALANCE_AFTER:
                    b = bin_of(ts.cut_target_now(robot.info["trusses"][robot._tgt_k], robot.d, robot.m)[0][0])
                    if (bins[b] + 1) / (kept["n"] - kept["hold"] + 1) > BALANCE_MAX_SHARE:
                        skip(f"balance: row part {b + 1}/{BALANCE_BINS} has {bins[b]} of "
                             f"{kept['n'] - kept['hold']}"); continue
                miss = not a.teacher and bool(rng.random() < max(0, round(MISS_SHARE * n_eps) - kept["miss"]) / left)
                rec["cut_x"] = round(float(ts.cut_target_now(robot.info["trusses"][robot._tgt_k], robot.d,
                                                             robot.m)[0][0]), 4)
                rec.update(target=robot._box_info["target_name"], target_source=robot._box_info["source"],
                           cut_source=robot._box_info["cut_source"], miss=miss, env=robot.env_state().round(4).tolist())
                fails, ok, plan = [], False, None
                for k in range(1 if a.teacher else PLAN_TRIES):    # the teacher is deterministic: one try
                    if k:
                        robot.new_scene(seed, log=False)
                        robot.get_observation()
                    try:
                        if a.teacher:
                            plan = tt.TrussTeacher(robot.m, robot.d, robot.info, robot._tgt_k).plan()
                            # ⚠️ The cross + environment_state mark u = 0.5 (truss_target.cut_pixel); marking the
                            # teacher's own u is generator v3 (step 2) work. Until then only scenes cut at 0.5.
                            if plan.meta["cut_u"] != 0.5:
                                fails.append(f"teacher cuts at u {plan.meta['cut_u']}, the cue marks 0.5")
                                break
                            noise = Noise(a.noise, NOISE_RHO, seed, attempts)
                        else:
                            plan = tx.TrussPlanner(robot.m, robot.d, robot.info, robot._tgt_k).plan(rng, miss=miss)
                    except tx.PlanError as e:
                        fails.append(f"no plan: {e}")
                        break
                    t0, watch = play(robot, plan, realtime=a.watch and a.dry, noise=noise)
                    ok, why = judge(robot, plan, t0, watch)
                    if ok:
                        break
                    fails.append(f"dry run: {why}")
                rec["plan_tries"] = k + 1
                if noise is not None:
                    rec["noise"]["max_norm"] = round(noise.max, 2)
                if not ok:
                    skip(" | ".join(fails)); continue
                rec["plan"] = plan.meta
                if a.dry:
                    kept["n"] += 1; kept["miss"] += miss
                    bins[bin_of(rec["cut_x"])] += 1
                    rec.update(kept=True, reason="dry ok", wall_s=round(time.time() - t_try, 1))
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"  seed {seed}: dry ok ({kept['n']}/{n_eps}) {plan.meta['seconds']} s"
                          f"{', miss' if miss else ''}  [{time.time() - t_try:.0f} s wall]", flush=True)
                    continue

            # --- recording pass: same seed, same plan ---
            robot.config.viewer = a.watch
            robot.new_scene(seed, log=True, hold=hold)
            t_ep = time.time()
            obs = robot.get_observation()
            if not hold and robot._box_info["target_name"] != rec["target"]:
                skip(f"recording picked another truss ({robot._box_info['target_name']})"); continue
            env = robot.env_state()
            cur = {"obs": obs}
            # ⚠️ the box + cut point aren't robot observations: building the frame from all of ds.features looked
            # them up in the robot's dict (KeyError 'box_cx', pilot 2026-11-03); they're added below instead
            obs_feats = {k: v for k, v in ds.features.items() if k != OBS_ENV_STATE}

            def frame(sent, cur=cur, obs_feats=obs_feats, env=env):   # bound now (B023); play() calls it right away
                t_f = time.time()
                o = cur["obs"]
                fr = {**build_dataset_frame(obs_feats, rop(o), prefix=OBS_STR),
                      **build_dataset_frame(ds.features, tap((sent, o)), prefix=ACTION), "task": TASK}
                fr[OBS_ENV_STATE] = env
                ds.add_frame(fr)
                cur["obs"] = robot.get_observation()
                if a.watch:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
                return cur["obs"]

            t0, watch = play(robot, plan, on_frame=frame,
                             noise=None if noise is None else Noise(a.noise, NOISE_RHO, seed, attempts))
            ok, why = judge_hold(robot, watch) if hold else judge(robot, plan, t0, watch)
            if not ok:
                ds.clear_episode_buffer()
                skip(f"recording differs from the dry run: {why}"); continue
            ds.save_episode()
            ep = kept["n"]
            kept["n"] += 1; kept["miss"] += miss; kept["hold"] += hold
            if not hold:
                bins[bin_of(rec["cut_x"])] += 1
            robot.extra = dict(episode_index=ep, dataset=repo_id, scripted=plan.meta, env_state=env.tolist())
            rec.update(kept=True, reason="ok", episode_index=ep, wall_s=round(time.time() - t_try, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            m = plan.meta
            if hold:
                print(f"✅ episode {ep} ({kept['n']}/{n_eps}): seed {seed}, HOLD (nothing ready) {m['seconds']} s, "
                      f"neighbours moved {watch.max * 1000:.0f} mm  [{time.time() - t_ep:.0f} s wall]", flush=True)
                continue
            print(f"✅ episode {ep} ({kept['n']}/{n_eps}): seed {seed}, {m['target']} ({m['n_fruit']} fruit, "
                  f"stalk {m['stalk_mm']} mm), {m['seconds']} s{', miss + recovery' if m['miss'] else ''}, "
                  f"neighbours moved {watch.max * 1000:.0f} mm  [{time.time() - t_ep:.0f} s wall]", flush=True)
    finally:
        robot.disconnect()
        if ds is not None:
            ds.finalize()
        n = max(kept["n"], 1)
        mins = round((time.time() - t_start) / 60, 1)
        log.write(json.dumps(dict(summary=True, kept=kept["n"], attempts=attempts, miss=kept["miss"], hold=kept["hold"],
                                  row_parts=bins.tolist(), minutes=mins)) + "\n")
        log.close()
        print(f"\nkept {kept['n']} of {attempts} attempts in {mins} min ({mins / n:.1f} min per kept episode), "
              f"miss + recovery {kept['miss'] / n:.0%}, hold {kept['hold']}, targets per row part {bins.tolist()}\n"
              f"log {log_path}", flush=True)


if __name__ == "__main__":
    main()
