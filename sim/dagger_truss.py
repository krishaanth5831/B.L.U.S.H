#!/usr/bin/env python3
"""DAgger for the truss policy: the policy drives, the scripted expert takes over when it veers off (K7, 2026-11-05).

    python dagger_truss.py --dry --episodes 10           # pilot: keep rate, triggers, time; nothing written
    python dagger_truss.py --dry --watch --episodes 1    # watch one takeover in the MuJoCo window
    python dagger_truss.py                               # 100 episodes appended to krish5831/blush_sim_truss_v1

⚠️ Birdseye rig only, as written for blush_sim_truss_v1 + smolvla_truss_expert_v1 (2026-11-05): the sim is built
without `rig=`, so a rig "base" policy / dataset (blush_sim_truss_v2, act_truss_v1) needs changes first.

Per scene seed (fresh range DEFAULT_SEED, never trained or evaluated on):
  1. policy pass (nothing recorded): the scene as in eval_truss.py; the policy drives, every action it sends is
     kept, and every frame is checked by Trigger (the signs of veering off, well before the judge's limits).
     The policy finishing the job alone, with no trigger, is logged and skipped: nothing to correct.
  2. takeover test: the takeover frame is REWIND_S (random) before the trigger, so the expert starts before the
     damage. Rebuild the seed, replay the policy's actions up to there (physics only, checked against the qpos the
     policy pass saw), plan from that state (truss_expert takeover "before_cut" / "after_cut"), play it, judge it
     with generate_truss.judge (the same rules as the 200 scripted episodes, over the WHOLE episode from frame 0).
     PLAN_TRIES plans, each from a fresh rebuild + replay.
     ⚠️ Rebuild + replay instead of mj_getState/mj_setState: the cutters and the detacher keep their state in
     Python. And the stored actions, not the policy again: SmolVLA samples noise, so a second run differs (ACT
     doesn't, but replaying is cheaper than inference either way).
  3. recording pass: rebuild, replay the policy's actions and then the expert's plan, recording every frame like
     generate_truss.py; judged again, a mismatch is discarded.
The episode holds the policy's own actions up to the takeover (so it starts at the scene's first frame like the
other 200: the target pick + environment_state come from frame 0), then the expert's correction. The takeover
frame is logged per episode, so the policy prefix can be dropped or down-weighted later.

Writes: appends to the dataset (~/.cache/huggingface/lerobot/<repo-id>, LeRobotDataset.resume), the sim state log
(~/.cache/blush_sim/states/), the attempt log (~/.cache/blush_sim/truss/dagger_<stamp>.jsonl). Never pushes.
⚠️ Back up the dataset before a run (it appends in place; an interrupted run is finalized in `finally`).
Run in the lerobot env, outside the Claude sandbox (GPU + display), under the memory cap, with
BLUSH_RIPENESS_PYTHON set as for generate_truss.py.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_truss as ev  # noqa: E402
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402
import truss_expert as tx  # noqa: E402
import truss_ik  # noqa: E402
import truss_scene as ts  # noqa: E402

DEFAULT_POLICY = HERE.parent / "outputs/train/smolvla_truss_expert_v1/checkpoints/060000/pretrained_model"
# v1 training 210000-210512, pilot 200000-200008, eval 5000000-5000019 + ~1.6e9; ⚠️ blush_sim_truss_v2 was
# recorded from 7600000 (7600000-7600533), so a run past ~600k attempts would reach it
DEFAULT_SEED = 7_000_000
EPISODE_S = ev.EPISODE_S          # the policy's time per scene (45 s)
PLAN_TRIES = 3                    # takeover plans per seed (generate_truss.PLAN_TRIES)
REWIND_S = (0.5, 1.0)             # takeover this long before the trigger (design choice: before the damage)
# Trigger: signs the policy is going wrong, kept well inside the judge's limits so a rewound takeover can still
# pass it (all guesses, unverified; tune from the pilot's trigger log).
# A neighbour's fruit pushed this far (judge: 15 mm, generate_truss). ⚠️ At 6 mm 5 of 6 pilot triggers fired in the
# policy's first 0.8-2 s (neighbours sway < 1 mm with the arm still), so the expert took over at the start pose
# and the policy never reached the cut or the carry, where it failed in the eval (2026-11-05).
NUDGE_M = 0.010
TUG_SHARE = 0.5                   # a tether's pull past this share of the way from its rest to its break force
CLOSED_CMD_NORM = 8.0             # gripper command below this = shut (0 deg = 7.9 norm; the expert closes to 0-2)
CLOSED_STALL_S = 1.5              # shut this long without a cut: closing on nothing / on the stalk without cutting
                                  # (the expert's miss close + pause is 0.8-1.3 s)
CUT_BY_S = 15.0                   # no cut by then (scripted episodes cut ~6-12 s in, 19-31 s long; guess)
CRATE_AFTER_CUT_S = 15.0          # cut but not in the crate this long after (guess)
HELD_M = 0.015                    # the cut stalk's cut zone within this of the jaw's cut point = still held (guess)
AFTER_CRATE_S = ev.AFTER_CRATE_S  # policy alone got it in: run on this long, then call it a policy success
# A takeover earlier than this is a plain scripted episode (the policy's part rewound away): skipped. Pilot seed
# 7000105 took over at 0.0 s (2026-11-05). Design choice.
MIN_POLICY_S = 1.0


def jaw_point(robot):
    """The jaw's cut point (world) and frame now."""
    k = robot._tgt_k
    r = robot.info["trusses"][k]["r"]
    return truss_ik.truss_ik().frame(robot.d.qpos[robot.qidx][:5], r)


def held(robot):
    t = robot.info["trusses"][robot._tgt_k]
    p, _ = jaw_point(robot)
    return min(np.linalg.norm(ts.cut_target_now(t, robot.d, robot.m, u)[0] - p) for u in np.linspace(0, 1, 6)) < HELD_M


class Trigger:
    """Per frame of the policy pass: why to take over now, or None."""

    def __init__(self, robot):
        self.r = robot
        self.watch = gt.Watch(robot)
        self.f0 = robot.det.force.copy()
        # ⚠️ det.peak holds the settle's swing too: restart it here (the physics never reads it, the judge neither)
        robot.det.peak[:] = self.f0
        self.t0 = robot.d.time
        self.shut_since = None

    def check(self, sent):
        r, d = self.r, self.r.d
        t = d.time - self.t0
        self.watch.step()
        if self.watch.max > NUDGE_M:
            return f"nudged {self.watch.worst} {self.watch.max * 1000:.0f} mm"
        det = r.det
        if not np.all(np.isnan(det.detached_at)):
            return "a fruit came off"
        tug = (det.peak - self.f0) / np.maximum(det.thresholds - self.f0, 1e-6)
        if tug.max() > TUG_SHARE:
            return f"tugged {r.info['fruit'][int(np.argmax(tug))]['tomato']} ({tug.max():.0%} of its break)"
        k = r._tgt_k
        for j, c in enumerate(r.cutters):
            if j != k and (c.peak["squeeze"] > 0 or c.cut_t is not None):
                return f"blade on the wrong truss {r.info['trusses'][j]['name']}"
        cut = r.cutters[k].cut_t
        if cut is None:
            if sent["gripper.pos"] < CLOSED_CMD_NORM:
                self.shut_since = t if self.shut_since is None else self.shut_since
                if t - self.shut_since > CLOSED_STALL_S:
                    return f"jaw shut {CLOSED_STALL_S} s without a cut"
            else:
                self.shut_since = None
            if t > CUT_BY_S:
                return f"no cut by {CUT_BY_S:.0f} s"
            return None
        if not held(r):
            return "the cut truss left the jaw"
        if not ts.in_crate(r.info, d, r.m, k) and t - (cut - self.t0) > CRATE_AFTER_CUT_S:
            return f"not in the crate {CRATE_AFTER_CUT_S:.0f} s after the cut"
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--repo-id", default=gt.DEFAULT_REPO)
    ap.add_argument("--seed-start", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="policy pass + takeover test only: write no dataset")
    ap.add_argument("--watch", action="store_true",
                    help="MuJoCo window at real time on the recording pass (with --dry: on the takeover test)")
    ap.add_argument("--no-yolo", action="store_true", help="true box at the start frame, no YOLO overlay")
    a = ap.parse_args()

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    root = HF_LEROBOT_HOME / a.repo_id
    meta = LeRobotDatasetMetadata(a.repo_id, root=root)
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    tap, _, rop = make_default_processors()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_path = a.policy.resolve()
    policy, pre, post, pcfg = ev.load_policy(policy_path, device, meta)
    wants_env = OBS_ENV_STATE in getattr(pcfg, "input_features", {})

    yolo = None if a.no_yolo or not gt.YOLO_TRUSS.exists() else str(gt.YOLO_TRUSS)
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=a.seed_start, viewer=False, overlay="yolo" if yolo else "true_box", realtime=False,
        fps=expert.FPS, domain_rand=True, settle_s=1.0, snap_to_first_action=False, task="truss",
        yolo_weights=yolo))

    ds = None
    if not a.dry:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset
        ds = LeRobotDataset.resume(a.repo_id, root=root, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1"),
                                   image_writer_threads=4)
        if OBS_ENV_STATE not in ds.features:
            sys.exit(f"⚠️ {a.repo_id} has no {OBS_ENV_STATE}: not a truss dataset")
    n_before = meta.total_episodes

    gt.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = gt.LOG_DIR / f"dagger{'_dry' if a.dry else ''}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    print(f"policy {policy_path}\ndataset {a.repo_id if ds is not None else '(dry, none)'} ({n_before} episodes "
          f"before)\nattempt log {log_path}", flush=True)
    robot.connect()

    def start(seed, viewer=False):
        """Fresh build of seed + its first frame (the target pick). Returns the first observation."""
        robot.config.viewer = viewer
        robot.new_scene(seed, log=False)
        return robot.get_observation()

    def replay(actions, watch, on_frame=None, realtime=False):
        obs = None
        for act in actions:
            t_f = time.time()
            sent = robot.send_action(act)
            if on_frame is not None:
                obs = on_frame(sent)
            watch.step()
            if realtime:
                time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
        return obs

    def play(plan, watch, obs=None, on_frame=None, realtime=False):
        """generate_truss.play, continuing an episode: the Watch runs on from frame 0."""
        plan.reset()
        t0 = robot.d.time
        while not plan.done:
            t_f = time.time()
            state = robot._present_norm() if obs is None else [obs[f"{j}.pos"] for j in expert.arm_map.JOINTS]
            sent = robot.send_action(plan.act(state))
            if on_frame is not None:
                obs = on_frame(sent)
            watch.step()
            if realtime:
                time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
        return t0

    kept, seed, attempts, t_start = 0, a.seed_start - 1, 0, time.time()
    stats = {}
    try:
        while kept < a.episodes and (a.max_attempts is None or attempts < a.max_attempts):
            seed += 1
            attempts += 1
            rec = dict(seed=seed)
            t_try = time.time()

            def done(ok, why, rec=rec, t_try=t_try, seed=seed, **kw):   # bound now (B023); called this iteration
                rec.update(kept=ok, reason=why, wall_s=round(time.time() - t_try, 1), **kw)
                log.write(json.dumps(rec) + "\n"); log.flush()
                key = why.split(":")[0] if not ok else "kept"
                stats[key] = stats.get(key, 0) + 1
                if not ok:
                    print(f"  seed {seed}: skip, {why}  [{time.time() - t_try:.0f} s wall]", flush=True)

            # --- 1. policy pass ---
            obs = start(seed)
            if not np.all(np.isnan(robot.det.detached_at)):
                done(False, "a fruit fell off while the vines settled"); continue
            if robot._tgt_k is None:
                done(False, f"no target ({robot._box_info['reason']})"); continue
            k, target = robot._tgt_k, robot._box_info["target_name"]
            rng = np.random.default_rng([seed, 0xDA66])
            # Only scenes the expert can do from the start, like the 200 scripted ones. ⚠️ Without this the pilot's
            # takeovers kept landing in scenes where even the start pose (jaw open) wasn't clear of a truss: the
            # policy "veered" in its first 0.4 s and no expert plan existed (seeds 7000003, 7000100, 2026-11-05).
            try:
                tx.TrussPlanner(robot.m, robot.d, robot.info, k).plan(rng)
            except tx.PlanError as e:
                done(False, f"no expert plan from the start: {e}"[:200]); continue
            env = robot.env_state()
            rec.update(target=target, target_source=robot._box_info["source"])
            policy.reset()
            trig = Trigger(robot)
            actions, qpos, cut_at, why, t_crate = [], [robot.d.qpos.copy()], None, None, None
            while robot.d.time - trig.t0 < EPISODE_S:
                frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
                if wants_env:
                    frame[OBS_ENV_STATE] = env
                with torch.inference_mode():
                    act = post(policy.select_action(pre(prepare_observation_for_inference(
                        frame, device, gt.TASK, robot.name))))
                sent = robot.send_action(make_robot_action(act, meta.features))
                actions.append(sent)
                qpos.append(robot.d.qpos.copy())
                obs = robot.get_observation()
                t = robot.d.time - trig.t0
                if cut_at is None and robot.cutters[k].cut_t is not None:
                    cut_at = len(actions)
                why = trig.check(sent)
                if why:
                    break
                if t_crate is None and ts.in_crate(robot.info, robot.d, robot.m, k):
                    t_crate = t
                if t_crate is not None and t - t_crate > AFTER_CRATE_S:
                    break
            if not why:
                done(False, "policy did it alone" if t_crate is not None else "no trigger, no success (45 s)",
                     policy_s=round(robot.d.time - trig.t0, 1))
                continue
            i_trig = len(actions)
            i_take = max(0, i_trig - int(round(rng.uniform(*REWIND_S) * expert.FPS)))
            phase = "after_cut" if cut_at is not None and cut_at <= i_take else "before_cut"
            rec.update(trigger=why, t_trigger=round(i_trig / expert.FPS, 2), t_takeover=round(i_take / expert.FPS, 2),
                       takeover=phase)
            if i_take < MIN_POLICY_S * expert.FPS:
                done(False, f"veered at the very start ({why} at {i_trig / expert.FPS:.1f} s): would be a plain "
                            f"scripted episode"); continue
            print(f"▶ seed {seed}: {target}, policy veered at {i_trig / expert.FPS:.1f} s ({why}); expert takes over "
                  f"at {i_take / expert.FPS:.1f} s, {phase.replace('_', ' ')}", flush=True)

            # --- 2. takeover test: rebuild, replay to the takeover, plan, dry run, judge ---
            fails, ok, plan, fresh = [], False, None, False
            for n_try in range(PLAN_TRIES):  # noqa: B007  n_try is read after the loop (plan_tries)
                if not fresh:                            # planning doesn't touch the sim: rebuild only after a play
                    start(seed, viewer=a.watch and a.dry)
                    if robot._box_info["target_name"] != target:
                        fails.append(f"rebuild picked another truss ({robot._box_info['target_name']})"); break
                    watch = gt.Watch(robot)
                    replay(actions[:i_take], watch, realtime=a.watch and a.dry)
                    if not np.allclose(robot.d.qpos, qpos[i_take], atol=1e-6):
                        fails.append(f"replay differs from the policy pass "
                                     f"({np.abs(robot.d.qpos - qpos[i_take]).max():.2g})"); break
                    if phase == "after_cut" and not held(robot):
                        fails.append("cut truss not in the jaw at the takeover"); break
                    fresh = True
                try:
                    plan = tx.TrussPlanner(robot.m, robot.d, robot.info, k).plan(rng, takeover=phase)
                except tx.PlanError as e:
                    fails.append(f"no plan: {e}"[:200])
                    if "no escape" in str(e):            # the same pose again: no other draw changes that
                        break
                    continue
                fresh = False
                t0 = play(plan, watch, realtime=a.watch and a.dry)
                ok, jwhy = gt.judge(robot, plan, t0, watch)
                if ok:
                    break
                fails.append(f"dry run: {jwhy}")
            rec["plan_tries"] = n_try + 1
            if not ok:
                done(False, "takeover failed: " + " | ".join(fails)); continue
            rec["plan"] = plan.meta
            if a.dry:
                kept += 1
                done(True, "dry ok", seconds=round(i_take / expert.FPS + plan.meta["seconds"], 1))
                print(f"  ✅ dry ok ({kept}/{a.episodes}): policy {i_take / expert.FPS:.1f} s + expert "
                      f"{plan.meta['seconds']} s, neighbours moved {watch.max * 1000:.0f} mm  "
                      f"[{time.time() - t_try:.0f} s wall]", flush=True)
                continue

            # --- 3. recording pass: same seed, the policy's actions, then the same plan ---
            robot.config.viewer = a.watch
            robot.new_scene(seed, log=True)              # this one goes into the state log
            obs = robot.get_observation()
            if robot._box_info["target_name"] != target:
                done(False, f"recording picked another truss ({robot._box_info['target_name']})"); continue
            env = robot.env_state()
            cur = {"obs": obs}

            def frame(sent, cur=cur, env=env):   # bound now (B023); replay() / play() call it right away
                t_f = time.time()
                o = cur["obs"]
                fr = {**build_dataset_frame(obs_feats, rop(o), prefix=OBS_STR),
                      **build_dataset_frame(ds.features, tap((sent, o)), prefix=ACTION), "task": gt.TASK}
                fr[OBS_ENV_STATE] = env
                ds.add_frame(fr)
                cur["obs"] = robot.get_observation()
                if a.watch:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
                return cur["obs"]

            watch = gt.Watch(robot)
            replay(actions[:i_take], watch, on_frame=frame)
            t0 = play(plan, watch, on_frame=frame)       # first state from the sim, as in the dry run
            ok, jwhy = gt.judge(robot, plan, t0, watch)
            if not ok:
                ds.clear_episode_buffer()
                done(False, f"recording differs from the dry run: {jwhy}"); continue
            ds.save_episode()
            ep = n_before + kept
            kept += 1
            robot.extra = dict(episode_index=ep, dataset=a.repo_id, dagger=dict(
                policy=str(policy_path), trigger=why, takeover_frame=i_take, takeover=phase), scripted=plan.meta,
                env_state=env.tolist())
            done(True, "ok", episode_index=ep, takeover_frame=i_take)
            print(f"✅ episode {ep} ({kept}/{a.episodes}): seed {seed}, policy {i_take / expert.FPS:.1f} s + expert "
                  f"{plan.meta['seconds']} s, neighbours moved {watch.max * 1000:.0f} mm  "
                  f"[{time.time() - t_try:.0f} s wall]", flush=True)
    finally:
        robot.disconnect()
        if ds is not None:
            ds.finalize()
        mins = round((time.time() - t_start) / 60, 1)
        log.write(json.dumps(dict(summary=True, kept=kept, attempts=attempts, minutes=mins, outcomes=stats)) + "\n")
        log.close()
        print(f"\nkept {kept} of {attempts} attempts in {mins} min ({mins / max(kept, 1):.1f} min per kept episode)\n"
              f"outcomes {stats}\nlog {log_path}", flush=True)


if __name__ == "__main__":
    main()
