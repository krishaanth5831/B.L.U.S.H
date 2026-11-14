#!/usr/bin/env python3
"""Run a trained truss-harvesting policy on the sim arm in scenes it never trained on, and watch it (K7, 2026-11-04).

    python eval_truss.py ../outputs/train/act_truss_v1/checkpoints/last/pretrained_model
    python eval_truss.py <policy> --scenes 5 --seed-start 1234567      # replay: the seeds are printed
    python eval_truss.py <policy> --headless --no-dr --no-yolo --same-seed --seed-start 9100009 --scenes 20 --noise 1.0
                                                                       # post-mortem step 1: the one-layout check
    python eval_truss.py <policy> --headless --no-dr --no-yolo --n-action-steps 100 --holds 5 \
        --seeds-from heldout_v3.jsonl                                  # step 3: the 20 held-out scenes

Windows: the MuJoCo viewer (third person, mouse to orbit) and one per camera showing exactly what the policy gets
(truss-YOLO overlay, green target box + cut-point cross, as in the recorded episodes). Keys in a camera window:
N = give up on this scene, Q = quit. Real time (30 fps): the sim steps 1/30 s per action, the loop sleeps the rest.

Scenes: truss_scene.build at a fresh random seed per run (>= 1,000,000; blush_sim_truss_v1 used 210000-210512, the
pilot 200000-200008, blush_sim_truss_v2 7600000-7600533: a random start can land there, ~1 in 4 million per
10-scene run), same settings as generate_truss.py's recording pass (domain randomisation, 1 s settle,
truss YOLO picks the ready truss on the first frame, true box when it's wrong). The policy gets
observation.environment_state (box + cut point) if it was trained with it; the box + cross are drawn on the image
either way. The rig (cameras) and the dataset whose features are used come from the checkpoint (trained_on).

--seeds-from (step 3, 2026-11-14): the scenes are the kept seeds of a generate_truss.py attempt log instead of
seed-start + 1, + 2, ... Teacher v3 solves about a third of the scenes with a ready truss (step 0: 73 of 200) and the
dataset holds only those, so the step-3 test set is 20 seeds from 9900000 (outside every recorded range) that a clean
teacher dry run solved:
    python generate_truss.py --teacher --no-dr --no-yolo --dry --noise-share 0 --seed-start 9900000 --episodes 20 \
        --max-attempts 400 --repo-id krish5831/blush_eval_heldout_v3
(2026-11-14: 20 of 53 scenes solved, seeds 9900006 to 9900052; its kept rows are sim/heldout_v3.jsonl. Unbalanced,
as scenes come: 13 of the 20 targets hang in the first quarter of the row.) A success rate on them is "of the
scenes the teacher solves", as S.H.E.A.R's "of teacher-feasible cuts".
--holds K adds K nothing-ready scenes (truss_scene.build hold, as the dataset's stay-home episodes) from
--hold-seed-start: the policy runs HOLD_EVAL_S; success = generate_truss.judge_hold (nothing cut, no fruit off,
nothing moved) and the arm stayed within HOLD_MOVE_NORM of where it started. Reported apart, not in the success count.

Graded like generate_truss.judge, at the end of each scene: success = the target truss was cut and lies whole in
the crate, no other truss was cut, no fruit came off anywhere. Also logged: cut at all, which fruit came off, how
far the neighbours moved, time to the crate. Results: ../outputs/eval/<run>_<checkpoint>_<stamp>.jsonl.
Nothing is recorded or pushed. Run in the lerobot env, outside the Claude sandbox (GPU + display), with
BLUSH_RIPENESS_PYTHON set as for generate_truss.py (the YOLO worker).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")      # cv2's Qt windows under Wayland, via XWayland

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402
import truss_ik  # noqa: E402
import truss_scene as ts  # noqa: E402

TRAIN_DATASET = "krish5831/blush_sim_truss_v2"   # fallback for a checkpoint without train_config.json (act-truss's)
EPISODE_S = 45.0          # scripted truss episodes run 19-31 s (run_v1.log); 45 s gives the policy slack
AFTER_CRATE_S = 3.0       # keep going this long after the truss lands, so a late knock-off still counts
NOISE_NEAR_M = 0.03       # --noise is off within this of the cut point (generate_truss.NOISE_OFF at the stalk)
HOLD_EVAL_S = 6.0         # --holds: the dataset's stay-home episodes last 4-6 s (generate_truss.HOLD_S)
HOLD_MOVE_NORM = 3.0      # --holds: the arm "stayed home" within this on every joint, norm units (guess, unverified)
HOLD_SEED_START = 9_950_000   # --holds: outside every recorded range (step 2 ended at 9722070)


def cameras(path):
    """The camera features a checkpoint takes, from its config.json input_features (e.g. observation.images.base)."""
    cfg = path / "config.json"
    feats = (json.loads(cfg.read_text()).get("input_features") or {}) if cfg.exists() else {}
    return {k for k in feats if k.startswith("observation.images.")}


def trained_on(path):
    """(dataset repo id, rig) a checkpoint was trained on: the rig is "base" when the base camera is among its inputs
    (rig "base", 2026-11-06), else "birdseye"; the dataset from its train_config.json."""
    rig = "base" if "observation.images.base" in cameras(path) else "birdseye"
    cfg = path / "train_config.json"
    return (json.loads(cfg.read_text())["dataset"]["repo_id"] if cfg.exists() else TRAIN_DATASET), rig


def load_policy(path, device, ds_meta, overrides=None):
    """A checkpoint via lerobot's make_policy, as trained (the policy type is config.json's "type": ACT now).
    overrides: config fields to change for this run (the execution sweep: temporal_ensemble_coeff, n_action_steps)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import make_policy
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = str(path)
    cfg.device = str(device)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    policy = make_policy(cfg, ds_meta=ds_meta).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, pre, post, cfg


def show(obs, label):
    for c in [c for c in ("birdseye", "base", "claw") if c in obs]:   # whichever rig the policy uses
        img = cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(f"{c} camera (what the policy sees)", img)
    return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=pathlib.Path, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--scenes", type=int, default=10, help="scenes to run")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="first scene seed; default: a random one per run (replay a scene with its printed seed)")
    ap.add_argument("--fast", action="store_true", help="don't sleep to real time")
    ap.add_argument("--no-yolo", action="store_true", help="true box at the start frame, no YOLO overlay")
    ap.add_argument("--headless", action="store_true", help="no windows (implies --fast): just the graded results")
    # post-mortem step 1 (2026-11-11): the one-layout pipeline check and the execution sweep
    ap.add_argument("--no-dr", action="store_true", help="no domain randomisation (the teacher datasets have none)")
    ap.add_argument("--same-seed", action="store_true", help="every scene the layout of --seed-start")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="generate_truss.Noise on the executed action (norm units), on the approach only: before the "
                         "cut and while the jaw is more than NOISE_NEAR_M from the cut point; a fresh seed per scene")
    ap.add_argument("--ensemble", type=float, default=None, help="override temporal_ensemble_coeff (ACT)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="override n_action_steps (ACT, no ensembling)")
    # post-mortem step 3 (2026-11-14): a fixed held-out set the teacher solves, and stay-home scenes
    ap.add_argument("--seeds-from", type=pathlib.Path, default=None,
                    help="a generate_truss.py attempt log (.jsonl): run its kept, non-hold seeds, at most --scenes")
    ap.add_argument("--holds", type=int, default=0, help="also this many nothing-ready scenes (the arm must stay home)")
    ap.add_argument("--hold-seed-start", type=int, default=HOLD_SEED_START)
    a = ap.parse_args()
    seeds = None
    if a.seeds_from is not None:
        rows = [json.loads(ln) for ln in a.seeds_from.expanduser().read_text().splitlines() if ln.strip()]
        seeds = [r["seed"] for r in rows if r.get("kept") is True and not r.get("hold")]
        a.seed_start = seeds[0]
    a.fast = a.fast or a.headless
    policy_path = a.policy.resolve()
    if a.seed_start is None:
        a.seed_start = int(np.random.default_rng().integers(1_000_000, 2**31 - 100_000))
    print(f"seeds from {a.seed_start} (replay one: --seed-start <its seed> --scenes 1)", flush=True)

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    train_ds, rig = trained_on(policy_path)
    meta = LeRobotDatasetMetadata(train_ds, root=HF_LEROBOT_HOME / train_ds)
    features = meta.features
    obs_feats = {k: v for k, v in features.items() if k != OBS_ENV_STATE}
    _, _, rop = make_default_processors()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    over = {}
    if a.ensemble is not None:
        over.update(temporal_ensemble_coeff=a.ensemble if a.ensemble > 0 else None,
                    n_action_steps=1 if a.ensemble > 0 else (a.n_action_steps or 100))
    elif a.n_action_steps is not None:
        over.update(temporal_ensemble_coeff=None, n_action_steps=a.n_action_steps)
    policy, pre, post, pcfg = load_policy(policy_path, device, meta, over)
    wants_env = OBS_ENV_STATE in getattr(pcfg, "input_features", {})
    ens = getattr(pcfg, "temporal_ensemble_coeff", None)   # ACT: None = plays n_action_steps per chunk instead
    print(f"policy {policy_path} ({pcfg.type}) on {device}; task {gt.TASK!r}; box + cut point as numbers: {wants_env}; "
          f"temporal ensembling {ens}; rig {rig} (trained on {train_ds})", flush=True)

    yolo = None if a.no_yolo or not gt.YOLO_TRUSS.exists() else str(gt.YOLO_TRUSS)
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=a.seed_start, viewer=not a.headless, overlay="yolo" if yolo else "true_box", realtime=False,
        fps=expert.FPS, domain_rand=not a.no_dr, settle_s=1.0, snap_to_first_action=False,
        yolo_weights=yolo, state_log_dir=None, rig=rig))
    robot.connect()
    out_dir = HERE.parent / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = policy_path.parent.name if policy_path.name == "pretrained_model" else policy_path.name
    run = policy_path.parents[2].name if policy_path.name == "pretrained_model" else ""
    log_path = out_dir / f"{run}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    results, holds, seed, quit_, n_tried = [], [], a.seed_start - 1, False, 0
    try:
        while len(results) < a.scenes and not quit_:
            if seeds is not None:
                if n_tried >= len(seeds):
                    break
                seed = seeds[n_tried]
                n_tried += 1
            else:
                seed = a.seed_start if a.same_seed else seed + 1
            robot.new_scene(seed, log=False)
            obs = robot.get_observation()              # the start frame: picks the target, like the episodes
            if robot._tgt_k is None:
                print(f"  seed {seed}: skip, no ready truss ({robot._box_info['reason']})", flush=True)
                continue
            k = robot._tgt_k
            t_info = robot.info["trusses"][k]
            n = len(results) + 1
            print(f"\n▶ scene {n}/{a.scenes}: seed {seed}, target {t_info['name']} ({t_info['n']} fruit, "
                  f"stalk {t_info['r'] * 2000:.1f} mm), box from {robot._box_info['source']}", flush=True)
            env = robot.env_state()
            policy.reset()
            noise = gt.Noise(a.noise, gt.NOISE_RHO, seed, 1000 + len(results)) if a.noise > 0 else None
            p_cut, _ = ts.cut_target_now(t_info, robot.d, robot.m)
            grip = truss_ik.cut_point_in_gripper(t_info["r"])
            gid = robot.m.body("arm/gripper").id
            closest = np.inf
            frames = 0
            watch = gt.Watch(robot)
            t0, t_crate, gave_up = robot.d.time, None, False
            while robot.d.time - t0 < EPISODE_S:
                t_f = time.time()
                frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
                if wants_env:
                    frame[OBS_ENV_STATE] = env
                with torch.inference_mode():
                    x = prepare_observation_for_inference(frame, device, gt.TASK, robot.name)
                    act = post(policy.select_action(pre(x)))
                cmd = make_robot_action(act, features)
                jaw_pt = robot.d.xpos[gid] + robot.d.xmat[gid].reshape(3, 3) @ grip
                dist = float(np.linalg.norm(jaw_pt - p_cut))
                closest = min(closest, dist) if robot.cutters[k].cut_t is None else closest
                if noise is not None:
                    near = dist < NOISE_NEAR_M or robot.cutters[k].cut_t is not None
                    cmd = noise("approach" if not near else "settle_at_stalk", cmd)   # NOISE_OFF has settle_at_stalk
                robot.send_action(cmd)
                frames += 1
                obs = robot.get_observation()
                watch.step()
                t = robot.d.time - t0
                if t_crate is None and ts.in_crate(robot.info, robot.d, robot.m, k):
                    t_crate = t
                    print(f"  🍅 truss in the crate at {t:.1f} s", flush=True)
                if t_crate is not None and t - t_crate > AFTER_CRATE_S:
                    break
                key = 0 if a.headless else show(obs, f"scene {n}/{a.scenes}  seed {seed}  t {t:4.1f} s")
                if key == ord("q"):
                    quit_ = gave_up = True
                    break
                if key == ord("n"):
                    gave_up = True
                    break
                if not a.fast:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
            cut = robot.cutters[k].cut_t is not None
            wrong = [robot.info["trusses"][j]["name"] for j, c in enumerate(robot.cutters)
                     if j != k and c.cut_t is not None]
            off = [robot.info["fruit"][j]["tomato"] for j in range(len(robot.info["fruit"]))
                   if not np.isnan(robot.det.detached_at[j])]
            in_crate = bool(ts.in_crate(robot.info, robot.d, robot.m, k))
            ok = cut and in_crate and not wrong and not off
            ok_strict = ok and watch.max <= gt.NEIGHBOUR_MOVE_M          # generate_truss.judge's neighbour rule too
            rec = dict(seed=seed, target=t_info["name"], box_source=robot._box_info["source"], success=ok,
                       success_strict=ok_strict, closest_mm=round(closest * 1000, 1), noise=a.noise, cut=cut,
                       in_crate=in_crate, wrong_cut=wrong, fruit_off=off,
                       neighbours_moved_mm=round(watch.max * 1000, 1),
                       t_crate=None if t_crate is None else round(t_crate, 2), seconds=round(robot.d.time - t0, 2),
                       gave_up=gave_up, frames=frames, rig=rig)
            results.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            verdict = ("✅ success" if ok else
                       "❌ cut the wrong truss " + ", ".join(wrong) if wrong else
                       "❌ never cut the target" if not cut else
                       "❌ cut, but not (all) in the crate" if not in_crate else
                       f"❌ in the crate, but {len(off)} fruit came off")
            print(f"  {verdict}  (neighbours moved {watch.max * 1000:.0f} mm, {frames} frames)   "
                  f"[{sum(r['success'] for r in results)}/{len(results)} so far]", flush=True)
        for i in range(0 if quit_ else a.holds):
            seed = a.hold_seed_start + i
            robot.new_scene(seed, log=False, hold=True)
            obs = robot.get_observation()
            env = robot.env_state()
            policy.reset()
            start = np.asarray(robot._present_norm(), float)
            watch, t0, moved = gt.Watch(robot), robot.d.time, 0.0
            while robot.d.time - t0 < HOLD_EVAL_S:
                frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
                if wants_env:
                    frame[OBS_ENV_STATE] = env
                with torch.inference_mode():
                    x = prepare_observation_for_inference(frame, device, gt.TASK, robot.name)
                    act = post(policy.select_action(pre(x)))
                robot.send_action(make_robot_action(act, features))
                obs = robot.get_observation()
                watch.step()
                moved = max(moved, float(np.abs(np.asarray(robot._present_norm(), float) - start).max()))
            ok, why = gt.judge_hold(robot, watch)
            if ok and moved > HOLD_MOVE_NORM:
                ok, why = False, f"the arm moved {moved:.1f} norm units"
            rec = dict(seed=seed, hold=True, success=bool(ok), why=why, moved_norm=round(moved, 1), rig=rig)
            holds.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  {'✅' if ok else '❌'} hold scene {i + 1}/{a.holds}: seed {seed}, "
                  f"{'stayed home' if ok else why} (moved {moved:.1f})", flush=True)
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()
        if results:
            s = sum(r["success"] for r in results)
            ens = getattr(pcfg, "temporal_ensemble_coeff", None)
            summary = dict(summary=True, policy=str(policy_path), policy_type=pcfg.type, temporal_ensemble_coeff=ens,
                           n_action_steps=getattr(pcfg, "n_action_steps", None), noise=a.noise, no_dr=a.no_dr,
                           same_seed=a.same_seed, success_strict=sum(r["success_strict"] for r in results),
                           closest_mm_median=float(np.median([r["closest_mm"] for r in results])),
                           rig=rig, scenes=len(results), success=s, cut=sum(r["cut"] for r in results),
                           in_crate=sum(r["in_crate"] for r in results), seed_start=a.seed_start,
                           seeds_from=None if a.seeds_from is None else str(a.seeds_from),
                           holds=len(holds), holds_ok=sum(r["success"] for r in holds))
            log.write(json.dumps(summary) + "\n")
            held = f"; holds {summary['holds_ok']}/{len(holds)}" if holds else ""
            print(f"\n{s}/{len(results)} successes, {summary['success_strict']} with neighbours within "
                  f"{gt.NEIGHBOUR_MOVE_M * 1000:.0f} mm (target cut in {summary['cut']}, in the crate in "
                  f"{summary['in_crate']}; closest approach median {summary['closest_mm_median']:.1f} mm{held}).  "
                  f"log {log_path}", flush=True)
        log.close()


if __name__ == "__main__":
    main()
