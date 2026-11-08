#!/usr/bin/env python3
"""Run a trained policy on the sim arm in vine layouts it never trained on, and watch it (K7, 2026-11-01).

    python eval_policy.py ../outputs/train/act_scripted_v1/checkpoints/last/pretrained_model
    python eval_policy.py <policy> --scenes 10 --min-novel-mm 40      # targets further from the training ones

Windows: the MuJoCo viewer (third person, mouse to orbit) and one per camera, showing exactly what the
policy gets (YOLO overlay included, as in training). Keys in a camera window: N = give up on this scene,
Q = quit. Runs at real time (30 fps): the sim steps 1/30 s per action, like generate_scripted.py, and the
loop sleeps off the rest of each frame.

Novel scenes: every run starts at a fresh random seed (>= 1,000,000, so never one of the training seeds
100000-100640), so each run shows new tomato positions and orientations. The seeds are printed; replay
one scene with --seed-start <seed> --scenes 1. Every scene runs (K7, 2026-11-01: no skipping); each
prints how far its target is from the nearest training episode's target (positions from rebuilding those
200 seeds once; cached in ~/.cache/blush_sim/train_targets_<name>.json). --min-novel-mm X skips the close
ones again. Only a scene with no ripe target at all is skipped: there is nothing to grade.
Same scene settings as the scripted episodes: domain randomisation on, vines settled 1 s, YOLO overlay.

Graded like generate_scripted.py's judge, at the end of each scene: success = target in the crate, no
other fruit came off. Also logged: target picked at all, which others came off, time to the crate.
Results: ../outputs/eval/<policy name>_<stamp>.jsonl. Nothing is recorded or pushed.
Run in the lerobot env, outside the Claude sandbox (it hides the GPU and the display).
⚠️ Single-fruit task only, and it can't run as is: it reads its features from TRAIN_DATASET, deleted locally and
on the Hub 2026-11-03 (K7). Whole trusses: eval_truss.py.
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
import scene  # noqa: E402
from add_box_feature import box_vector  # noqa: E402
from generate_scripted import task_string  # noqa: E402

# ⚠️ Deleted locally and on the Hub 2026-11-03 (K7): main() reads the features from it, so it would fail. train_targets()
# still runs off its cache file (present 2026-11-08); the TRAIN_SESSION rebuild needs no dataset either.
TRAIN_DATASET = "krish5831/blush_sim_scripted_v1"
# The recording session of that dataset (its session.json names each kept episode's seed + target).
TRAIN_SESSION = pathlib.Path.home() / ".cache" / "blush_sim" / "states" / "20261029_173515"
CACHE = pathlib.Path.home() / ".cache" / "blush_sim"
EPISODE_S = 40.0          # scripted episodes run 20-27 s (2026-10-29 log); 40 s gives the policy slack
AFTER_CRATE_S = 3.0       # keep going this long after the target lands, so a late knock-off still counts
CAMERAS = ("birdseye", "claw")


def sim_config(**kw):
    from lerobot_robot_so101_mujoco import SO101MujocoConfig
    # Same as generate_scripted.py's recording pass, except the viewer and no state log.
    base = dict(id="sim", realtime=False, fps=expert.FPS, domain_rand=True, settle_s=1.0,
                snap_to_first_action=False, state_log_dir=None)
    return SO101MujocoConfig(**{**base, **kw})


def train_targets():
    """World position of every training episode's target fruit, after the 1 s settle."""
    cache = CACHE / f"train_targets_{TRAIN_DATASET.split('/')[-1]}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text())["xyz"])
    from lerobot_robot_so101_mujoco import SO101Mujoco
    kept = [s for s in json.loads((TRAIN_SESSION / "session.json").read_text())["scenes"] if "episode_index" in s]
    print(f"rebuilding the {len(kept)} training scenes once for their target positions ...", flush=True)
    robot = SO101Mujoco(sim_config(viewer=False, overlay="none"))
    robot.connect()
    xyz = []
    for s in kept:
        robot.new_scene(s["seed"], log=False)
        xyz.append(robot.d.xpos[robot.m.body(s["target"]).id].tolist())
    robot.disconnect()
    cache.write_text(json.dumps({"dataset": TRAIN_DATASET, "seeds": [s["seed"] for s in kept], "xyz": xyz}))
    return np.array(xyz)


def load_policy(path, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, pre, post


def show(obs, label):
    for c in CAMERAS:
        img = cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(f"{c} camera (what the policy sees)", img)
    return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=pathlib.Path, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--scenes", type=int, default=20, help="novel scenes to run")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="first scene seed; default: a random one per run (replay a scene with its printed seed)")
    ap.add_argument("--min-novel-mm", type=float, default=0.0,
                    help="skip scenes whose target is closer than this to a training target "
                         "(a tomato is ~28 mm wide); 0 = run all")
    ap.add_argument("--fast", action="store_true", help="don't sleep to real time")
    a = ap.parse_args()
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
    from lerobot_robot_so101_mujoco import SO101Mujoco

    seen = train_targets()
    features = LeRobotDatasetMetadata(TRAIN_DATASET, root=HF_LEROBOT_HOME / TRAIN_DATASET).features
    _, _, rop = make_default_processors()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = load_policy(policy_path, device)
    task = task_string()
    print(f"policy {policy_path} ({policy.config.type}) on {device}; task {task!r}", flush=True)

    robot = SO101Mujoco(sim_config(viewer=True, overlay="yolo", seed=a.seed_start))
    robot.connect()
    out_dir = HERE.parent / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = policy_path.parent.name if policy_path.name == "pretrained_model" else policy_path.name
    run = policy_path.parents[2].name if policy_path.name == "pretrained_model" else ""
    log_path = out_dir / f"{run}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    results, seed, quit_ = [], a.seed_start - 1, False
    try:
        while len(results) < a.scenes and not quit_:
            seed += 1
            robot.new_scene(seed, log=False)
            obs = robot.get_observation()              # the start frame: picks the target, like training
            if robot._tgt_k is None:
                print(f"  seed {seed}: skip, no target ({robot._box_info['reason']})", flush=True)
                continue
            tgt = robot.d.xpos[robot._tgt_body].copy()
            novel_mm = float(np.linalg.norm(seen - tgt, axis=1).min() * 1000)
            if novel_mm < a.min_novel_mm:
                print(f"  seed {seed}: skip, target only {novel_mm:.0f} mm from a training target", flush=True)
                continue
            n = len(results) + 1
            print(f"\n▶ scene {n}/{a.scenes}: seed {seed}, target {robot._box_info['target_name']} "
                  f"({robot._box_info['source']}), {novel_mm:.0f} mm from the nearest training target", flush=True)
            policy.reset()
            t0, t_crate, k = robot.d.time, None, robot._tgt_k
            gave_up = False
            while robot.d.time - t0 < EPISODE_S:
                t_f = time.time()
                frame = build_dataset_frame(features, rop(obs), prefix=OBS_STR)
                if policy.config.env_state_feature:      # the box as numbers (add_box_feature.py)
                    frame[OBS_ENV_STATE] = box_vector(robot._box)
                with torch.inference_mode():
                    x = prepare_observation_for_inference(frame, device, task, robot.name)
                    act = post(policy.select_action(pre(x)))
                robot.send_action(make_robot_action(act, features))
                obs = robot.get_observation()
                t = robot.d.time - t0
                if t_crate is None and scene.in_crate(robot.d.xpos[robot._tgt_body]):
                    t_crate = t
                    print(f"  🍅 target in the crate at {t:.1f} s", flush=True)
                if t_crate is not None and t - t_crate > AFTER_CRATE_S:
                    break
                key = show(obs, f"scene {n}/{a.scenes}  seed {seed}  t {t:4.1f} s")
                if key == ord("q"):
                    quit_ = gave_up = True
                    break
                if key == ord("n"):
                    gave_up = True
                    break
                if not a.fast:
                    time.sleep(max(0.0, 1 / expert.FPS - (time.time() - t_f)))
            det = robot.det
            picked = not np.isnan(det.detached_at[k])
            others = [robot.info["fruit"][j]["tomato"] for j in range(len(det.detached_at))
                      if j != k and not np.isnan(det.detached_at[j])]
            in_crate = bool(scene.in_crate(robot.d.xpos[robot._tgt_body]))
            ok = in_crate and not others
            rec = dict(seed=seed, target=robot._box_info["target_name"], target_source=robot._box_info["source"],
                       novel_mm=round(novel_mm, 1), success=ok, target_picked=picked, target_in_crate=in_crate,
                       others_off=others, t_crate=None if t_crate is None else round(t_crate, 2),
                       seconds=round(robot.d.time - t0, 2), gave_up=gave_up)
            results.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            verdict = "✅ success" if ok else ("❌ in crate but knocked off " + ", ".join(others) if in_crate
                                              else "❌ target picked, not in the crate" if picked
                                              else "❌ target never picked")
            print(f"  {verdict}   [{sum(r['success'] for r in results)}/{len(results)} so far]", flush=True)
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()
        if results:
            s = sum(r["success"] for r in results)
            summary = dict(summary=True, policy=str(policy_path), scenes=len(results), success=s,
                           picked=sum(r["target_picked"] for r in results),
                           in_crate=sum(r["target_in_crate"] for r in results),
                           min_novel_mm=a.min_novel_mm, seed_start=a.seed_start)
            log.write(json.dumps(summary) + "\n")
            print(f"\n{s}/{len(results)} successes on novel scenes (target picked in {summary['picked']}, "
                  f"in the crate in {summary['in_crate']}).  log {log_path}", flush=True)
        log.close()


if __name__ == "__main__":
    main()
