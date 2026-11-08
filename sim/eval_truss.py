#!/usr/bin/env python3
"""Run a trained truss-harvesting policy on the sim arm in scenes it never trained on, and watch it (K7, 2026-11-04).

    python eval_truss.py ../outputs/train/smolvla_truss_lora_v1/checkpoints/last/pretrained_model
    python eval_truss.py <policy> --scenes 5 --seed-start 1234567      # replay: the seeds are printed

Windows: the MuJoCo viewer (third person, mouse to orbit) and one per camera showing exactly what the policy gets
(truss-YOLO overlay, green target box + cut-point cross, as in the recorded episodes). Keys in a camera window:
N = give up on this scene, Q = quit. Real time (30 fps): the sim steps 1/30 s per action, the loop sleeps the rest.

Scenes: truss_scene.build at a fresh random seed per run (>= 1,000,000; the training episodes used 210000-210512,
the pilot 200000-200008), same settings as generate_truss.py's recording pass (domain randomisation, 1 s settle,
truss YOLO picks the ready truss on the first frame, true box when it's wrong). Works for any policy: ACT gets
observation.environment_state (box + cut point) if it was trained with it; SmolVLA sees them drawn in the image.

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
import truss_scene as ts  # noqa: E402

TRAIN_DATASET = "krish5831/blush_sim_truss_v1"   # features (names, shapes) come from its metadata
EPISODE_S = 45.0          # scripted truss episodes run 19-31 s (run_v1.log); 45 s gives the policy slack
AFTER_CRATE_S = 3.0       # keep going this long after the truss lands, so a late knock-off still counts


def trained_on(path):
    """(dataset repo id, rig) a checkpoint was trained on, from its train_config.json: the rig is "base" when the
    base camera was among its inputs (rig "base", 2026-11-06), else "birdseye"."""
    cfg = path / "train_config.json"
    if not cfg.exists():
        return TRAIN_DATASET, "birdseye"
    c = json.loads(cfg.read_text())
    rename = c.get("rename_map") or {}
    return c["dataset"]["repo_id"], "base" if "observation.images.base" in rename else "birdseye"


class RTCDriver:
    """Real-time chunking for SmolVLA, copied from S.H.E.A.R sim/rollout_policy.py (dev 2140393): a new 50-move chunk
    every `reinfer` moves, steered ("guided") toward the moves of the previous chunk that weren't played yet, so chunk
    handovers don't jump. Dream Machines' pi0.5 write-up: their biggest serving-side gain (53 -> 93 %). No latency in
    the sim (it waits for the policy), so inference_delay is 0. ⚠️ Not under inference_mode: guided RTC takes a
    gradient inside the denoising loop."""

    def __init__(self, policy, pre, post, reinfer):
        self.policy, self.pre, self.post, self.reinfer = policy, pre, post, reinfer
        self.H = policy.config.rtc_config.execution_horizon
        self.chunk = self.actions = None
        self.k = 0

    def reset(self):
        self.policy.reset()
        self.chunk = self.actions = None
        self.k = 0

    def __call__(self, x):
        if self.chunk is None or self.k >= self.reinfer:
            prev = None
            if self.chunk is not None:
                left = self.chunk[self.k:]
                prev = torch.zeros((self.H, left.shape[-1]), device=left.device, dtype=left.dtype)
                prev[:min(self.H, len(left))] = left[:self.H]
            new = self.policy.predict_action_chunk(self.pre(x), inference_delay=0, prev_chunk_left_over=prev)
            with torch.no_grad():
                self.chunk = new.detach().squeeze(0)
                self.actions = self.post(new.detach()).squeeze(0)
            self.k = 0
        a = self.actions[self.k]
        self.k += 1
        return a.unsqueeze(0)


class Smoother:
    """SockBot's command filter (walnut sensei part 5): a = alpha * new + (1 - alpha) * previous. He used 0.5 (jitter
    down to 38 %); 0.3 cut more but lagged. alpha 0 = off."""

    def __init__(self, alpha):
        self.alpha, self.prev = alpha, None

    def reset(self):
        self.prev = None

    def __call__(self, a):
        if self.alpha <= 0:
            return a
        self.prev = a if self.prev is None else self.alpha * a + (1.0 - self.alpha) * self.prev
        return self.prev


def load_policy(path, device, ds_meta, n_action_steps=None, rtc_horizon=None):
    """Plain checkpoints and PEFT ones (LoRA adapter on a base policy, e.g. SmolVLA) alike, via make_policy.
    n_action_steps: how many actions of each predicted chunk to play before predicting again (None = as trained).
    rtc_horizon: turn real-time chunking on (RTCDriver plays it), steering this many moves of each new chunk."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import make_policy
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = str(path)
    cfg.device = str(device)
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if rtc_horizon:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=rtc_horizon)
    rename = None
    train_cfg = path / "train_config.json"
    if train_cfg.exists():
        rename = json.loads(train_cfg.read_text()).get("rename_map") or None
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename).eval()
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
    # ⚠️ SmolVLA trains with chunk_size = n_action_steps = 50: it plays 50 actions (1.7 s) blind before looking again.
    # Fewer here re-plans more often without retraining (K7 2026-11-06, v2 fumbled at the stalk).
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="actions played per predicted chunk before predicting again (default: as trained)")
    ap.add_argument("--rtc", type=int, default=None, metavar="REINFER",
                    help="real-time chunking: a new chunk every REINFER moves (S.H.E.A.R / Dream Machines: 10)")
    ap.add_argument("--rtc-horizon", type=int, default=20, help="old-chunk moves each new chunk is steered toward")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="ALPHA",
                    help="command smoothing, ALPHA = weight of the new command (SockBot: 0.5); 0 = off")
    a = ap.parse_args()
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
    policy, pre, post, pcfg = load_policy(policy_path, device, meta, a.n_action_steps,
                                          a.rtc_horizon if a.rtc else None)
    driver = RTCDriver(policy, pre, post, a.rtc) if a.rtc else None
    smooth = Smoother(a.smooth)
    wants_env = OBS_ENV_STATE in getattr(pcfg, "input_features", {})
    # Count the policy's real predictions per scene, so an ignored n_action_steps override shows up (frames/50 vs /10)
    # ⚠️ SmolVLA's select_action refills its queue through _get_action_chunk, not predict_action_chunk
    calls = {"n": 0}
    refill = next((m for m in ("_get_action_chunk", "predict_action_chunk") if hasattr(policy, m)), None)
    if refill is not None:
        _chunk = getattr(policy, refill)

        def counted(*args, **kw):
            calls["n"] += 1
            return _chunk(*args, **kw)
        setattr(policy, refill, counted)
    n_steps = getattr(policy.config, "n_action_steps", None)
    print(f"policy {policy_path} ({pcfg.type}{', LoRA' if getattr(pcfg, 'use_peft', False) else ''}) on {device}; "
          f"task {gt.TASK!r}; box + cut point as numbers: {wants_env}; n_action_steps {n_steps}; rig {rig} "
          f"(trained on {train_ds}); "
          f"{f'RTC every {a.rtc} moves, horizon {a.rtc_horizon}' if a.rtc else 'no RTC'}; smoothing {a.smooth}",
          flush=True)

    yolo = None if a.no_yolo or not gt.YOLO_TRUSS.exists() else str(gt.YOLO_TRUSS)
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=a.seed_start, viewer=not a.headless, overlay="yolo" if yolo else "true_box", realtime=False,
        fps=expert.FPS, domain_rand=True, settle_s=1.0, snap_to_first_action=False, task="truss",
        yolo_weights=yolo, state_log_dir=None, rig=rig))
    robot.connect()
    out_dir = HERE.parent / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = policy_path.parent.name if policy_path.name == "pretrained_model" else policy_path.name
    run = policy_path.parents[2].name if policy_path.name == "pretrained_model" else ""
    steps_tag = ("" if a.n_action_steps is None else f"_nas{a.n_action_steps}") + \
        (f"_rtc{a.rtc}h{a.rtc_horizon}" if a.rtc else "") + (f"_sm{a.smooth}" if a.smooth else "")
    log_path = out_dir / f"{run}_{name}{steps_tag}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = log_path.open("w")
    results, seed, quit_ = [], a.seed_start - 1, False
    try:
        while len(results) < a.scenes and not quit_:
            seed += 1
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
            (driver or policy).reset()
            smooth.reset()
            calls["n"], frames = 0, 0
            watch = gt.Watch(robot)
            t0, t_crate, gave_up = robot.d.time, None, False
            while robot.d.time - t0 < EPISODE_S:
                t_f = time.time()
                frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
                if wants_env:
                    frame[OBS_ENV_STATE] = env
                if driver is not None:
                    act = driver(prepare_observation_for_inference(frame, device, gt.TASK, robot.name))
                else:
                    with torch.inference_mode():
                        x = prepare_observation_for_inference(frame, device, gt.TASK, robot.name)
                        act = post(policy.select_action(pre(x)))
                act = smooth(act)
                robot.send_action(make_robot_action(act, features))
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
            wrong = [robot.info["trusses"][j]["name"] for j, c in enumerate(robot.cutters) if j != k and c.cut_t is not None]
            off = [robot.info["fruit"][j]["tomato"] for j in range(len(robot.info["fruit"]))
                   if not np.isnan(robot.det.detached_at[j])]
            in_crate = bool(ts.in_crate(robot.info, robot.d, robot.m, k))
            ok = cut and in_crate and not wrong and not off
            rec = dict(seed=seed, target=t_info["name"], box_source=robot._box_info["source"], success=ok, cut=cut,
                       in_crate=in_crate, wrong_cut=wrong, fruit_off=off,
                       neighbours_moved_mm=round(watch.max * 1000, 1),
                       t_crate=None if t_crate is None else round(t_crate, 2), seconds=round(robot.d.time - t0, 2),
                       gave_up=gave_up, n_action_steps=n_steps, frames=frames, predictions=calls["n"],
                       rtc=a.rtc, rtc_horizon=a.rtc_horizon if a.rtc else None, smooth=a.smooth, rig=rig)
            results.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            verdict = ("✅ success" if ok else
                       "❌ cut the wrong truss " + ", ".join(wrong) if wrong else
                       "❌ never cut the target" if not cut else
                       "❌ cut, but not (all) in the crate" if not in_crate else
                       f"❌ in the crate, but {len(off)} fruit came off")
            print(f"  {verdict}  (neighbours moved {watch.max * 1000:.0f} mm, {calls['n']} predictions in {frames} "
                  f"frames)   "
                  f"[{sum(r['success'] for r in results)}/{len(results)} so far]", flush=True)
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()
        if results:
            s = sum(r["success"] for r in results)
            summary = dict(summary=True, policy=str(policy_path), n_action_steps=n_steps, scenes=len(results), success=s,
                           cut=sum(r["cut"] for r in results), in_crate=sum(r["in_crate"] for r in results),
                           seed_start=a.seed_start)
            log.write(json.dumps(summary) + "\n")
            print(f"\n{s}/{len(results)} successes (target cut in {summary['cut']}, in the crate in "
                  f"{summary['in_crate']}).  log {log_path}", flush=True)
        log.close()


if __name__ == "__main__":
    main()
