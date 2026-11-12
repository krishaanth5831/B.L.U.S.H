"""Step-2 gate: replay recorded episodes in the sim and check they reproduce (post-mortem note §6, step 2).

    python replay_check.py ~/.cache/blush_sim/truss/blush_sim_teacher_v3_s*_2026*.jsonl --n 10

Reads the attempt logs of generate_truss.py --part-size runs (each kept row names its seed, hold flag, noise and
part), picks --n kept episodes spread over all of them, rebuilds each scene from its seed and sends the dataset's
recorded actions frame by frame. Two checks per episode:
  - clean episodes: the arm state after every frame must match the recorded observation.state (max |diff| in norm
    units, must stay under --tol); noisy ones can't match, because the dataset holds the teacher's clean labels and
    the arm ran label + noise (DART), so their drift is printed but not judged;
  - every episode: graded like generate_truss.judge / judge_hold (cut, whole in the crate, no fruit off, no wrong
    cut; hold = nothing touched). For a noisy episode this asks whether the labels alone still harvest.
Clean episodes are picked first (at least --n // 2 when there are that many). Nothing is written or pushed.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert  # noqa: E402
import generate_truss as gt  # noqa: E402
import truss_scene as ts  # noqa: E402


def kept_rows(logs):
    """(part, local episode index, row) for every kept, recorded episode, in log order."""
    out, seen = [], {}
    for path in logs:
        for line in open(path):
            r = json.loads(line)
            if r.get("reason") != "ok" or "part" not in r:
                continue
            k = seen.get(r["part"], 0)
            seen[r["part"]] = k + 1
            out.append((r["part"], k, r))
    return out


def episode(root, ep):
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
                    for f in sorted(root.glob("data/*/*.parquet"))])
    df = df[df.episode_index == ep]
    return np.stack(df["action"].to_numpy()), np.stack(df["observation.state"].to_numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=pathlib.Path, help="attempt logs (~/.cache/blush_sim/truss/*.jsonl)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.5, help="clean episodes: max state diff, norm units")
    a = ap.parse_args()

    rows = kept_rows(a.logs)
    if not rows:
        sys.exit("no kept episodes with a part in these logs (run generate_truss.py with --part-size)")
    clean = [x for x in rows if "noise" not in x[2]]
    noisy = [x for x in rows if "noise" in x[2]]
    n_clean = min(len(clean), max(a.n // 2, a.n - len(noisy)))
    pick = lambda xs, n: [xs[i] for i in np.linspace(0, len(xs) - 1, n).round().astype(int)] if n else []  # noqa: E731
    chosen = pick(clean, n_clean) + pick(noisy, min(len(noisy), a.n - n_clean))

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig

    names = LeRobotDatasetMetadata(chosen[0][0], root=HF_LEROBOT_HOME / chosen[0][0]).features["action"]["names"]
    robot = SO101Mujoco(SO101MujocoConfig(
        id="sim", seed=chosen[0][2]["seed"], viewer=False, overlay="true_box", realtime=False, fps=expert.FPS,
        domain_rand=False, settle_s=1.0, snap_to_first_action=False, state_log_dir=None, rig="base"))
    robot.connect()
    results = []
    try:
        for part, ep, r in chosen:
            act, state = episode(HF_LEROBOT_HOME / part, ep)
            hold = bool(r.get("hold"))
            robot.new_scene(r["seed"], log=False, hold=hold)
            obs = robot.get_observation()
            if not hold and robot._box_info["target_name"] != r["target"]:
                print(f"❌ {part} ep {ep}: seed {r['seed']} picked {robot._box_info['target_name']}, "
                      f"recorded {r['target']}", flush=True)
                results.append(dict(ok=False, clean="noise" not in r)); continue
            watch = gt.Watch(robot)
            k = robot._tgt_k
            diff = 0.0
            for i, row in enumerate(act):
                cur = np.array([obs[n] for n in names])
                diff = max(diff, float(np.abs(cur - state[i]).max()))
                robot.send_action({n: float(v) for n, v in zip(names, row)})
                obs = robot.get_observation()
                watch.step()
            if hold:
                ok, why = gt.judge_hold(robot, watch)
            else:
                cut = robot.cutters[k].cut_t is not None
                off = int(np.sum(~np.isnan(robot.det.detached_at)))
                wrong = [j for j, c in enumerate(robot.cutters) if j != k and c.cut_t is not None]
                crate = bool(ts.in_crate(robot.info, robot.d, robot.m, k))
                ok = cut and crate and not wrong and not off
                why = "ok" if ok else f"cut {cut}, in crate {crate}, wrong cut {len(wrong)}, fruit off {off}"
            is_clean = "noise" not in r
            match = diff <= a.tol if is_clean else None
            results.append(dict(ok=ok, clean=is_clean, match=match))
            tag = "clean" if is_clean else "noisy, labels only"
            print(f"{'✅' if ok and match is not False else '❌'} {part} ep {ep} ({len(act)} frames, seed {r['seed']}, "
                  f"{'hold' if hold else r['target']}, {tag}): {why}; state diff max {diff:.2f}"
                  f"{'' if is_clean else ' (not judged)'}", flush=True)
    finally:
        robot.disconnect()
    c = [x for x in results if x["clean"]]
    nz = [x for x in results if not x["clean"]]
    print(f"\nclean: {sum(x['ok'] and x.get('match') for x in c)}/{len(c)} reproduce and pass; "
          f"noisy (labels only): {sum(x['ok'] for x in nz)}/{len(nz)} still pass")


if __name__ == "__main__":
    main()
