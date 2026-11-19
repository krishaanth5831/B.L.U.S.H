#!/usr/bin/env python3
"""Via-pose search for teacher v5 (2026-11-18), as VIA_NORM was found for v3 (2026-11-09): on scenes kept apart from
the tuning set (9900000-9900153) and the gate (9930000-9930199), collect the cases where a candidate's cut leg is
clear but no template (line, VIA_NORM via, chain of two) reaches its standoff from the parked arm, and an offline RRT
route there (rrt.plan, seeded, 8000 iterations; the teacher never drives one). Then which single poses, used as one
extra via, unblock the most cases.
    via_search.py collect SEEDFILE OUT.jsonl      # cases + RRT way-points (norm units)
    via_search.py score SEEDFILE CASES.jsonl CANDS.json OUT.jsonl   # per case, which candidate vias clear it
Candidates (u 0.5; tilt 0, -10, +10; every approach angle; strict and brush) as teacher v5's phase C builds them."""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sim/
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402

import rrt  # noqa: E402
import truss_expert as tx  # noqa: E402
import truss_teacher as tt  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()
from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig  # noqa: E402

MODES = ("strict", "brush")
TILTS = (0.0, -10.0, 10.0)


def robot():
    rob = SO101Mujoco(SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=30,
                                        domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig="base"))
    rob.connect()
    return rob


def cases(rob, seed):
    """(teacher, [(mode, tilt, beta, goal rad)]) for a seed: blocked-by-templates standoffs (the longest clear one)."""
    rob.new_scene(seed, log=False)
    rob.get_observation()
    if rob._tgt_k is None:
        return None, []
    T = tt.TrussTeacher(rob.m, rob.d, rob.info, rob._tgt_k)
    r, q0 = T.t["r"], T.q_now()
    out = []
    for mode in MODES:
        T._mode(mode)
        T.jaw(tx.JAW_OPEN_DEG)
        if not T.chk.free(q0, **T.kw):
            continue
        for tilt in TILTS:
            for beta in tt.APPROACH_BLEND_DEG:
                leg = T._cut_leg(0.5, beta, r, tilt, True)
                if isinstance(leg, str) or not leg["approaches"]:
                    continue
                T.jaw(tx.JAW_OPEN_DEG)
                if T._to_standoff(q0, dict(leg))[0] is not None:
                    continue
                out.append((mode, tilt, beta, np.asarray(leg["approaches"][-1][0][0], float)))
    return T, out


def collect(seedfile, out_path):
    rob = robot()
    with open(out_path, "w") as f:
        for seed in [int(x) for x in open(seedfile).read().split()]:
            T, cs = cases(rob, seed)
            for mode, tilt, beta, goal in cs:
                T._mode(mode)
                T.jaw(tx.JAW_OPEN_DEG)
                path = None
                for k in range(2):
                    path = rrt.plan(T.chk, T.q_now(), goal, np.random.default_rng([seed, int(beta) + 100, k]),
                                    lo=T.lo, hi=T.hi, iters=8000, **T.kw)
                    if path is not None:
                        break
                via = [] if path is None else [T.fm.rad_to_norm(np.r_[x, 0.0])[:5].round(2).tolist()
                                               for x in path[1:-1]]
                f.write(json.dumps(dict(seed=seed, mode=mode, tilt=tilt, beta=beta,
                                        goal=T.fm.rad_to_norm(np.r_[goal, 0.0])[:5].round(3).tolist(),
                                        rrt=path is not None, via=via)) + "\n")
                f.flush()
            print(seed, len(cs), "blocked cases", flush=True)
    rob.disconnect()


def score(seedfile, cases_path, cands_path, out_path):
    """For each recorded case of the seeds in seedfile: which candidate vias (norm units) clear it, alone or after /
    before one of VIA_NORM's (the teacher's free_move with the candidate added)."""
    rob = robot()
    want = {int(x) for x in open(seedfile).read().split()}
    rows = [json.loads(x) for x in open(cases_path) if x.strip()]
    cands = json.load(open(cands_path))
    with open(out_path, "w") as f:
        for seed in sorted(want):
            mine = [c for c in rows if c["seed"] == seed]
            if not mine:
                continue
            T, _ = cases(rob, seed)
            q0 = T.q_now()
            V = [T.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in cands]
            old = T.vias()
            for c in mine:
                T._mode(c["mode"])
                T.jaw(tx.JAW_OPEN_DEG)
                goal = T.fm.norm_to_rad(np.r_[c["goal"], 0.0])[:5]
                ok = []
                for j, v in enumerate(V):
                    clear = T.chk.path_ok([q0, v, goal], **T.kw) or any(
                        T.chk.path_ok([q0, o, v, goal], **T.kw) or T.chk.path_ok([q0, v, o, goal], **T.kw)
                        for o in old)
                    if clear:
                        ok.append(j)
                f.write(json.dumps(dict(seed=seed, mode=c["mode"], tilt=c["tilt"], beta=c["beta"], ok=ok)) + "\n")
                f.flush()
            print(seed, len(mine), "cases scored", flush=True)
    rob.disconnect()


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        score(*sys.argv[2:6])
