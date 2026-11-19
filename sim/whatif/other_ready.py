#!/usr/bin/env python3
"""What-if for K7 (2026-11-18): in scenes where teacher v5 has no successful plan for the picked target, is there
another ready truss it could harvest instead (the target picked by what the arm can reach, not by the most visible
pixels)? Per seed: the ready trusses, and for each other one: teacher v5's plan + dry run + judge (retries off).
    python other_ready.py SEED [SEED ...]     (run in the lerobot env from anywhere)"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sim/
sys.path.insert(0, SIM)
import generate_truss as gt  # noqa: E402
import truss_teacher as tt  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()
from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig  # noqa: E402

rob = SO101Mujoco(SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False, fps=30,
                                    domain_rand=False, settle_s=1.0, snap_to_first_action=False, rig="base"))
rob.connect()
for seed in map(int, sys.argv[1:]):
    rob.new_scene(seed, log=False)
    rob.get_observation()
    k0, ready = rob._tgt_k, list(rob.info["ready"])
    out = dict(seed=seed, target=None if k0 is None else rob.info["trusses"][k0]["name"],
               ready=[rob.info["trusses"][k]["name"] for k in ready], others=[])
    for k in ready:
        if k == k0:
            continue
        rob.new_scene(seed, log=False)
        rob.get_observation()
        rob._tgt_k = k                      # judge and Watch read the target from here
        try:
            T = tt.TrussTeacher(rob.m, rob.d, rob.info, k).plan()
        except tt.PlanError as e:
            out["others"].append(dict(truss=rob.info["trusses"][k]["name"], n=rob.info["trusses"][k]["n"],
                                      ok=False, why="no plan: " + str(e)[-80:]))
            continue
        t0, w = gt.play(rob, T)
        ok, why = gt.judge(rob, T, t0, w)
        out["others"].append(dict(truss=rob.info["trusses"][k]["name"], n=rob.info["trusses"][k]["n"], ok=ok,
                                  why=why, phase=T.meta.get("phase")))
    print(json.dumps(out), flush=True)
rob.disconnect()
