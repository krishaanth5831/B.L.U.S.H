#!/usr/bin/env python3
"""Can dataset v5's episodes be re-made from the simulator's state logs? READ-ONLY check, nothing is written to any
dataset (K7 2026-11-26: the 600 new episodes go into the same dataset as the old 800; the old 800 have two cameras at
the old spots and the faulty picture storage, so their pictures would have to be rendered again from the logged
states with the new cameras).

    python whatif/rebuild_v5_check.py [--n 12] [--png out.png]        # from sim/, env lerobot-mj4, needs a display/EGL

The recorder logged, per recorded frame, the simulator's full state (~/.cache/blush_sim/states/<session>/
scene*_seed<seed>.npz: qpos, eq_active) and per scene its seed, target and cue (session.json). A re-make sets those
states into a model REBUILT from the seed and renders. That only works if today's builder gives the same model for
the seed. Checked here for --n scenes spread over the sessions (cuts and nothing-ripe scenes):
    - same number of joints coordinates (nq) and constraints (neq) as the logged state
    - the builder's own target = the logged target
    - the cue (box + cut point in the base camera, 6 numbers) of the rebuilt scene = the logged cue
    - with the LOGGED start state set into the rebuilt model: the cue again, and how far the rebuilt scene's own
      settled start state is from the logged one (v5 was recorded on MuJoCo 3.8.1, this runs on 4.0.1)
v5's world: crate 23 x 28 cm at (0.6175, 0.17) (truss_scene.py at commit 2efd61b); today's is 30 x 28.
"""
import argparse
import ast
import glob
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
STATES = pathlib.Path.home() / ".cache" / "blush_sim" / "states"
V5_SESSIONS = ("20261118_21", "20261119_08")           # the recording of 18-19 Nov (32 parts, 800 episodes)
V5_CRATE = ((0.6175, 0.17), (0.23, 0.28, 0.05))        # truss_scene.CRATE_CENTRE, CRATE_SIZE at commit 2efd61b


def v5_scenes():
    """Every logged scene of the v5 recording: dict(session, seed, hold, target, env_state, dataset, episode, npz)."""
    out = []
    for s in sorted(d for d in os.listdir(STATES) if V5_SESSIONS[0] <= d <= V5_SESSIONS[1]):
        for sc in json.load(open(STATES / s / "session.json"))["scenes"]:
            f = glob.glob(str(STATES / s / f"scene*_seed{sc['seed']}.npz"))
            meta = sc.get("scripted")
            meta = ast.literal_eval(meta) if isinstance(meta, str) else (meta or {})
            out.append(dict(session=s, seed=int(sc["seed"]), target=sc.get("target"), env_state=sc.get("env_state"),
                            dataset=sc.get("dataset"), episode=sc.get("episode_index"), npz=f[0] if f else None,
                            hold=bool(meta.get("hold")) or sc.get("target") is None, frames=sc.get("frames"),
                            cut_u=float(meta.get("cut_u", 0.5))))
    return out


def set_v5_world():
    import truss_scene as ts
    ts.set_world("v5")
    assert (ts.CRATE_CENTRE, ts.CRATE_SIZE) == V5_CRATE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--png", type=pathlib.Path, default=None, help="save the base picture of the first scene's "
                    "logged start state (rebuilt model) for a look")
    a = ap.parse_args()
    S = v5_scenes()
    holds = [x for x in S if x["hold"]]
    cuts = [x for x in S if not x["hold"]]
    print(f"{len(S)} logged scenes of the v5 recording: {len(cuts)} cuts, {len(holds)} nothing-ripe; "
          f"{sum(x['npz'] is None for x in S)} without a state file")
    pick = [cuts[i] for i in np.linspace(0, len(cuts) - 1, max(1, a.n - 2)).astype(int)] + holds[:1] + holds[-1:]
    set_v5_world()
    import mujoco
    import expert
    import truss_target as ttg
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    robot = SO101Mujoco(SO101MujocoConfig(id="sim", seed=pick[0]["seed"], viewer=False, overlay="true_box",
                                          realtime=False, fps=expert.FPS, domain_rand=False, settle_s=1.0,
                                          snap_to_first_action=False, yolo_weights=None, state_log_dir=None,
                                          rig="base"))
    robot.connect()
    ok_all = True
    try:
        for x in pick:
            z = np.load(x["npz"], allow_pickle=True)
            robot.new_scene(x["seed"], log=False, hold=x["hold"])
            robot.get_observation()                             # the start frame: picks the target
            if x["cut_u"] != 0.5:
                robot.set_cut_u(x["cut_u"])                     # as the recorder: the cue marks the teacher's cut_u
            m, d = robot.m, robot.d
            same_n = (m.nq, m.neq) == (z["qpos"].shape[1], z["eq_active"].shape[1])
            tgt = None if robot._tgt_k is None else robot.info["trusses"][robot._tgt_k]["name"]
            env = np.asarray(robot.env_state(), float)
            want = np.asarray(x["env_state"] if x["env_state"] is not None else np.zeros(6), float)
            cue_px = float(np.abs((env - want) * np.array([640, 480, 640, 480, 640, 480])).max())
            line = (f"seed {x['seed']} ({'hold' if x['hold'] else 'cut '}, {x['dataset'][-3:]} ep {x['episode']}): "
                    f"nq {m.nq} vs {z['qpos'].shape[1]}, neq {m.neq} vs {z['eq_active'].shape[1]} | target {tgt} vs "
                    f"{x['target']} | cue off by {cue_px:.2f} px")
            good = same_n and tgt == x["target"] and cue_px < 0.6
            if same_n:
                settle = float(np.abs(d.qpos - z["qpos"][0]).max())
                d.qpos[:] = z["qpos"][0]
                d.eq_active[:] = z["eq_active"][0]
                mujoco.mj_forward(m, d)
                if robot._tgt_k is not None:
                    v = ttg.view_target(robot._cameras(), d, m, robot.info["trusses"][robot._tgt_k],
                                        robot.cam_names[0], u=robot.cut_u)
                    env2 = np.asarray(ttg.env_vector(*v), float)
                    cue2 = float(np.abs((env2 - want) * np.array([640, 480, 640, 480, 640, 480])).max())
                    good = good and cue2 < 0.6
                    line += f" | from the logged state {cue2:.2f} px"
                line += f" | own settle vs logged start: largest coordinate difference {settle:.4f}"
                if a.png is not None and x is pick[0]:
                    import cv2
                    a.png.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(a.png), cv2.cvtColor(robot._cameras().rgb(d, robot.cam_names[0]),
                                                         cv2.COLOR_RGB2BGR))
            print(("OK   " if good else "DIFF ") + line, flush=True)
            ok_all = ok_all and good
    finally:
        robot.disconnect()
    print("\nrebuild check:", "PASS: every checked scene rebuilds to the logged model, target and cue" if ok_all
          else "FAIL: see the DIFF lines")


if __name__ == "__main__":
    main()
