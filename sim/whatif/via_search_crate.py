#!/usr/bin/env python3
"""Via-pose search for the v6 teacher's carry to the crate (2026-11-25), as via_search.py found VIA_NORM_V5 for the
way in (2026-11-18). At base 0.34/0.14 with the new start pose, 6-8 of the 14 no-plan scenes in seeds 9730030-179
were blocked only from the retreat's end to the crate (census, outputs/v6/census_noplan_*.jsonl): the line and the
fixed vias all hit a neighbour's fruit with the cut truss in hand.

    python whatif/via_search_crate.py collect --base 0.34,0.14 --seeds 9740000:9740060 --out via/crate_cases.jsonl
    python whatif/via_pick.py cands via/crate_cases.jsonl                     # -> via/cands.json (as for v5)
    python whatif/via_search_crate.py score --base 0.34,0.14 --seeds 9740000:9740060 --cands via/cands.json \\
        --out via/crate_score.jsonl
    python whatif/via_pick.py pick via/crate_score.jsonl                      # -> via/chosen.json

collect: plans every scene (TrussTeacher.plan, nothing played) with TrussTeacher._via wrapped: each time the carry
from a retreat's end to a release pose is blocked, the case (retreat end, the pose above the crate) gets an offline
RRT route with that same carried-truss checker (rrt.plan, seeded, ITERS; the teacher never drives one, rule 3,
K7 2026-11-09), at most CASES_PER_SEED distinct cases per scene. score: plans the scenes again, and at each of the
same first CASES_PER_SEED blocked cases tests every candidate via alone and chained with one of the teacher's
(free_move's order). Only scenes that found no plan are written (--all: every scene). Seeds 9740000+ are used by
nothing else (tuning 9730000-9730179, final test 9750000-9750199).
The carry model is truss_expert.V6_CARRY (set it with BLUSH_V6, as for base_grid). lerobot-mj4, outside the sandbox.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sim/
sys.path.insert(0, SIM)
import numpy as np  # noqa: E402

import scene  # noqa: E402

CASES_PER_SEED = 6     # distinct blocked carries per scene that get an RRT run / are scored (guess: a no-plan scene
                       # hits thousands, mostly the same retreat ends and release poses again)
ITERS = 8000           # RRT iterations per try, 2 tries (via_search.py's numbers)
KEY_ROUND = 2          # cases are the same when retreat end + pose above the crate agree to this many decimals (rad)
ESCAPES = 300          # start not free: random poses tried as a first hop off it (guess)
ESC_RAD = 0.35         # ... each joint within this of the retreat's end (20 deg, guess)


def setup(base, jaw, old_start=False):
    """scene.ARM_BASE + START_NORM for this base (base_grid.find_start; old_start: scene.START_NORM as it is), then
    the simulator."""
    import base_grid
    b = [float(v) for v in base.split(",")]
    scene.ARM_BASE = (b[0], b[1], 0.0)
    if not old_start:
        scene.START_NORM = tuple(float(v) for v in base_grid.find_start(jaw)[1])
    import expert
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    from lerobot_robot_so101_mujoco import SO101Mujoco, SO101MujocoConfig
    rob = SO101Mujoco(SO101MujocoConfig(id="sim", seed=0, viewer=False, overlay="true_box", realtime=False,
                                        fps=expert.FPS, domain_rand=False, settle_s=1.0, snap_to_first_action=False,
                                        rig="base", start_jitter_deg=0.0, yolo_weights=None, jaw=jaw))
    rob.connect()
    return rob


def seeds_of(spec):
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":"))
        return list(range(a, b))
    return [int(x) for x in spec.split(",")]


def run(a):
    rob = setup(a.base, a.jaw, a.old_start)
    import rrt
    import truss_expert as tx
    import truss_teacher as tt
    orig = tt.TrussTeacher._via
    cands = None
    if a.mode == "score":
        cands = json.load(open(a.cands))
    out = open(a.out, "w")
    st = dict(seed=None, seen=set(), rows=[], control=0)

    def routed(self, cchk, xa, xb, seed3, kw):
        """(route or None, facts) for one carry xa -> xb. ⚠️ rrt.plan gives up at once when xa or xb touches
        something, but the teacher's own moves never check a path's first way-point (Checker.path_ok). So when only
        the start touches: a first hop off it by the teacher's rule (segment_free skips its first point) onto a free
        pose within ESC_RAD, and the RRT from there. No hop found = no via-pose can help either, the retreat's end is
        the blocker."""
        hs, hg = cchk.hits(xa, **kw), cchk.hits(xb, **kw)
        facts = dict(start_free=not hs, goal_free=not hg, start_hits=sorted(hs)[:6], goal_hits=sorted(hg)[:6])
        q0, path = xa, None
        if hs and not hg:
            rng = np.random.default_rng([*seed3, 99])
            q0 = None
            for _ in range(ESCAPES):
                q = np.clip(xa + rng.uniform(-ESC_RAD, ESC_RAD, 5), self.lo, self.hi)
                if cchk.free(q, **kw) and cchk.segment_free(xa, q, **kw):
                    q0 = q
                    break
            facts.update(escape=q0 is not None)
        if q0 is not None and not hg:
            for k in range(2):
                path = rrt.plan(cchk, q0, xb, np.random.default_rng([*seed3, k]), lo=self.lo, hi=self.hi,
                                iters=ITERS, **kw)
                if path is not None:
                    break
            if path is not None and q0 is not xa:
                path = [xa] + list(path)
        return path, facts

    def wrapped(self, rel, ret, opt, cchk, r, alternate, extra=False):
        n5 = lambda q: self.fm.rad_to_norm(np.r_[q, 0.0])[:5].round(2).tolist()   # noqa: E731
        if a.control and st["control"] < a.control:      # a carry the teacher CAN do: does the RRT find one too?
            xa, xb = np.asarray(ret[-1], float), np.asarray(rel[1], float)
            kw = dict(soft_clearance=tx.CARRY_CLEAR_MIN_M)
            fm_path, label = self.free_move(xa, xb, cchk, r, extra=extra, **kw)
            if fm_path is not None:
                st["control"] += 1
                path, facts = routed(self, cchk, xa, xb, [st["seed"], 500 + st["control"]], kw)
                st["rows"].append(dict(seed=st["seed"], control=True, teacher=str(label), rrt=path is not None,
                                       via=[] if path is None else [n5(x) for x in path[1:-1]], **facts))
        res = orig(self, rel, ret, opt, cchk, r, alternate, extra=extra)
        if res != "retreat -> crate blocked, via-pose too" or len(st["seen"]) >= CASES_PER_SEED:
            return res
        xa, xb = np.asarray(ret[-1], float), np.asarray(rel[1], float)
        key = (tuple(np.round(xa, KEY_ROUND)), tuple(np.round(xb, KEY_ROUND)))
        if key in st["seen"]:
            return res
        st["seen"].add(key)
        kw = dict(soft_clearance=tx.CARRY_CLEAR_MIN_M)        # free_move's looser clearance, the last it tries
        row = dict(seed=st["seed"], case=len(st["seen"]) - 1, retreat_end=n5(xa), above_crate=n5(xb),
                   soft_clearance=self.kw.get("soft_clearance"))
        if a.mode == "collect":
            path, facts = routed(self, cchk, xa, xb, [st["seed"], row["case"]], kw)
            row.update(rrt=path is not None, via=[] if path is None else [n5(x) for x in path[1:-1]], **facts)
        else:
            V = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in cands]
            old = self.vias() + (self.vias_v5() if extra else [])
            ok = []
            for j, v in enumerate(V):
                if cchk.path_ok([xa, v, xb], **kw) or any(
                        cchk.path_ok([xa, o, v, xb], **kw) or cchk.path_ok([xa, v, o, xb], **kw) for o in old):
                    ok.append(j)
            row.update(ok=ok)
        st["rows"].append(row)
        return res

    tt.TrussTeacher._via = wrapped
    for seed in seeds_of(a.seeds):
        st.update(seed=seed, seen=set(), rows=[], control=0)
        rob.new_scene(seed, log=False)
        rob.get_observation()
        if rob._tgt_k is None:
            continue
        why = ""
        try:
            tt.TrussTeacher(rob.m, rob.d, rob.info, rob._tgt_k).plan()
            planned = True
        except tx.PlanError as e:
            planned, why = False, str(e)[:300]
        if a.scenes:                        # one line per scene: did it plan, and why not (the census of this run)
            with open(a.scenes, "a") as f:
                f.write(json.dumps(dict(seed=seed, planned=planned, why=why)) + "\n")
        for row in st["rows"]:              # the vias are for scenes with no plan at all (--all: every scene);
            if not planned or a.all or row.get("control"):          # control rows always
                out.write(json.dumps(dict(row, planned=planned)) + "\n")
        out.flush()
        print(f"{seed}: plan {planned}, {len(st['rows'])} blocked carries", flush=True)
    rob.disconnect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("collect", "score"))
    ap.add_argument("--base", default="0.34,0.14")
    ap.add_argument("--jaw", default="v6", choices=scene.JAWS)
    ap.add_argument("--seeds", required=True, help="a:b (range) or a comma list")
    ap.add_argument("--cands", help="score: candidate vias (norm units, via_pick.py cands)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="keep the cases of scenes that found a plan too")
    ap.add_argument("--old-start", action="store_true", help="keep scene.START_NORM (the old setup: --base 0.37,0.20)")
    ap.add_argument("--scenes", help="append one line per scene here: seed, planned, why not")
    ap.add_argument("--control", type=int, default=0,
                    help="collect: also run the RRT on the first N carries per scene that the teacher CAN do (rows "
                         "with control=true): the check that 'no route' is not the search's own fault")
    run(ap.parse_args())
