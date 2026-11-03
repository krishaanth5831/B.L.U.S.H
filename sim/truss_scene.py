"""Crowded truss scenes for whole-truss harvesting (K7, 2026-11-03).

    m, d, info = build(seed)            the greenhouse, 3-4 stems with 1-3 trusses each, the cut-and-hold arm
    info["trusses"]                     truss.truss_xml infos, each with "ready", "cut_ik" (a cut pose exists)
    info["ready"]                       indices of the ready trusses the arm can cut (1-2 per scene)
    in_crate(info, d, m)                every fruit of the target truss inside the (bigger) truss crate

Same greenhouse, cameras, arm base and start pose as scene.build (the single-fruit scenes). Changed:
- every truss carries 6-8 fruit on a cuttable stalk (truss.py), ripening from the stem end;
- more stems and trusses inside the reach, trusses crossing in front of each other (K7: "crowded, so the arm
  learns tough situations"); fruit never overlap, stalks may pass close by fruit of other trusses;
- 1-2 ready trusses per scene, all with a cut pose (truss_ik.py), next to "almost" decoys (red, tip still
  green) that the box has to tell apart, and unripe ones;
- only trusses the arm can reach and close on (cut pose + a clear approach) are made ready;
- the arm has the rigid cut-and-hold fingers (scene.add_cut_hold);
- the crate is bigger: a truss is ~10-15 cm long;
- trusses hanging into the arm at its start pose are left out (START_CLEAR_M).
"""
import mujoco
import numpy as np

import scene
import tomato as tm
import truss as tr
import truss_ik

# --- layout (design choices, K7 2026-11-03 "crowded"; ranges unverified against a real greenhouse) ---
# Along the row, in front of the arm (scene.STEM_X was 0.26-0.48). ⚠️ Not past 0.48: trusses of a stem at 0.52
# spread over the crate, and the carried truss hit them while lowering in (6 of 14 dry runs, 2026-11-03).
STEM_X = (0.20, 0.48)
# ⚠️ More than it looks: trusses hanging into the arm's start pose are dropped afterwards (START_CLEAR_M), which
# left 1-5 per scene at 3-4 stems x 1-3 trusses (2026-11-03).
N_STEMS = (4, 5)
STEM_GAP = 0.05                  # m between stems (scene.STEM_GAP 0.08): trusses of neighbouring stems overlap
TRUSSES_PER_STEM = (2, 3)
KNUCKLE_Z = (0.14, 0.32)         # where a stalk leaves its stem. truss_ik scan (2026-11-03): cut poses exist
                                 # at 0.10-0.22 m for a 45 deg droop and 0.18-0.34 m for 70 deg
KNUCKLE_GAP_Z = 0.07             # m between trusses on one stem
# Stalk direction: toward the aisle, turned along the row by SPREAD, tilted down by DROOP. Steep stalks
# because a level one has no cut pose above ~0.2 m (truss_ik scan); real peduncles bend down under the fruit.
DROOP_DEG = (40.0, 70.0)
SPREAD_DEG = 35.0                # +- along the row
STEM_R = 0.004                   # scene.vines_xml's stem capsule
TRIES = 30                       # draws per truss before a stem slot is left empty
N_READY = (1, 2)
STAGE_P = {"almost": 0.5, "unripe": 0.5}   # the other trusses (design choice: half are hard decoys)
# Crate for whole trusses: ~10-15 cm long trusses need ~18 cm. ⚠️ A truss hangs 12-15 cm below the grip, so
# the release point is high; an IK scan (2026-11-03) reached it over x 0.56-0.62 but not near the base
# (x < 0.50, y < 0.18), where the first spot (0.56, 0.17) put half the crate.
# IK scan: reachable over x 0.56-0.62, y 0.06-0.26. ⚠️ Toward the aisle: at y 0.17 the arm reaching over the crate
# passed through trusses hanging out from the row (seed 200000, 2026-11-03). x max 0.73 < the end frame at 0.74.
CRATE_CENTRE = (0.6175, 0.14)
# 22 x 18 cm: held by its stalk a truss sticks out 10-17 cm from the jaw, and at 18 x 16 the release search
# found no pose fitting every fruit inside the walls in 3 of 14 dry runs (2026-11-03).
# As big as the spot allows: x 0.50-0.735 (end frame tie at 0.74), y 0.03-0.25 (aisle anchor rail to 0.025). ⚠️ A
# 6-8 fruit truss lying flat is up to ~20 cm long: at 20 x 18 cm no release pose fit 2 of 3 seeds (2026-11-03).
CRATE_SIZE = (0.23, 0.22, 0.05)


IN_CRATE_TOP_M = 0.03            # a fruit may sit this far over the rim: the truss leans on a wall, still inside

def in_crate(info, d, m, k):
    """Every fruit of truss k inside the crate walls, at most IN_CRATE_TOP_M over the rim (lying, or leaning on a
    wall: a 6-8 fruit truss is up to ~20 cm long, about the crate's size)."""
    (cx, cy), (sx, sy, h) = CRATE_CENTRE, CRATE_SIZE
    for f in info["trusses"][k]["fruit"]:
        p = d.xpos[m.body(f["tomato"]).id]
        if not (abs(p[0] - cx) < sx / 2 and abs(p[1] - cy) < sy / 2 and p[2] < h + IN_CRATE_TOP_M):
            return False
    return True


def _direction(rng, toward_aisle):
    sp, dr = np.radians(rng.uniform(-SPREAD_DEG, SPREAD_DEG)), np.radians(rng.uniform(*DROOP_DEG))
    return np.array([np.sin(sp) * np.cos(dr), toward_aisle * np.cos(sp) * np.cos(dr), -np.sin(dr)])


def _stalk_spheres(t):
    """The stalk as spheres (centre, radius) so later trusses' fruit keep clear of it."""
    pts = [t["knuckle"] + (t["peduncle_end"] - t["knuckle"]) * u for u in np.linspace(0, 1, 6)]
    pts += [t["peduncle_end"] + (t["rachis_end"] - t["peduncle_end"]) * u for u in np.linspace(0, 1, 8)]
    return [(p, t["r"] + 0.002) for p in pts]


def cut_target(t, rng=None):
    """Cut point (world) and stalk direction for truss info t: the middle of its cut zone, or a draw in it."""
    u = 0.5 if rng is None else rng.uniform(0, 1)
    a, b = t["cut_zone"]
    return a + (b - a) * u, t["dir"]


CRATE_KEEP_OUT_M = 0.04          # no truss hangs within this of the crate's outline (seen from above). ⚠️ Trusses of
                                 # the right-hand stems spread over it, and the arm reaching in brushed them (2026-11-03)


def _over_crate(t):
    (cx, cy), (sx, sy, _) = CRATE_CENTRE, CRATE_SIZE
    pts = [f["pos"] for f in t["fruit"]] + [t["peduncle_end"], t["rachis_end"]]
    return any(abs(p[0] - cx) < sx / 2 + CRATE_KEEP_OUT_M and abs(p[1] - cy) < sy / 2 + CRATE_KEEP_OUT_M for p in pts)


def cut_target_now(t, d, m, u=0.5):
    """Cut point and stalk direction (world) where the truss hangs NOW (it sags and sways on its knuckle).
    The truss body's frame is world-aligned at the knuckle when built (truss.truss_xml)."""
    b = m.body(t["body"]).id
    R, o = d.xmat[b].reshape(3, 3), d.xpos[b]
    a, c = t["cut_zone"]
    p = a + (c - a) * u
    return o + R @ (p - t["knuckle"]), R @ t["dir"]


def vines_xml(rng):
    y_stem = scene.TARGET_ROW - np.sign(scene.TARGET_ROW - scene.ARM_BASE[1]) * scene.KEYHOLE_OFF
    toward = -np.sign(scene.TARGET_ROW - scene.ARM_BASE[1])
    n = int(rng.integers(N_STEMS[0], N_STEMS[1] + 1))
    xs = []
    for _ in range(200):                               # ⚠️ bounded, as scene.vines_xml
        x = rng.uniform(*STEM_X)
        if all(abs(x - x2) >= STEM_GAP for x2 in xs):
            xs.append(x)
        if len(xs) == n:
            break
    static, knuckles, trusses_xml, bodies, eqs, excl, trusses, placed = [], [], [], [], [], [], [], []
    for i, x in enumerate(sorted(xs)):
        static.append(f'<geom name="stem{i}" type="capsule" fromto="{x:.4f} {y_stem:.4f} {scene.CROP_Z:.4f} '
                      f'{x:.4f} {y_stem:.4f} 0.05" size="{STEM_R}" rgba="{tm.STEM_RGBA}" contype="0" conaffinity="0"/>')
        zs = []
        for _ in range(int(rng.integers(TRUSSES_PER_STEM[0], TRUSSES_PER_STEM[1] + 1))):
            for _ in range(20):
                z = rng.uniform(*KNUCKLE_Z)
                if all(abs(z - z2) >= KNUCKLE_GAP_Z for z2 in zs):
                    zs.append(z)
                    break
        for k, z in enumerate(sorted(zs)):
            for _ in range(TRIES):
                dvec = _direction(rng, toward)
                k0 = np.array([x, y_stem, z]) + np.r_[dvec[:2] / np.linalg.norm(dvec[:2]), 0] * STEM_R
                out = tr.truss_xml(f"s{i}t{k}", k0, dvec, rng, "unripe", stem_r=STEM_R, avoid=placed)
                if out is None:
                    continue
                kn, body, fb, eq, ex, t = out
                lowest = min(f["pos"][2] - f["radius"] for f in t["fruit"])
                if lowest < 0.02:                      # fruit on the floor
                    continue
                if _over_crate(t):
                    continue
                t["stem"] = i
                knuckles.append(kn); trusses_xml.append(body); bodies.append(fb); eqs.append(eq); excl.append(ex)
                trusses.append(t)
                placed += [(f["pos"], f["radius"]) for f in t["fruit"]] + _stalk_spheres(t)
                break
    return static, list(zip(knuckles, trusses_xml, bodies, eqs, excl)), trusses


BUILD_SETTLE_S = 0.5             # the vines settle this long before reach is judged (the plugin settles 1 s more)
START_CLEAR_M = 0.01             # no truss within this of the arm at the start pose (it rests there 0.4-0.8 s
                                 # and the vines settle around it; fruit hanging into it fell off, 2026-11-03)


def _compose(static, parts, arm):
    xml = "\n".join(static + [p[0] for p in parts] + [p[1] for p in parts])
    return scene.compose(xml, "\n".join(p[2] for p in parts), "\n".join(p[3] for p in parts), arm, cut_hold=True,
                         excludes="\n".join(p[4] for p in parts), crate=(CRATE_CENTRE, CRATE_SIZE))


def _start_pose(m, d):
    import arm_map
    q, u = scene.arm_ids(m)
    d.qpos[q] = arm_map.FollowerMap().norm_to_rad(scene.START_NORM)
    d.ctrl[u] = d.qpos[q]


def _touching_arm(m, d, trusses):
    """Indices of trusses with a stalk or fruit within START_CLEAR_M of the arm at the start pose."""
    import copy
    mc = copy.deepcopy(m)
    dc = mujoco.MjData(mc)
    dc.qpos[:] = d.qpos
    arm = np.array([mc.body(mc.geom_bodyid[g]).name.startswith("arm/") for g in range(mc.ngeom)])
    mc.geom_margin[arm] = START_CLEAR_M
    mujoco.mj_fwdPosition(mc, dc)
    owner = {}
    for k, t in enumerate(trusses):
        for n in t["geoms"] + [f["geom"] for f in t["fruit"]]:
            owner[mc.geom(n).id] = k
    hit = set()
    for c in dc.contact[:dc.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if arm[a] and b in owner and c.dist < START_CLEAR_M:
                hit.add(owner[b])
    return hit


LAYOUT_TRIES = 4                 # layouts drawn per seed until one has a reachable truss to make ready. ⚠️ About a third
                                 # of single draws had none (12 of 30 dry-run seeds "no target", 2026-11-03)


def build(seed=0, arm=True):
    """Full truss scene. Ripeness is set after the layout: 1-2 of the trusses with a cut pose are ready,
    the others are almost-ready decoys or unripe (STAGE_P). Colours go on the model (scene.build does the same).
    The layout is drawn again (sub-seeds [seed, 1], [seed, 2], ...) while none of its trusses is reachable."""
    for j in range(LAYOUT_TRIES if arm else 1):
        out = _build_once(seed, arm, np.random.default_rng(seed if j == 0 else [seed, j]))
        if out[2]["ready"]:
            break
    out[2]["layout_try"] = j
    return out


def _build_once(seed, arm, rng):
    static, parts, trusses = vines_xml(rng)
    m, d = _compose(static, parts, arm)
    if arm:
        _start_pose(m, d)
        bad = _touching_arm(m, d, trusses)
        if bad:                                        # rebuild without them (rng draws below stay the same)
            parts = [p for k, p in enumerate(parts) if k not in bad]
            trusses = [t for k, t in enumerate(trusses) if k not in bad]
            m, d = _compose(static, parts, arm)
    ik = truss_ik.truss_ik()
    if arm:
        # ⚠️ Settle first, then judge reach where the trusses really hang. Cut poses sit at the reach edge: a 0.4 mm /
        # 1.3 deg settle took seed 200001's ready truss from 5 of 7 cut points with a pose to none (2026-11-03).
        _start_pose(m, d)
        for _ in range(int(BUILD_SETTLE_S / m.opt.timestep)):
            mujoco.mj_step(m, d)
    for t in trusses:
        # a cut pose at the middle of the cut zone AND near its ends, so a little sway doesn't take it away
        t["cut_ik"] = all(ik.solve(*cut_target_now(t, d, m, u), t["r"]) is not None for u in (0.3, 0.5, 0.7))
    # ⚠️ Crowded on purpose, but a ready truss must be reachable: some cut point with a clear slide-in and a
    # closing jaw that only touches its stalk (truss_expert.approach_ok, neighbours brushed at most). Without
    # this 6 of 12 dry runs failed on a neighbour in the way of every approach (2026-11-03).
    t_ok = []
    if arm:
        import truss_expert
        _start_pose(m, d)
        mujoco.mj_forward(m, d)                        # ⚠️ body poses (cut_target_now reads them) aren't set before
        check_rng = np.random.default_rng([seed, 0xA77])
        tmp = dict(trusses=trusses, fruit=[f for t in trusses for f in t["fruit"]])
        for k, t in enumerate(trusses):
            t["approach_ok"] = bool(t["cut_ik"] and truss_expert.TrussPlanner(m, d, tmp, k).approach_ok(check_rng))
    cand = [k for k, t in enumerate(trusses) if t["cut_ik"] and t.get("approach_ok", True)]
    del t_ok
    n_ready = min(len(cand), int(rng.integers(N_READY[0], N_READY[1] + 1)))
    ready = sorted(int(k) for k in rng.choice(cand, n_ready, replace=False)) if n_ready else []
    stages = list(STAGE_P)
    for k, t in enumerate(trusses):
        stage = "ready" if k in ready else str(rng.choice(stages, p=list(STAGE_P.values())))
        for _ in range(20):                            # the stage steers; is_ready decides
            rip = tr.draw_ripeness(t["n"], stage, rng)
            if tr.is_ready(rip) == (stage == "ready"):
                break
        t["stage"], t["ripeness"], t["ready"] = stage, rip, tr.is_ready(rip)
        for f, r in zip(t["fruit"], rip):
            f["ripeness"] = r
            m.geom_rgba[m.geom(f["geom"]).id] = tm.RIPENESS_RGBA[r]
    if arm:
        _start_pose(m, d)
    mujoco.mj_forward(m, d)
    fruit = [f for t in trusses for f in t["fruit"]]
    return m, d, dict(seed=seed, kind="truss", trusses=trusses, fruit=fruit,
                      ready=[k for k in ready if trusses[k]["ready"]], crate=(CRATE_CENTRE, CRATE_SIZE),
                      ripe=[], reachable=cand)


if __name__ == "__main__":       # layout stats over seeds
    import sys
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
    for s in seeds:
        m, d, info = build(s)
        T = info["trusses"]
        print(f"seed {s}: {len(T)} trusses, {sum(t['n'] for t in T)} fruit, cut pose on {len(info['reachable'])}, "
              f"ready {info['ready']}, stages {[t['stage'][0] for t in T]}", flush=True)
