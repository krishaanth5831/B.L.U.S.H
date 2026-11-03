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
- the arm has the rigid cut-and-hold fingers (scene.add_cut_hold);
- the crate is bigger: a truss is ~10-15 cm long.
"""
import mujoco
import numpy as np

import scene
import tomato as tm
import truss as tr
import truss_ik

# --- layout (design choices, K7 2026-11-03 "crowded"; ranges unverified against a real greenhouse) ---
STEM_X = (0.22, 0.52)            # along the row, in front of the arm (scene.STEM_X was 0.26-0.48)
N_STEMS = (3, 4)
STEM_GAP = 0.06                  # m between stems (scene.STEM_GAP 0.08): trusses of neighbouring stems overlap
TRUSSES_PER_STEM = (1, 3)
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
# Crate for whole trusses: ~10-15 cm long trusses need ~20 cm. Same spot as scene.CRATE_CENTRE, a little
# toward the arm (guess, unverified: the planner checks IK above it).
CRATE_CENTRE = (0.56, 0.17)
CRATE_SIZE = (0.20, 0.16, 0.05)


def in_crate(info, d, m, k):
    """Every fruit of truss k inside the crate walls, below rim + 1 cm (a truss lies on its fruit)."""
    (cx, cy), (sx, sy, h) = CRATE_CENTRE, CRATE_SIZE
    for f in info["trusses"][k]["fruit"]:
        p = d.xpos[m.body(f["tomato"]).id]
        if not (abs(p[0] - cx) < sx / 2 and abs(p[1] - cy) < sy / 2 and p[2] < h + 0.01):
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
                t["stem"] = i
                knuckles.append(kn); trusses_xml.append(body); bodies.append(fb); eqs.append(eq); excl.append(ex)
                trusses.append(t)
                placed += [(f["pos"], f["radius"]) for f in t["fruit"]] + _stalk_spheres(t)
                break
    static_xml = "\n".join(static + knuckles + trusses_xml)
    return static_xml, "\n".join(bodies), "\n".join(eqs), "\n".join(excl), trusses


def build(seed=0, arm=True):
    """Full truss scene. Ripeness is set after the layout: 1-2 of the trusses with a cut pose are ready,
    the others are almost-ready decoys or unripe (STAGE_P). Colours go on the model (scene.build does the same)."""
    rng = np.random.default_rng(seed)
    static, bodies, eqs, excl, trusses = vines_xml(rng)
    m, d = scene.compose(static, bodies, eqs, arm, cut_hold=True, excludes=excl, crate=(CRATE_CENTRE, CRATE_SIZE))
    ik = truss_ik.truss_ik()
    for t in trusses:
        p, s = cut_target(t)
        t["cut_ik"] = ik.solve(p, s, t["r"]) is not None
    cand = [k for k, t in enumerate(trusses) if t["cut_ik"]]
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
        import arm_map
        q, u = scene.arm_ids(m)
        d.qpos[q] = arm_map.FollowerMap().norm_to_rad(scene.START_NORM)
        d.ctrl[u] = d.qpos[q]
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
