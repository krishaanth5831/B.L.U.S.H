"""Whole cherry-tomato trusses that the arm harvests by cutting the stalk (K7, 2026-11-03).

    knuckle, truss_body, fruit_bodies, eqs, excludes, info = truss_xml(name, stem_point, direction, rng, stage)
    is_ready(ripeness)              the harvest rule: tip fruit coloured, most of the rest ripe
    TrussCutter(m, d, info)         after every mj_step: cuts the stalk when both blades press it hard enough

How a truss hangs. The stalk (peduncle) leaves the vine stem at the knuckle and runs PEDUNCLE_LEN out to the
first fruit; the rachis carries the fruit from there, alternating sides (tomato.py's pedicels and tethers,
reused). The whole stalk is ONE free body (peduncle + rachis capsules, colliding, so the jaw can grip it and
the blades can press it), held to a sprung knuckle on the stem by a weld: the "cut weld". Cutting = switching
that weld off (d.eq_active), the same no-recompile trick as tomato.Detacher. The fruit stay welded to their
pedicels on the truss, so a fruit squeezed or knocked off the target truss still drops (tomato.Detacher).

Cut-and-hold (K7 2026-11-03: a blade on the fin-ray fingertips, one close cuts and holds; SHEAR's rule "cut,
never snap"). The fingertips carry a blade pair on their stem-side face and spring-loaded clamp pads on the
rest of their width (scene.add_cut_hold). TrussCutter watches the two blades: when both press the target
stalk with CUT_FORCE for CUT_DWELL_S, the weld goes off and the blades stop colliding (they have gone through
the stalk); the pads keep squeezing it, so the truss stays in the jaw.

Ripening runs from the stem end (proximal) to the tip (distal), as on a real truss.
"""
import mujoco
import numpy as np

import tomato as tm

# Fruit per truss. K7 (2026-11-03): 6-8.
FRUITS = (6, 8)
# Stalk radius, stem to first fruit and along the rachis. A cherry-tomato peduncle is ~3-5 mm across (guess
# from supermarket vines, unverified: measure a few with calipers).
PEDUNCLE_R = (0.0015, 0.0025)
RACHIS_R_FRACTION = 0.8          # rachis a little thinner than the peduncle (guess, unverified)
# Knuckle to the first fruit along the stalk (guess from photos, unverified).
PEDUNCLE_LEN = (0.045, 0.060)
# Where a cut belongs: this far out along the peduncle from the stem surface. Growers cut just past the
# knuckle; closer than 10 mm and the blade would rub the main stem (guess, unverified). ⚠️ The far end is set
# by the jaw, not the plant: the clamp pads span the 18 mm finger width on the truss side of the blade, and
# they only hold where the stalk runs straight under them (check_truss_cut.py: cut 28 mm out on a 35 mm
# peduncle, the rachis bent away under the pads and the cut truss fell, 2026-11-03). So cut + 18 mm finger +
# 5 mm margin stays inside the shortest peduncle.
CUT_ZONE_M = (0.010, 0.020)
# The rachis turns from the peduncle by this much, + = further down (photos, unverified). With the steep
# peduncles truss_scene needs (40-70 deg, for a cut pose), bending further down hung every truss as a vertical
# string of fruit (render, 2026-11-03), so it mostly levels off a little instead.
RACHIS_BEND_DEG = (-30.0, 5.0)
# Fruit along the rachis: each next pedicel root this far past the last (alternating sides, so consecutive
# fruit sit on opposite sides). Grown until no two fruit overlap (FRUIT_GAP_M); the rachis ends past the last.
STEP_M = (0.010, 0.016)
RACHIS_TAIL_M = 0.008
STALK_DENSITY = 900.0            # green stalk, a little below water (guess, unverified)
STALK_FRICTION = (0.9, 0.01, 0.001)   # guess (unverified), like the fruit's
# The cut weld: stiff, so a hanging truss doesn't sag on it (the knuckle's ball spring does the swaying).
# ⚠️ With MuJoCo's default solimp and a 2 g knuckle the weld let the truss twist 11-14 deg off its knuckle
# while settling (2026-11-03): a soft weld's stiffness scales with the lighter body. Tight solimp + a 20 g
# knuckle (the stem section the truss hangs on, guess) hold it.
CUT_WELD_SOLREF = (0.004, 1.0)   # timeconst = 2 x the 2 ms step (MuJoCo's stability floor)
CUT_WELD_SOLIMP = (0.99, 0.999, 0.0001, 0.5, 2)   # as scene.FINRAY_LOOP_SOLIMP
KNUCKLE_MASS = 0.02
# Knuckle spring (guess, unverified: push a real truss with a scale and measure). ⚠️ At tomato.RACHIS_STIFFNESS's
# 0.3 N.m/rad a 6-8 fruit truss sagged 20-23 deg while the scene settled (2026-11-03), turning its stalk out of
# the narrow band where a cut pose exists (truss_ik scan) after the scene had checked it. 3 N.m/rad: ~2 deg.
KNUCKLE_STIFFNESS = 3.0          # N.m/rad
KNUCKLE_DAMPING = 0.05           # N.m.s/rad, guess: settles in well under 1 s at the stiffer spring
KNUCKLE_ARMATURE = 1e-4          # ⚠️ the near-massless knuckle body jitters without it

# --- the harvest rule (K7 2026-11-03: "tip fruit coloured, most of the rest ripe") ---
READY_TIP = ("turning", "ripe")  # the last fruit to ripen is at least turning
READY_RIPE_SHARE = 2 / 3         # "most of the rest": at least this share of the other fruit ripe (guess)

# --- stages: where the ripening front is, as a fraction of the truss from stem (0) to tip (1) ---
# Fruit before the front - TURNING_BAND is ripe, within +-TURNING_BAND turning, past it green; each fruit's
# place is jittered by FRONT_JITTER so two trusses at one stage differ. Ranges are design choices: "ready"
# always passes is_ready, "almost" is the hard decoy (most of it red, the tip still green), "unripe" is
# mostly green. Every drawn truss is checked against is_ready afterwards; the stage only steers the draw.
TURNING_BAND = 0.18
FRONT_JITTER = 0.08
STAGES = {"ready": (1.05, 1.5), "almost": (0.55, 0.85), "unripe": (-0.3, 0.35)}

# --- cutting ---
# Cut force: SHEAR's placeholder for leaf petioles, 2.1 N per mm of stalk diameter (plant.CUT_FORCE_PER_MM:
# 15 N for 7 mm, inside the literature's 0.2-5.5 N/mm bracket). ⚠️ unverified for tomato peduncles: cut a
# few with the real blade on a kitchen scale and replace this.
CUT_FORCE_PER_MM = 15.0 / 7.0
# A stroke takes time: the force must hold this long (SHEAR plant.CUT_DWELL_S, unverified). It also keeps the
# blade's landing spike from counting as a cut.
CUT_DWELL_S = 0.05
CAPTURE_N = 0.2                  # the fixed jaw (blade or pad) must press the stalk this much for it to count as held


def cut_force_for(r):
    return CUT_FORCE_PER_MM * 2 * r * 1000


def is_ready(ripeness):
    """The harvest rule on a truss's fruit, stem end first (ripeness[-1] is the tip)."""
    if len(ripeness) < FRUITS[0] or ripeness[-1] not in READY_TIP:
        return False
    rest = ripeness[:-1]
    return sum(r == "ripe" for r in rest) >= READY_RIPE_SHARE * len(rest)


def draw_ripeness(n, stage, rng):
    front = rng.uniform(*STAGES[stage])
    out = []
    for i in range(n):
        u = i / (n - 1) + rng.uniform(-FRONT_JITTER, FRONT_JITTER)
        out.append("ripe" if u < front - TURNING_BAND else "turning" if u < front + TURNING_BAND else "green")
    return out


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot_towards_down(d, deg):
    """d turned down (toward -z) by deg, in the vertical plane through d."""
    d = _unit(d)
    down = np.array([0, 0, -1.0]) - d * (d @ [0, 0, -1.0])     # straight down, minus its part along d
    if np.linalg.norm(down) < 1e-6:
        return d
    t = np.radians(deg)
    return _unit(d * np.cos(t) + _unit(down) * np.sin(t))


def truss_xml(name, stem_point, direction, rng, stage, stem_r=0.004, avoid=(), n=None, peduncle_len=None):
    """One truss leaving the stem surface at stem_point (world, m) along `direction` (unit-ish, world).

    stage: a STAGES key. avoid: (centre, radius) of fruit already placed elsewhere. Returns
    (knuckle_xml, truss_xml, fruit_bodies_xml, eq_xml, exclude_xml, info) or None when the fruit don't fit without
    overlapping something (the caller draws another direction).
    info: name, ripeness list (stem end first), ready, fruit (tomato.py info dicts, with "truss"), the stalk
    geometry (world): knuckle, peduncle_end, rachis_end, r, cut zone endpoints, stalk geom names, weld name.
    """
    n = int(rng.integers(FRUITS[0], FRUITS[1] + 1)) if n is None else n
    k0 = np.asarray(stem_point, float)
    d0 = _unit(direction)
    r_ped = rng.uniform(*PEDUNCLE_R)
    p1 = k0 + d0 * (rng.uniform(*PEDUNCLE_LEN) if peduncle_len is None else peduncle_len)
    d1 = _rot_towards_down(d0, rng.uniform(*RACHIS_BEND_DEG))
    out = np.cross(d1, [0, 0, 1.0])
    out = _unit(out) if np.linalg.norm(out) > 1e-6 else np.array([1.0, 0, 0])
    rip = draw_ripeness(n, stage, rng)
    placed = list(avoid)
    fruit_specs, s = [], 0.0
    for i in range(n):
        side = 1 if i % 2 == 0 else -1
        r = rng.uniform(*tm.RADIUS_M)
        lp = rng.uniform(*tm.PEDICEL_LEN_M)
        lean = np.radians(rng.uniform(*tm.PEDICEL_LEAN_DEG))
        sw = np.radians(rng.uniform(-tm.PEDICEL_SWING_DEG, tm.PEDICEL_SWING_DEG))
        o = side * (np.cos(sw) * out + np.sin(sw) * np.cross([0, 0, 1.0], out))
        tip = (np.sin(lean) * o + np.array([0, 0, -np.cos(lean)])) * lp
        tilt, az = np.radians(rng.uniform(*tm.HANG_TILT_DEG)), rng.uniform(0, 2 * np.pi)
        hang = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
        s += rng.uniform(*STEP_M) if i else 0.0
        for _ in range(12):                      # slide along the rachis until it fits
            root = p1 + d1 * s
            c = root + tip + hang * r
            if all(np.linalg.norm(c - c2) >= r + r2 + tm.FRUIT_GAP_M for c2, r2 in placed):
                break
            s += 0.004
        else:
            return None
        placed.append((c, r))
        fruit_specs.append((root, tip, hang, r, rng.uniform(0, 2 * np.pi)))
    p2 = p1 + d1 * (s + RACHIS_TAIL_M)
    # Pedicels hang off the truss body, whose frame sits at the knuckle (world-aligned).
    peds, bodies, eqs, fruit = [], [], [], []
    for i, (root, tip, hang, r, spin) in enumerate(fruit_specs):
        ped, body, eq, info = tm._fruit(f"{name}_t{i}", root - k0, root, tip, r, rip[i],
                                        rng.uniform(*tm.DETACH_FORCE_N), hang=hang, spin=spin)
        info["truss"] = name
        peds.append(ped); bodies.append(body); eqs.append(eq); fruit.append(info)
    f = lambda v: f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"   # noqa: E731
    a, b = p1 - k0, p2 - k0
    col = f'friction="{STALK_FRICTION[0]} {STALK_FRICTION[1]} {STALK_FRICTION[2]}" condim="4"'
    knuckle = (f'<body name="{name}_knuckle" pos="{f(k0)}">'
               f'<joint name="{name}_knuckle_ball" type="ball" stiffness="{KNUCKLE_STIFFNESS}" '
               f'damping="{KNUCKLE_DAMPING}" armature="{KNUCKLE_ARMATURE}"/>'
               f'<geom type="sphere" size="{stem_r:.4f}" mass="{KNUCKLE_MASS}" rgba="{tm.STEM_RGBA}" contype="0" '
               f'conaffinity="0"/></body>')
    truss = (f'<body name="{name}_truss" pos="{f(k0)}"><freejoint/>'
             f'<geom name="{name}_peduncle" type="capsule" fromto="0 0 0 {f(a)}" size="{r_ped:.5f}" '
             f'density="{STALK_DENSITY}" rgba="{tm.STEM_RGBA}" {col}/>'
             f'<geom name="{name}_rachis" type="capsule" fromto="{f(a)} {f(b)}" '
             f'size="{r_ped * RACHIS_R_FRACTION:.5f}" density="{STALK_DENSITY}" rgba="{tm.STEM_RGBA}" {col}/>'
             + "".join(peds) + "</body>")
    eq_xml = (f'<weld name="{name}_cut" body1="{name}_truss" body2="{name}_knuckle" '
              f'solref="{CUT_WELD_SOLREF[0]} {CUT_WELD_SOLREF[1]}" '
              f'solimp="{" ".join(str(v) for v in CUT_WELD_SOLIMP)}"/>' + "".join(eqs))
    # ⚠️ The truss's own fruit sit a pedicel away from its stalk; their contacts with it are excluded (they're
    # one plant, the tether holds them) so a fruit pressed onto its rachis doesn't fight its own weld.
    excl = "".join(f'<exclude body1="{name}_truss" body2="{fi["tomato"]}"/>' for fi in fruit)
    cz = [k0 + d0 * (stem_r + z) for z in CUT_ZONE_M]
    info = dict(name=name, ripeness=rip, ready=is_ready(rip), stage=stage, fruit=fruit, n=n,
                knuckle=k0, peduncle_end=p1, rachis_end=p2, r=r_ped, dir=d0, cut_zone=cz,
                geoms=[f"{name}_peduncle", f"{name}_rachis"], weld=f"{name}_cut", body=f"{name}_truss")
    return knuckle, truss, "\n".join(bodies), eq_xml, excl, info


class TrussCutter:
    """Cuts one truss's stalk when the moving blade presses it with its cut force for CUT_DWELL_S, while the
    stalk is captured against the fixed jaw (its blade or pad touching it).

    ⚠️ Bypass blades, not SHEAR's edge-on-edge pair: the moving blade shears the stalk past the fixed one, and the
    reaction comes from the fixed jaw AND the stalk's own support. With the stiffened knuckle the moving blade
    pressed 18.5 N while the fixed blade felt 2.1 N at the servo's full torque (2026-11-03), so SHEAR's rule
    (the weaker blade) never cut. squeeze = the moving blade's normal force once captured.
    blades: (fixed, moving) blade geom names. Per step it keeps squeeze, pad force (what the clamp pads press
    the stalk with, the weaker side) and their peaks.
    ⚠️ It edits the model when it cuts (the blades stop colliding); a scene rebuild puts it back."""

    def __init__(self, m, d, info, blades=("arm/blade_fixed", "arm/blade_moving"),
                 pads=("arm/pad_fixed", "arm/pad_moving")):
        self.m, self.d, self.info = m, d, info
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)   # noqa: E731
        self.stalk = {gid(g) for g in info["geoms"]}
        self.blades = tuple(gid(b) for b in blades)
        self.pads = tuple(gid(p) for p in pads)
        self.weld = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, info["weld"])
        self.cut_force = cut_force_for(info["r"])
        self._bits = {b: (m.geom_contype[b], m.geom_conaffinity[b]) for b in self.blades}
        self._f6 = np.zeros(6)
        self.cut_t, self._over_t = None, None
        self.squeeze = self.pad_force = 0.0
        self.peak = dict(squeeze=0.0, pad_force=0.0)
        self.cut_point = None                 # where the blades crossed the stalk (world), at the cut

    @property
    def attached(self):
        return bool(self.d.eq_active[self.weld])

    def step(self):
        m, d = self.m, self.d
        f = {g: 0.0 for g in self.blades + self.pads}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.stalk:
                other = g2
            elif g2 in self.stalk:
                other = g1
            else:
                continue
            if other in f:
                mujoco.mj_contactForce(m, d, i, self._f6)
                f[other] += self._f6[0]
        fixed_b, moving_b = self.blades
        captured = f[fixed_b] + f[self.pads[0]] > CAPTURE_N
        self.squeeze = f[moving_b] if captured else 0.0
        self.pad_force = min(f[p] for p in self.pads)
        for k in self.peak:
            self.peak[k] = max(self.peak[k], getattr(self, k))
        if not self.attached:
            return False
        if self.squeeze < self.cut_force:
            self._over_t = None
            return False
        self._over_t = d.time if self._over_t is None else self._over_t
        if d.time - self._over_t < CUT_DWELL_S - 1e-9:
            return False
        d.eq_active[self.weld] = 0
        self.cut_t = d.time
        self.cut_point = np.mean([d.geom_xpos[b] for b in self.blades], axis=0)
        for b in self.blades:                 # through the stalk: they don't push it any more
            m.geom_contype[b], m.geom_conaffinity[b] = 0, 0
        return True

    def restore(self):
        for b, (ct, ca) in self._bits.items():
            self.m.geom_contype[b], self.m.geom_conaffinity[b] = ct, ca
