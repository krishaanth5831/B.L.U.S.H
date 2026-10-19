"""Cherry-tomato trusses for the MuJoCo sim, with OrchardBench-style fruit detachment.

Each tomato is a free body held to the tip of its pedicel by a soft weld (the "tether"). The
weld's constraint force is read every step; once it stays above that tomato's break force for
DETACH_HOLD_S, the weld is switched off (d.eq_active = 0) and the tomato falls free. No model
edit, no recompile. Gripper pulls, pushes and gravity all show up in the same weld force, so a
contact-gripped pull detaches the fruit the same way a direct pull does.

Ported from OrchardBench (arXiv 2607.06337, apples on a one-sided spring tether, detach when the
pull exceeds f_detach for a few frames). They run Newton / MuJoCo-Warp; this is CPU MuJoCo, where
turning a weld off frees the body (checked by sim/check_tomato_release.py).

    truss_xml(...)   -> one truss (rachis + n tomatoes), spaced so no two fruits overlap
    single_xml(...)  -> one tomato hanging at an exact point (grasp tests)
    Detacher(m, d)   -> call .step() after every mj_step
"""
import mujoco
import numpy as np

# Calyx-joint detachment force of cherry tomatoes, 0.58-2.46 N (Agriculture 2021, 11(9):815,
# abstract only). Randomised per tomato. ⚠️ unverified for our supermarket vine tomatoes: pull a
# few off with a kitchen/luggage scale and replace this range.
DETACH_FORCE_N = (0.58, 2.46)

# How long the force must stay over the threshold. OrchardBench says "a few frames"; 20 ms
# (10 steps at 2 ms) is a guess (unverified). It stops a single contact spike from plucking.
DETACH_HOLD_S = 0.02

# Cherry tomato diameter 20-30 mm (guess from supermarket cherry tomatoes, unverified).
RADIUS_M = (0.010, 0.015)
# Water-like density gives 4-14 g for that diameter range; real cherry tomatoes are ~10-20 g
# (guess, unverified), so density is set a little above water.
DENSITY = 1050.0

# Pedicel from the truss stem (rachis) to the calyx, ~10-15 mm (guess, unverified).
PEDICEL_LEN_M = (0.010, 0.015)
# Ball joint at the rachis lets the fruit swing like a real one. Stiffness/damping are guesses
# picked so a 10 g fruit sags a few degrees and stops swinging within ~1 s (unverified).
PEDICEL_STIFFNESS = 0.004     # N.m/rad
PEDICEL_DAMPING = 2e-4        # N.m.s/rad
PEDICEL_ARMATURE = 1e-6       # ⚠️ without it the near-massless pedicel body makes the ball joint jitter
PEDICEL_LEAN_DEG = 30.0       # pedicels lean out from the rachis, alternating sides (guess from photos)

# The truss stem (rachis) hangs off the vine on a sprung ball joint, so a truss sways when the
# gripper drags a fruit. Rigid, it pushed all of a closing jaw's ~8 mm drag into the tether and
# plucked the weakest fruit (check_scene.py). 0.3 N.m/rad = ~10 deg of sag for a 10-fruit truss on an
# 8 cm rachis: guess, unverified (push a real truss with a scale and measure the sway).
RACHIS_STIFFNESS = 0.3        # N.m/rad
RACHIS_DAMPING = 0.01         # N.m.s/rad, guess: settles in ~1 s
RACHIS_ARMATURE = 1e-4        # ⚠️ the 2 mm rachis capsule is nearly massless; without this the joint jitters
RACHIS_LEVER_M = 0.08         # single_xml's stand-in truss length

# Weld softness = the tether spring. (timeconst, dampratio) in MuJoCo solref. 20 ms is MuJoCo's
# default: stiff enough that a hanging fruit moves < 1 mm. (unverified against a real pedicel)
TETHER_SOLREF = (0.02, 1.0)

# Gap kept between neighbouring fruits when they're laid out (guess; real trusses touch sometimes).
FRUIT_GAP_M = 0.002

# Ripeness colours, rough sRGB of the classes the gate knows (guess from photos, unverified).
RIPENESS_RGBA = {
    "green":   (0.35, 0.55, 0.20, 1),
    "turning": (0.85, 0.55, 0.20, 1),
    "ripe":    (0.75, 0.08, 0.05, 1),
}
STEM_RGBA = "0.30 0.45 0.15 1"


def _rgba(rip):
    return " ".join(f"{c:.2f}" for c in RIPENESS_RGBA[rip])


def _fruit(name, local, world, tip, r, rip, threshold):
    """Pedicel body (local pos in its parent), tomato body (world), weld, info dict."""
    tn, pn = f"{name}", f"{name}_ped"
    ped = (f'<body name="{pn}" pos="{local[0]:.5f} {local[1]:.5f} {local[2]:.5f}">'
           f'<joint name="{pn}_ball" type="ball" stiffness="{PEDICEL_STIFFNESS}" '
           f'damping="{PEDICEL_DAMPING}" armature="{PEDICEL_ARMATURE}"/>'
           f'<geom type="capsule" fromto="0 0 0 {tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}" size="0.0012" '
           f'mass="0.0005" rgba="{STEM_RGBA}" contype="0" conaffinity="0"/></body>')
    c = world + tip + np.array([0, 0, -r])
    # Tomatoes are world children (free bodies), not children of the pedicel, so that switching
    # the weld off really frees them.
    body = (f'<body name="{tn}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}"><freejoint/>'
            f'<geom name="{tn}_g" type="sphere" size="{r:.5f}" density="{DENSITY}" rgba="{_rgba(rip)}" '
            f'friction="0.8 0.01 0.001" condim="4"/></body>')
    eq = (f'<weld name="{tn}_tether" body1="{tn}" body2="{pn}" '
          f'solref="{TETHER_SOLREF[0]} {TETHER_SOLREF[1]}"/>')
    return ped, body, eq, dict(tomato=tn, pedicel=pn, eq=f"{tn}_tether", geom=f"{tn}_g", radius=r,
                               ripeness=rip, threshold=threshold, pos=c)


def truss_xml(name, rachis_from, rachis_to, n, rng, ripeness=None, avoid=()):
    """MJCF for one truss: a static rachis capsule with up to n tomatoes hanging off it.

    rachis_from/to: world points (m) of the truss stem. ripeness: list of RIPENESS_RGBA keys (len
    n), random if None. avoid: (centre, radius) of fruits already placed elsewhere. A fruit that
    would overlap a neighbour gets a smaller radius, then is dropped, so a truss can come out
    with fewer than n tomatoes.
    Returns (rachis_body_xml, tomato_bodies_xml, equality_xml, fruit_info_list).
    """
    a, b = np.asarray(rachis_from, float), np.asarray(rachis_to, float)
    mid, axis = (a + b) / 2, b - a
    L = np.linalg.norm(axis)
    out = np.cross(axis / L, [0, 0, 1])
    out = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-6 else np.array([1.0, 0, 0])
    lean = np.radians(PEDICEL_LEAN_DEG)
    placed = list(avoid)
    peds, bodies, eqs, fruit = [], [], [], []
    for i in range(n):
        p_world = a + (i + 0.5) / n * axis
        side = 1 if i % 2 == 0 else -1
        lp = rng.uniform(*PEDICEL_LEN_M)
        tip = (side * np.sin(lean) * out + np.array([0, 0, -np.cos(lean)])) * lp
        rip = ripeness[i] if ripeness else str(rng.choice(list(RIPENESS_RGBA)))
        r = rng.uniform(*RADIUS_M)
        for _ in range(4):
            c = p_world + tip + np.array([0, 0, -r])
            if all(np.linalg.norm(c - c2) >= r + r2 + FRUIT_GAP_M for c2, r2 in placed):
                break
            r *= 0.85
        else:
            continue
        if r < RADIUS_M[0] * 0.8:
            continue
        ped, body, eq, info = _fruit(f"{name}_t{i}", p_world - mid, p_world, tip, r, rip,
                                     rng.uniform(*DETACH_FORCE_N))
        placed.append((info["pos"], r))
        peds.append(ped); bodies.append(body); eqs.append(eq); fruit.append(info)
    rachis = (f'<body name="{name}_rachis" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}">'
              f'<joint name="{name}_rachis_ball" type="ball" pos="{-axis[0]/2:.5f} {-axis[1]/2:.5f} {-axis[2]/2:.5f}" '
              f'stiffness="{RACHIS_STIFFNESS}" damping="{RACHIS_DAMPING}" armature="{RACHIS_ARMATURE}"/>'
              f'<geom type="capsule" fromto="{-axis[0]/2:.5f} {-axis[1]/2:.5f} {-axis[2]/2:.5f} '
              f'{axis[0]/2:.5f} {axis[1]/2:.5f} {axis[2]/2:.5f}" size="0.002" '
              f'rgba="{STEM_RGBA}" contype="0" conaffinity="0"/>' + "".join(peds) + "</body>")
    return rachis, "\n".join(bodies), "\n".join(eqs), fruit


def single_xml(name, centre, r, threshold, ripeness="ripe", pedicel_len=0.012):
    """One tomato whose centre sits exactly at `centre`, hanging straight down from the end of a
    stand-in truss: an anchor on a sprung ball joint RACHIS_LEVER_M away along +x (same compliance
    as a real truss in truss_xml)."""
    centre = np.asarray(centre, float)
    top = centre + np.array([0, 0, r + pedicel_len])
    ped, body, eq, info = _fruit(name, np.zeros(3), top, np.array([0, 0, -pedicel_len]), r, ripeness,
                                 threshold)
    anchor = (f'<body name="{name}_anchor" pos="{top[0]:.5f} {top[1]:.5f} {top[2]:.5f}">'
              f'<joint name="{name}_anchor_ball" type="ball" pos="{RACHIS_LEVER_M} 0 0" '
              f'stiffness="{RACHIS_STIFFNESS}" damping="{RACHIS_DAMPING}" armature="{RACHIS_ARMATURE}"/>'
              f'<geom type="capsule" fromto="0 0 0 {RACHIS_LEVER_M} 0 0" size="0.002" mass="0.002" '
              f'rgba="{STEM_RGBA}" contype="0" conaffinity="0"/>'
              f'{ped}</body>')
    return anchor, body, eq, info


class Detacher:
    """Watches every tomato tether and switches it off once its force stays over threshold."""

    def __init__(self, m, d, fruit, hold_s=DETACH_HOLD_S):
        self.m, self.d = m, d
        self.eq_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f["eq"]) for f in fruit], int)
        self.thresholds = np.array([f["threshold"] for f in fruit])
        self.hold_steps = max(1, int(round(hold_s / m.opt.timestep)))
        self.over = np.zeros(len(fruit), int)
        self.detached_at = np.full(len(fruit), np.nan)   # sim time of release
        self.force_at = np.full(len(fruit), np.nan)      # tether force at release (N)
        self.force = np.zeros(len(fruit))                # last measured tether force (N)
        self.peak = np.zeros(len(fruit))                 # highest force seen while attached (N)

    def tether_forces(self):
        """Translational weld force (N) per tomato, 0 for released ones."""
        d = self.d
        f = np.zeros(len(self.eq_ids))
        eq_rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        for k, e in enumerate(self.eq_ids):
            if not d.eq_active[e]:
                continue
            rows = np.flatnonzero(eq_rows & (d.efc_id == e))
            # A weld has 6 rows: 3 translational (N), then 3 rotational (N.m). Only the pull counts.
            f[k] = np.linalg.norm(d.efc_force[rows[:3]])
        return f

    def step(self):
        """Call after every mj_step. Returns indices of tomatoes released this step."""
        self.force = self.tether_forces()
        live = self.d.eq_active[self.eq_ids].astype(bool)
        self.peak = np.where(live, np.maximum(self.peak, self.force), self.peak)
        self.over = np.where(live & (self.force > self.thresholds), self.over + 1, 0)
        out = np.flatnonzero(self.over >= self.hold_steps)
        for k in out:
            self.d.eq_active[self.eq_ids[k]] = 0
            self.detached_at[k] = self.d.time
            self.force_at[k] = self.force[k]
            self.over[k] = 0
        return out
