"""Cherry-tomato trusses for the MuJoCo sim, with OrchardBench-style fruit detachment.

Each tomato is a free body held to the tip of its pedicel by a soft weld (the "tether"). The
weld's constraint force is read every step; once it stays above that tomato's break force for
DETACH_HOLD_S, the weld is switched off (d.eq_active = 0) and the tomato falls free. No model
edit, no recompile. Gripper pulls, pushes and gravity all show up in the same weld force, so a
contact-gripped pull detaches the fruit the same way a direct pull does.

Ported from OrchardBench (arXiv 2607.06337, apples on a one-sided spring tether, detach when the
pull exceeds f_detach for a few frames). They run Newton / MuJoCo-Warp; this is CPU MuJoCo, where
turning a weld off frees the body (checked by sim/check_tomato_release.py).

    truss_xml(...)  ->  (body_xml, equality_xml)  to paste into a scene
    Detacher(m, d)  ->  call .step() after every mj_step
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

# Weld softness = the tether spring. (timeconst, dampratio) in MuJoCo solref. 20 ms is MuJoCo's
# default: stiff enough that a hanging fruit moves < 1 mm. (unverified against a real pedicel)
TETHER_SOLREF = (0.02, 1.0)

# Ripeness colours, rough sRGB of the four classes the gate knows (guess from photos, unverified).
RIPENESS_RGBA = {
    "green":   (0.35, 0.55, 0.20, 1),
    "turning": (0.85, 0.55, 0.20, 1),
    "ripe":    (0.75, 0.08, 0.05, 1),
}


def truss_xml(name, rachis_from, rachis_to, n, rng, ripeness=None):
    """MJCF for one truss: a static rachis capsule with n tomatoes hanging off it.

    rachis_from/to: world points (m) of the truss stem. Tomatoes are spread evenly along it,
    alternating sides. ripeness: list of keys of RIPENESS_RGBA (len n), random if None.
    Returns (body_xml, equality_xml) and a list of per-tomato dicts (names, threshold, radius).
    """
    a, b = np.asarray(rachis_from, float), np.asarray(rachis_to, float)
    mid, axis = (a + b) / 2, b - a
    L = np.linalg.norm(axis)
    bodies = [f'<body name="{name}_rachis" pos="{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f}">'
              f'<geom type="capsule" fromto="{a[0]-mid[0]:.4f} {a[1]-mid[1]:.4f} {a[2]-mid[2]:.4f} '
              f'{b[0]-mid[0]:.4f} {b[1]-mid[1]:.4f} {b[2]-mid[2]:.4f}" size="0.002" '
              f'rgba="0.30 0.45 0.15 1" contype="0" conaffinity="0"/>']
    eqs, fruit = [], []
    for i in range(n):
        t = (i + 0.5) / n
        p = a + t * axis - mid                               # attach point in rachis frame
        r = rng.uniform(*RADIUS_M)
        lp = rng.uniform(*PEDICEL_LEN_M)
        side = 1 if i % 2 == 0 else -1
        # Pedicel leans out sideways, perpendicular to the rachis in the horizontal plane.
        out = np.cross(axis / L, [0, 0, 1])
        out = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-6 else np.array([1.0, 0, 0])
        tip = (side * 0.5 * out + np.array([0, 0, -0.866])) * lp   # 30 deg off vertical
        centre = tip + np.array([0, 0, -r])
        rip = ripeness[i] if ripeness else rng.choice(list(RIPENESS_RGBA))
        rgba = " ".join(f"{c:.2f}" for c in RIPENESS_RGBA[rip])
        tn, pn = f"{name}_t{i}", f"{name}_p{i}"
        # Pedicel: ball joint at the rachis, thin non-colliding capsule to the calyx.
        bodies.append(
            f'<body name="{pn}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}">'
            f'<joint name="{pn}_ball" type="ball" stiffness="{PEDICEL_STIFFNESS}" '
            f'damping="{PEDICEL_DAMPING}" armature="{PEDICEL_ARMATURE}"/>'
            f'<geom type="capsule" fromto="0 0 0 {tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}" size="0.0012" '
            f'mass="0.0005" rgba="0.30 0.45 0.15 1" contype="0" conaffinity="0"/>'
            f'<site name="{pn}_tip" pos="{tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}" size="0.001"/>'
            f'</body>')
        c = mid + p + centre                                  # world position of the fruit
        fruit.append(dict(tomato=tn, pedicel=pn, eq=f"{tn}_tether", radius=r, ripeness=rip,
                          threshold=rng.uniform(*DETACH_FORCE_N), pos=c))
        eqs.append(f'<weld name="{tn}_tether" body1="{tn}" body2="{pn}" '
                   f'solref="{TETHER_SOLREF[0]} {TETHER_SOLREF[1]}"/>')
    bodies.append("</body>")
    # Tomatoes are world children (free bodies), not children of the pedicel, so that switching
    # the weld off really frees them.
    for f in fruit:
        c = f["pos"]
        bodies.append(
            f'<body name="{f["tomato"]}" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"><freejoint/>'
            f'<geom name="{f["tomato"]}_g" type="sphere" size="{f["radius"]:.4f}" density="{DENSITY}" '
            f'rgba="{" ".join(f"{x:.2f}" for x in RIPENESS_RGBA[f["ripeness"]])}" '
            f'friction="0.8 0.01 0.001" condim="4"/></body>')
    return "\n".join(bodies), "\n".join(eqs), fruit


class Detacher:
    """Watches every tomato tether and switches it off once its force stays over threshold."""

    def __init__(self, m, d, fruit, hold_s=DETACH_HOLD_S):
        self.m, self.d = m, d
        self.eq_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f["eq"]) for f in fruit])
        self.thresholds = np.array([f["threshold"] for f in fruit])
        self.hold_steps = max(1, int(round(hold_s / m.opt.timestep)))
        self.over = np.zeros(len(fruit), int)
        self.detached_at = np.full(len(fruit), np.nan)   # sim time of release
        self.force_at = np.full(len(fruit), np.nan)      # tether force at release (N)
        self.force = np.zeros(len(fruit))                # last measured tether force (N)

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
        self.over = np.where(live & (self.force > self.thresholds), self.over + 1, 0)
        out = np.flatnonzero(self.over >= self.hold_steps)
        for k in out:
            self.d.eq_active[self.eq_ids[k]] = 0
            self.detached_at[k] = self.d.time
            self.force_at[k] = self.force[k]
            self.over[k] = 0
        return out
