"""Domain randomisation for one scene: lighting, colours, camera poses, and the physics guesses.

    params = apply(m, seed)       # after scene.build(seed), before the first frame

K7 (2026-10-27) asked for it with the scripted episodes, so the data isn't all one look and one feel.
Everything is drawn from the scene seed (its own stream, so the vine layout stays the same with or
without it) and returned as a dict, which the plugin writes into session.json: a re-render (step 6)
that rebuilds the scene from the seed and calls apply() again gets exactly the same scene.

What is randomised and why:
- Lights + table/greenhouse colours: the real room's lighting and the photo textures aren't known yet.
- Fruit colour, jittered INSIDE its ripeness class (hue/saturation/brightness), so a ripe fruit stays
  red and a green one never drifts toward ripe: the gate's classes must keep meaning the same thing.
- Camera poses: the birdseye mast (behind row R0, facing the row since 2026-10-28) isn't built yet and the claw
  camera sits on a 3D-printed mount, so both get a small pose jitter.
- Physics guesses that change the grasp: finger friction and stiffness, truss and pedicel stiffness,
  fruit mass. ⚠️ Not fruit friction: the finger geoms have contact priority, so their friction is the
  one MuJoCo uses on a finger-fruit contact. Break force is already drawn per fruit (tomato.DETACH_FORCE_N).
"""
import colorsys

import mujoco
import numpy as np

import tomato as tm

# All ranges are guesses (unverified) picked around the current nominal values; each says what it spans.
LIGHT_SCALE = (0.6, 1.4)          # x the nominal diffuse of each light
LIGHT_DIR_DEG = 15.0              # +- tilt of each light's direction
AMBIENT = (0.20, 0.40)            # headlight ambient (nominal 0.3)
SURFACE_VAL = (0.85, 1.15)        # x brightness of the table, ply, crate
# Per ripeness class: (hue +-, saturation x, value x). Hue in 0..1 turns; 0.02 = ~7 deg, which keeps
# ripe red-to-orange-red and green green.
FRUIT_HSV = {"ripe": (0.015, (0.85, 1.10), (0.80, 1.15)),
             "turning": (0.020, (0.85, 1.10), (0.85, 1.15)),
             "green": (0.025, (0.80, 1.15), (0.80, 1.20))}
BIRDSEYE_POS_M, BIRDSEYE_ROT_DEG = 0.010, 2.0   # mast placement by hand: +-1 cm, +-2 deg
CLAW_POS_M, CLAW_ROT_DEG = 0.002, 1.5           # printed wrist mount
FINGER_FRICTION = (0.9, 1.4)      # sliding; nominal 1.2 (scene.FINGER_FRICTION, a guess)
FINGER_STIFFNESS = (0.7, 1.4)     # x the fin-ray hinge springs; nominal = 2 N/mm at mid-finger (a guess)
TRUSS_STIFFNESS = (0.6, 1.6)      # x rachis + pedicel ball-joint springs (guesses in tomato.py)
FRUIT_MASS = (0.85, 1.25)         # x fruit mass (density guess in tomato.py)


def _rot(rng, max_deg):
    """Small random rotation as a quaternion (w, x, y, z)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-max_deg, max_deg))
    return np.r_[np.cos(a / 2), np.sin(a / 2) * axis]


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _hsv_jitter(rgb, rng, dh, s_rng, v_rng):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + rng.uniform(-dh, dh)) % 1.0
    return np.clip(colorsys.hsv_to_rgb(h, min(1, s * rng.uniform(*s_rng)), min(1, v * rng.uniform(*v_rng))), 0, 1)


def apply(m, seed, fruit=None):
    """Randomise model m in place for scene `seed`; fruit = scene info["fruit"]. Returns the drawn params."""
    rng = np.random.default_rng([int(seed), 0xD12])     # own stream: the layout draws are untouched
    p = {}
    # --- lights ---
    for i in range(m.nlight):
        k = float(rng.uniform(*LIGHT_SCALE))
        m.light_diffuse[i] *= k
        q = _rot(rng, LIGHT_DIR_DEG)
        dvec = np.zeros(3)
        mujoco.mju_rotVecQuat(dvec, m.light_dir[i].copy(), q)
        m.light_dir[i] = dvec
        p[f"light_{m.light(i).name}"] = {"diffuse_x": round(k, 3), "dir": np.round(dvec, 3).tolist()}
    amb = float(rng.uniform(*AMBIENT))
    m.vis.headlight.ambient[:] = amb
    p["ambient"] = round(amb, 3)
    # --- surfaces ---
    kv = {}
    for i in range(m.ngeom):
        n = m.geom(i).name
        group = ("table" if n == "table" else "crate" if n.startswith("crate") else
                 "ply" if n.startswith(("ef", "crop", "anchor", "ridge")) else None)
        if group:
            kv.setdefault(group, float(rng.uniform(*SURFACE_VAL)))
            m.geom_rgba[i, :3] = np.clip(m.geom_rgba[i, :3] * kv[group], 0, 1)
    p["surface_val_x"] = {k: round(v, 3) for k, v in kv.items()}
    # --- fruit: colour within class, mass ---
    km = float(rng.uniform(*FRUIT_MASS))
    for f in fruit or []:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f["geom"])
        dh, s_rng, v_rng = FRUIT_HSV[f["ripeness"]]
        base = np.array(tm.RIPENESS_RGBA[f["ripeness"]][:3])
        m.geom_rgba[g, :3] = _hsv_jitter(base, rng, dh, s_rng, v_rng)
        b = m.geom_bodyid[g]
        m.body_mass[b] *= km
        m.body_inertia[b] *= km
    p["fruit_mass_x"] = round(km, 3)
    # --- cameras ---
    for name, dp, dr in (("birdseye", BIRDSEYE_POS_M, BIRDSEYE_ROT_DEG), ("arm/claw", CLAW_POS_M, CLAW_ROT_DEG)):
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        off = rng.uniform(-dp, dp, 3)
        m.cam_pos[c] += off
        m.cam_quat[c] = _qmul(m.cam_quat[c], _rot(rng, dr))
        p[f"cam_{name.split('/')[-1]}"] = {"pos_offset_mm": np.round(off * 1000, 1).tolist()}
    # --- contact + stiffness guesses ---
    mu = float(rng.uniform(*FINGER_FRICTION))
    kf = float(rng.uniform(*FINGER_STIFFNESS))
    kt = float(rng.uniform(*TRUSS_STIFFNESS))
    for i in range(m.ngeom):
        n = m.geom(i).name
        if m.geom_contype[i] and (n.startswith("arm/finray_") or n == "arm/palm"):
            m.geom_friction[i, 0] = mu
    for j in range(m.njnt):
        n = m.joint(j).name
        if n.startswith("arm/finray_"):
            m.jnt_stiffness[j] *= kf
        elif n.endswith(("_rachis_ball", "_ped_ball")):
            m.jnt_stiffness[j] *= kt
    p.update(finger_friction=round(mu, 3), finger_stiffness_x=round(kf, 3), truss_stiffness_x=round(kt, 3))
    return p
