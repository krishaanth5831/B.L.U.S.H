"""Copied from S.H.E.A.R sim/rrt.py (commit 7e8e215) for truss harvesting (K7 2026-11-03: RRT for every route,
IK for the goal poses). Changed: joint names and limits come from arm_map (B.L.U.S.H's follower ranges, the ones
expert.ArmIK solves in) instead of SHEAR's cutpose; blade_guard keeps its name (B.L.U.S.H's blades are
arm/blade_fixed / arm/blade_moving, scene.add_cut_hold).

RRT-Connect route planning for the SO-101 arm, with a list of what it must not touch (K7 2026-11-01: plan every
arm move with RRT, so it always steers around what it shouldn't damage).

    chk = Checker(m, d, hard=geoms, soft=geoms)        a kinematic copy of the scene as it stands now
    path = plan(chk, q_start, q_goal, rng)             joint configs (5 arm joints, rad) from start to goal, or None
    chk.hits(q), chk.path_ok(path)                     what the arm touches at q / along a path

What RRT does and what IK does: IK (truss_ik.py here, cutpose.py in S.H.E.A.R) finds the joint angles that put the
blades on the stalk, the goal. RRT finds a route there through joint space that keeps the arm clear of everything
on the list: it grows two trees of collision-free arm poses, one from the start and one from the goal, until they
meet, then the route is shortened by cutting corners wherever the straight joint-space line between two of its points
is clear too.

The obstacle sets, as S.H.E.A.R defined them (leaves, petioles) and as B.L.U.S.H's truss_expert.py fills them:
- **hard**: never touched, never closer than the clearance. B.L.U.S.H: the greenhouse, crate, table, every stem.
- **firm**: always avoided on routes (firm_clearance); a finger pad may rest on them only where a caller asks for
  firm=False. B.L.U.S.H: the target truss's own stalk + fruit (firm=False on the way home, once it's dropped).
  ⚠️ S.H.E.A.R: when its fallback route could brush them too, it pushed a truss stalk with 8 N (2026-11-03).
- **soft**: avoided by soft_clearance. B.L.U.S.H: every OTHER truss; truss_expert retries with a negative
  soft_clearance (brushing by 2 mm) rather than S.H.E.A.R's soft=False.
- **blade_guard**: things the BLADES never come near, even with soft=False (S.H.E.A.R: truss stalks). Unused in
  B.L.U.S.H (no caller passes it).

The checker moves only the arm: the plant stands where it was when the Checker was made (settled), kinematically.
Whatever the arm pushes in the physics moves afterwards; that's the dry run's job to judge.
"""
import copy

import mujoco
import numpy as np

import arm_map

CLEARANCE = 0.004        # m kept from hard obstacles on a route (B.L.U.S.H expert.CLEARANCE_M)
BLADE_CLEARANCE = 0.002  # m the blades keep from blade_guard things (guess, unverified)
SOFT_CLEARANCE = 0.002   # m kept from soft / firm things by default (SHEAR's kept leaves; guess, unverified)
STEP = np.radians(3.0)   # joint-space resolution of collision checks along a segment (largest joint)
EXTEND = np.radians(12.0)    # RRT step: how far a tree grows toward a sample (design choice)
MAX_ITERS = 4000         # per plan() call
GOAL_BIAS = 0.15         # share of samples aimed at the other tree's newest node (RRT-Connect does this anyway)
SHORTCUT_TRIES = 150     # random corner cuts per path (copied from S.H.E.A.R, unverified)


class Checker:
    def __init__(self, m, d, hard=(), soft=(), ignore=(), blade_guard=(), firm=(), prefix="arm/"):
        self.m = copy.deepcopy(m)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = d.qpos
        self.adr = np.array([self.m.joint(prefix + j).qposadr[0] for j in arm_map.JOINTS[:5]])
        body_arm = np.array([self.m.body(b).name.startswith(prefix) for b in range(self.m.nbody)])
        self.is_arm = body_arm[self.m.geom_bodyid]
        for g in np.flatnonzero(self.is_arm & (self.m.geom_contype > 0)):
            self.m.geom_margin[g] = CLEARANCE           # report contacts this close, not only touching
        self.hard, self.soft, self.ignore, self.guard = set(hard), set(soft), set(ignore), set(blade_guard)
        self.firm = set(firm)
        # ⚠️ B.L.U.S.H's vine stems (and knuckles) don't collide in the physics (contype 0), so no contact would
        # ever report them. In this planning copy everything on a list collides with the arm.
        for g in self.hard | self.soft | self.firm | self.guard:
            if not self.m.geom_contype[g] and not self.m.geom_conaffinity[g]:
                self.m.geom_contype[g], self.m.geom_conaffinity[g] = 1, 1
        self.blades = {g for g in range(self.m.ngeom) if self.is_arm[g] and (self.m.geom(g).name or "").startswith(
            prefix + "blade_") and self.m.geom_contype[g]}
        self.n_checks = 0

    def hits(self, q, soft=True, clearance=CLEARANCE, soft_clearance=SOFT_CLEARANCE, ignore=(), firm=True,
             firm_clearance=SOFT_CLEARANCE):
        """Names of what the arm comes within the clearance of at joint config q (5, rad)."""
        self.n_checks += 1
        self.d.qpos[self.adr] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        out = set()
        for c in self.d.contact[:self.d.ncon]:
            a, b = self.is_arm[c.geom1], self.is_arm[c.geom2]
            if a == b:
                continue
            other = c.geom2 if a else c.geom1
            if other in self.ignore or other in ignore:
                continue
            mine = c.geom1 if a else c.geom2
            if other in self.guard and mine in self.blades and c.dist < BLADE_CLEARANCE:
                out.add("blade on " + (self.m.geom(other).name or f"geom{other}"))
            elif other in self.hard and c.dist < clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
            elif firm and other in self.firm and c.dist < firm_clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
            elif soft and other in self.soft and c.dist < soft_clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
        return out

    def free(self, q, **kw):
        return not self.hits(q, **kw)

    def segment_free(self, q0, q1, **kw):
        n = max(2, int(np.ceil(np.abs(np.asarray(q1) - q0).max() / STEP)) + 1)
        return all(self.free(q0 + (q1 - q0) * s, **kw) for s in np.linspace(0, 1, n)[1:])

    def path_ok(self, path, **kw):
        return all(self.segment_free(a, b, **kw) for a, b in zip(path[:-1], path[1:]))


def _steer(q_from, q_to, step=EXTEND):
    d = q_to - q_from
    n = np.abs(d).max()
    return q_to.copy() if n <= step else q_from + d * (step / n)


def _connect(chk, tree, parents, q, kw):
    """Grow `tree` toward q until it reaches q (returns 'reached') or hits something ('trapped')."""
    while True:
        i = int(np.argmin(np.abs(np.array(tree) - q).max(1)))
        q_new = _steer(tree[i], q)
        if not chk.segment_free(tree[i], q_new, **kw):
            return "trapped", None
        tree.append(q_new)
        parents.append(i)
        if np.abs(q_new - q).max() < 1e-9:
            return "reached", len(tree) - 1


def _trace(tree, parents, i):
    out = []
    while i is not None:
        out.append(tree[i])
        i = parents[i]
    return out[::-1]


def plan(chk, q_start, q_goal, rng, lo=None, hi=None, iters=MAX_ITERS, **kw):
    """RRT-Connect from q_start to q_goal (5 joints, rad) inside [lo, hi]. Returns the shortcut path, or None.
    kw goes to Checker.hits (soft/firm clearances, ignore, soft=False / firm=False)."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    lim = arm_map.FollowerMap().rad_limits()
    lo = np.asarray(lim[0][:5]) if lo is None else lo
    hi = np.asarray(lim[1][:5]) if hi is None else hi
    if not chk.free(q_start, **kw) or not chk.free(q_goal, **kw):
        return None
    if chk.segment_free(q_start, q_goal, **kw):
        return [q_start, q_goal]
    A, pa, B, pb = [q_start], [None], [q_goal], [None]
    for _ in range(iters):
        q_rand = rng.uniform(lo, hi)
        i = int(np.argmin(np.abs(np.array(A) - q_rand).max(1)))
        q_new = _steer(A[i], q_rand)
        if chk.segment_free(A[i], q_new, **kw):
            A.append(q_new)
            pa.append(i)
            status, j = _connect(chk, B, pb, q_new, kw)
            if status == "reached":
                pathA, pathB = _trace(A, pa, len(A) - 1), _trace(B, pb, j)
                path = pathA + pathB[::-1][1:]
                if A[0] is not q_start:                  # trees were swapped an odd number of times
                    path = path[::-1]
                return shortcut(chk, path, rng, **kw)
        A, pa, B, pb = B, pb, A, pa                      # grow the other tree next time
    return None


def shortcut(chk, path, rng, tries=SHORTCUT_TRIES, **kw):
    """Cut corners: join two random points of the path straight when that segment is clear."""
    path = [np.asarray(q, float) for q in path]
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if chk.segment_free(path[i], path[j], **kw):
            path = path[:i + 1] + path[j:]
    return path
