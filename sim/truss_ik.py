"""IK for cutting a truss stalk: the blades on the cut point, the stalk lying across the jaw (K7, 2026-11-03).

    ik = truss_ik()
    x = ik.solve(cut_point, stalk_dir, r)       5 arm joints (rad), or None
    p, R = ik.frame(x, r)                       where the cut point and the gripper axes end up

The cut-and-hold tips (scene.add_cut_hold) take the stalk across the jaw along gripper y, the blades on the
-y (stem) side. So a cut pose is: the blades' crossing point on the cut point, and gripper +y along the stalk
pointing away from the stem. That's 5 numbers (3 position + 2 for the y axis) for the arm's 5 joints, so the
finger direction is whatever makes them true; of the poses that do, the one with the fingers pointing down
(coming at the stalk from above, away from the fruit hanging below it) is kept. IK finds this goal pose
only; RRT (rrt.py) finds the route to it.
"""
import numpy as np

import expert
import scene

# The stalk's axis in the gripper frame at the cut, for a stalk of radius r: across the jaw (x) 1 mm off the
# fixed pad's face, at the blades in y and z. Measured on the cut_hold model, jaw open 25 deg
# (check_truss_cut.py "aimed": pad face at x = -3.0 mm, blade centre y -8.2, z -101.5 mm; 2026-11-03).
PAD_FACE_X = -0.0030
CUT_Y, CUT_Z = -0.0082, -0.1015
W_DIR = 0.05                     # m per unit of the y-axis residual (expert.ArmIK.W_ANG)
# Fingers run along gripper -z, so R[2,2] (gripper z's world-up part) is how far they point DOWN. At least this.
DOWN_MIN = 0.2


def cut_point_in_gripper(r):
    return np.array([PAD_FACE_X + r + 0.001, CUT_Y, CUT_Z])


class TrussIK(expert.ArmIK):
    def _res_y(self, x, p, point, s):
        o, R = self.fk(x)
        return np.r_[p - (o + R @ point), W_DIR * np.cross(R[:, 1], s)[[0, 1, 2]]]

    def solve(self, p, s, r, x0=None, iters=80):
        s = np.asarray(s, float) / np.linalg.norm(s)
        point = cut_point_in_gripper(r)
        g = self.P + np.einsum("nij,j->ni", self.R, point)
        cost = np.linalg.norm(g - p, axis=1) + W_DIR * (1 - np.einsum("ni,i->n", self.R[:, :, 1], s))
        cost += 0.05 * (self.R[:, 2, 2] < DOWN_MIN)      # prefer fingers-down starts
        starts = ([np.asarray(x0, float)] if x0 is not None else []) + [self.Q[i] for i in np.argsort(cost)[:12]]
        best = None
        for x in starts:
            x, e = self._lm_y(x.copy(), p, point, s, iters)
            _, R = self.fk(x)
            ok = np.linalg.norm(e[:3]) < 0.002 and R[:, 1] @ s > 0.995
            if ok and R[2, 2] >= DOWN_MIN:               # gripper +z up = fingers (-z) pointing down
                if best is None or R[2, 2] > best[1]:
                    best = (x, R[2, 2])
                if x0 is not None:
                    break
        return None if best is None else best[0]

    def _lm_y(self, x, p, point, s, iters):
        e = self._res_y(x, p, point, s)
        lam = 1e-3
        for _ in range(iters):
            if np.linalg.norm(e) < 2e-4:
                break
            J = np.empty((6, 5))
            for k in range(5):
                dx = np.zeros(5)
                dx[k] = 1e-5
                J[:, k] = (self._res_y(x + dx, p, point, s) - e) / 1e-5
            H, gr = J.T @ J, J.T @ e
            xn = np.clip(x - np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), gr), self.lo, self.hi)
            en = self._res_y(xn, p, point, s)
            if np.linalg.norm(en) < np.linalg.norm(e):
                x, e, lam = xn, en, max(lam / 3, 1e-7)
            else:
                lam *= 4
                if lam > 1e4:
                    break
        return x, e

    def frame(self, x, r):
        o, R = self.fk(x)
        return o + R @ cut_point_in_gripper(r), R


_IK = None


def truss_ik():
    global _IK
    if _IK is None:
        _IK = TrussIK()
    return _IK


if __name__ == "__main__":   # reach scan: which cut points (row x, height z) have a cut pose
    ik = truss_ik()
    y_stem = scene.TARGET_ROW - scene.KEYHOLE_OFF
    print("cut point 15 mm out of the stem; stalk toward the aisle, drooping 30 deg, spread 0;"
          " '.' = no cut pose, else how far the fingers point down (1 = straight down)")
    for z in (0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34):
        row = []
        for x in np.arange(0.22, 0.54, 0.04):
            s = np.array([0.0, -np.cos(np.radians(30)), -np.sin(np.radians(30))])
            p = np.array([x, y_stem, z]) + s * 0.019
            q = ik.solve(p, s, 0.002)
            row.append("." if q is None else f"{ik.fk(q)[1][2, 2]:.2f}")
        print(f"z {z:.2f}: " + " ".join(f"{c:>4}" for c in row))
