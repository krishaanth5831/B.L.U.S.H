#!/usr/bin/env python3
"""Export the fin-ray gripper parts as MuJoCo meshes (metres, in the MJCF body frames).

  python build_finray_meshes.py      writes sim/assets/finray/*.stl

Runs hardware/end_effector/finray/finray_gripper.py's build(), so the sim fingers are the exact
geometry that gets printed. Frames:
  stub_fixed, finger_fixed    body "gripper"             (= URDF gripper_link)
  stub_moving, finger_moving  body "moving_jaw_so101_v1" (= URDF moving_jaw_so101_v1_link)
Checked: the MJCF body poses equal the URDF origins finray_gripper.py uses (FIXED_VIS = geom quat
0 1 0 0, JOINT = body pos 0.0202 0.0188 -0.0234 quat 0.7071 0.7071 0 0, MOVING_VIS = pos 0 0 0.0189).

Needs: numpy shapely trimesh manifold3d (a throwaway venv works, like the finray generator).
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardware" / "end_effector" / "finray"))
import finray_gripper as fg   # noqa: E402

OUT = HERE / "assets" / "finray"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = fg.build()
    to_jaw = np.linalg.inv(fg.JOINT)          # gripper_link (q = 0) -> moving jaw link
    for name, part, T in (("stub_fixed", parts["fixed"]["stub"], np.eye(4)),
                          ("finger_fixed", parts["fixed"]["finger"], np.eye(4)),
                          ("stub_moving", parts["moving"]["stub"], to_jaw),
                          ("finger_moving", parts["moving"]["finger"], to_jaw)):
        m = part.copy()
        m.apply_transform(T)
        m.apply_scale(0.001)                  # ⚠️ finray_gripper works in mm, MuJoCo in metres
        m.export(OUT / f"{name}.stl")
        lo, hi = m.bounds
        print(f"{name:14s} {len(m.faces):6d} faces  bounds mm {np.round(lo*1000,1)} .. {np.round(hi*1000,1)}")


if __name__ == "__main__":
    main()
