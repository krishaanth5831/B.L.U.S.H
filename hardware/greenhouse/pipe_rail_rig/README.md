# Pipe-rail test rig (wooden, knock-down)

A portable demo and sim2real rig for two SO-101s. It has two vine rows of 3D-printed trusses on dowel stems and a two-tube pipe rail in the aisle. One cart rolls on the rail and carries both arms. It replaces the laser-cut house (`../lasercut/`, too expensive) and the printed one (`..`).

- **Size:** 0.99 x 0.44 x 0.76 m. The rail runs out to 1.2 m.
- **Takes apart:** the two row frames are screwed; the three cross ties come off with M6 wing nuts.
- **Parts:** only Gamma parts, about €112. See [`SHOPPING_LIST_GAMMA.md`](SHOPPING_LIST_GAMMA.md) for links, prices on 2026-11-19 and the cut list.
- **Onshape:** private doc "Vinea Pipe-Rail Test Rig",
  <https://cad.onshape.com/documents/366548c7ac101223d75c80f7/w/c597d55c8aa54a7960789577/e/107341004be3c9f5c68afa06>.
  - Part Studio 1 holds the "Pipe-Rail Test Rig" feature: the rig, part codes on the wood, and the CUT PLAN beside it.
  - The second feature, "SO-101 arms on the cart", places the imported arm meshes.

## What stays the same as the sim

These come from `sim/scene.py` and `sim/truss_scene.py`:
- The arm base sits mid-aisle, 200 mm from each row.
- The stems hang from string rails 520 mm above the arm base.
- Trusses hang 140-320 mm above the arm base.
- The birdseye camera stands 150 mm behind the row the arm doesn't face, 340 mm above the arm base.

## What differs from the sim

These are design choices, not yet checked for sim2real:
- The posts stand on the row lines. In the sim, the end frames are 370 mm from the arm; here they're about 225 mm.
- The two arms are 450 mm apart and face opposite rows.
- There is no riser.
- The stems sit 30 mm off the row line.

## Files

| File | What |
|---|---|
| `pipe_rail_rig.fs` | The rig as a FeatureScript feature. It takes 5 dimensions plus 2 switches, "Show cut plan" and "Emboss part codes". Every part is named `code \| what \| cut length \| [item] Gamma product`. |
| `so101_arms.fs` | Template that places the arm meshes. The `__E_*__` / `__MV__` placeholders are filled with the doc's element ids and microversion at upload. |
| `pose_so101_stl.py` | Writes the posed SO-101 meshes (`so101_<reach\|start>_<a\|b>.stl`) from the sim's MJCF. |
| `so101_*.stl` | Those meshes, in metres in the arm base frame. Each is split into two halves to stay under a 10 MB upload limit. |
| `SHOPPING_LIST_GAMMA.md` | Gamma links, quantities, the cut list with marks, and the wheel options. |
| `renders/` | Onshape shaded views of the rig, a joint close-up and the cut plan. |

## Rebuilding the arm meshes

Run this from `sim/`. It needs `fast_simplification`; `pip install --target <dir>` works.

```bash
PYTHONPATH=<dir> python ../hardware/greenhouse/pipe_rail_rig/pose_so101_stl.py ../hardware/greenhouse/pipe_rail_rig
```

⚠️ Onshape assemblies skip mesh pieces that import as open surfaces. That's why `so101_arms.fs` instantiates the imported Part Studios inside the rig's Part Studio instead of using an assembly.

⚠️ With the part codes and the cut plan switched on, the rig takes about 1.5 min to rebuild in Onshape.

## Not done yet

- Printed parts: the truss and clip STLs, and the U-groove wheel and bracket.
- Nothing has been built or tested on hardware.
- The sim has not been updated to this layout. It still has 3 rows, 1 arm, no rail, and end frames 370 mm from the arm.
