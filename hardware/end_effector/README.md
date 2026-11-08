# Picking end effector

STLs, CAD source and test results for the SO-101 jaw.

**Printable design:** [`finray/`](finray/), for single cherry tomatoes. TPU fin-ray fingers bolted into
PLA stubs of the stock jaws, so the servo, horn and wrist stay stock. Watch it (a window with a jaw-angle
slider) with `python finray/view_gripper.py --stock`.

⚠️ **Whole-truss harvesting (K7 2026-11-03) needs other fingers**, so far only in the sim: rigid PLA
fingers with a blade pair on the stem-side face of the tips and spring-loaded clamp pads on the rest of
their width, so one close cuts the truss's stalk and holds it (`sim/scene.py` `add_cut_hold`,
`sim/truss.py`; watch it with `python sim/check_truss_cut.py --windowed` in the lerobot env). There is no
printable CAD for them in this repo yet.

⚠️ Lock this before recording. The tool geometry is baked into every demonstration,
so changing it afterwards throws the episodes away.

From the vault (note 01, written for truss cutting; still relevant):

- lerobot limits gripper torque on every connect (`so_follower.py`: `Max_Torque_Limit`
  500, `Protection_Current` 250, `Overload_Torque` 25). Design for about **half**
  of the STS3215 stall torque.
- TPU (95A) or silicone pads. Print PETG/PLA+ with 4 walls and 40 %+ infill.
- Measure the actual grip force (luggage or kitchen scale) and write it below with how
  you measured it.

## Acceptance (teleop, real fruit on the rig)

Written for single fruit; for trusses the same lines apply to a cut truss held and carried whole.

- [ ] 10/10 clean single-fruit picks, with no neighbouring fruit knocked off
- [ ] 10/10 fruits still held after detaching, released over the crate without dropping early
- [ ] Gripper servo temperature stays sane over 10 consecutive picks
- [ ] The claw camera sees the target centred when the jaw is aimed right

## Results

_(date · version · grip force · pass/fail per line)_
