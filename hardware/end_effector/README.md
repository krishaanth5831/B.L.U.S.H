# Picking end effector

STLs, CAD source and test results for the SO-101 jaw that picks single cherry tomatoes.

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

- [ ] 10/10 clean single-fruit picks, with no neighbouring fruit knocked off
- [ ] 10/10 fruits still held after detaching, released over the crate without dropping early
- [ ] Gripper servo temperature stays sane over 10 consecutive picks
- [ ] The claw camera sees the target centred when the jaw is aimed right

## Results

_(date · version · grip force · pass/fail per line)_
