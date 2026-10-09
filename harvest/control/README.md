# Target-centring controller (hybrid approach)

Not built yet. This is only needed if the hybrid option in `../README.md` is chosen.

What it has to do: take the gate's target pixel in the claw frame (`pick_target()`
→ `centre`) and move the follower in small steps until that pixel sits at the
image point where the jaw closes. Then hand over to the policy for the final grasp.

Build it testable on its own: target pixel in, joint deltas out, with a "centred
within N px" exit condition. Measure N from where the jaw actually closes in the
claw view, and write how you measured it next to the constant.
