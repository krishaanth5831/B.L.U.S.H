# Holding a thin plant stalk between compliant pads in MuJoCo 4.0.1, and measured tomato-truss properties

Scope: (A) MuJoCo's contact and friction model for a capsule pinched between two box pads, ways to model pad compliance and patch contact, and how grasp holding is validated; (B) measured truss, peduncle and fruit properties, with bench tests for what is missing.

Version note: MuJoCo 4.0.1 ships its own documentation inside the Python wheel (`site-packages/mujoco/doc/*.rst`, read in the `lerobot-mj4` env on 2026-11-25). Wherever this note quotes "4.0.1 docs" it means those bundled files. The matching public pages are https://mujoco.readthedocs.io/en/latest/ (the "latest" docs may already be newer than 4.0.1). Source code quotes come from the `4.0.1` git tag unless marked "main".

Source checking:
- Read directly by me: the MuJoCo docs, changelog and source, plus the super-glue detail and the 0.58–2.46 N range in Xie et al. 2021 and the abrasion and slip quotes in van den Bent et al.
- Read by research sub-agents, not re-checked by me: the GitHub issue and PR threads (#3328, #3526, #3349, #786, #698, #239, #3877, #3901, #3628), the Menagerie, robosuite, Playground and ManiSkill XML values, and the paper numbers.

Own tests: several findings below come from small experiments run in MuJoCo 4.0.1 on 2026-11-25. They have no URL, so they are listed under Inferences and marked **own test, MuJoCo 4.0.1**. Test setup, so they can be repeated: gravity off; one free capsule "stalk" (radius 2 mm, 60 mm long, mass 0.09 kg); two box pads (half-sizes 3 x 6.5 x 18 mm, so a 13 x 36 mm face) on slide joints pushed by **constant-force motors of 4 N each** (not springs); pad geoms `friction="1.5 0.002 0.0001" condim="6"`, stalk `friction="0.9 0.01 0.001" condim="4"`; `solref="0.004 1"`, default `solimp="0.9 0.95 0.001 0.5 2"`, timestep 2 ms, Newton solver, elliptic cones, impratio 10 unless stated. Load = a constant external torque on the stalk body (`xfrc_applied`), rotation read every 0.5 s or 1 s. "Pivot" = rotation about the pad normal; "spin" = rotation about the stalk's own axis. Runs that diverged (MuJoCo BADQACC warning, then auto-reset) are left out or marked.

## A1. Contact points, condim, torsional and rolling coefficients, parameter mixing, and the torque one contact can resist

### Takeaway
A capsule on a box is handled by an analytic "primitive" collider that returns at most **2 contacts per pad**. In 4.0.1 they sit at the two ends of the overlap between the stalk and the pad face, each carrying half the pad force. Because the two geoms' friction values combine as an element-wise maximum, the stalk's torsional 0.01 m overrides the pad's 0.002 m. So the simulated grip currently assumes a 10 mm contact patch on a 4 mm stalk. Against rotation about the pad normal, this two-end-point contact gives about **2x the friction torque of a real, evenly loaded line contact**, plus the torsional term on top. Too few contact points is therefore not what makes the stalk pivot (see A2).

### Cited Findings
- **How many contacts each pair gets.** The pair-wise collider table in the 4.0.1 docs (also at https://mujoco.readthedocs.io/en/latest/computation/index.html#pair-wise-colliders):
  - Capsule-Box: "primitive", max **2**.
  - Capsule-Plane: primitive, 2.
  - Capsule-Capsule: primitive, 2.
  - Box-Plane: primitive, 4. Cylinder-Plane: primitive, 4.
  - Pairs involving a cylinder go through general convex collision (CCD):
    - Capsule-Cylinder: 1 contact with multiccd off, 2 with native multiccd (the default), 5 with the legacy pipeline.
    - Cylinder-Cylinder: 1 / 4 / 5 in the same three cases.
  - The function `mj_maxContact` returns these numbers at run time. — [MuJoCo docs, Computation: Pair-wise colliders](https://mujoco.readthedocs.io/en/latest/computation/index.html#pair-wise-colliders)
  - I did not read the Cylinder-Box cell of this table. Own test of `mj_maxContact(m, box, stalk, 0)` in 4.0.1:
    - Cylinder-box: 4 with multiccd on, 1 with it off.
    - Capsule-box: 4, even though the table says 2. In the parallel-face pinch the capsule-box collider actually produced 2 contacts per pair.
- **multiccd.** "Some colliders can return more than one contact per colliding pair … capsule-plane and box-plane colliders can return up to two or four contacts". General convex collision returns a single point unless multiccd is on.
  - Legacy multiccd rotates the geoms by ±1e-3 rad and re-collides, at about 5x the cost.
  - Single-shot multiccd (with nativeccd) has "very little performance overhead" and supports "capsules, cylinders, boxes, and meshes". — [Computation: Multiple contacts](https://mujoco.readthedocs.io/en/latest/computation/index.html#multiple-contacts)
- **multiccd is on by default and is a *disable* flag in 4.x.** The XML reference gives `multiccd: [disable, enable], "enable"`. In 4.0.1 the C enum is `mjDSBL_MULTICCD`, and `mjENBL_MULTICCD` no longer exists.
  - It became default-on in 3.8.0: "now enabled by default … little performance overhead and improves stability".
  - Single-shot multicontact for cylinders arrived in 3.13.0 and for capsules in 3.15.0.
  - 4.0.1 fixed cylinder-edge multicontact bugs (issues #3748 and #3752). — [XML reference: multiccd](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-flag-multiccd); [changelog](https://github.com/google-deepmind/mujoco/blob/main/doc/changelog.rst); [4.0.1 mjtype.h](https://github.com/google-deepmind/mujoco/blob/4.0.1/include/mujoco/mjtype.h)
- **Open 4.0.1 report (#3877): a box edge lying along a cylinder's side gets only one multiccd contact, at a position that depends on vertex order.** This matters if the stalk is ever modelled as a cylinder. — [issue #3877](https://github.com/google-deepmind/mujoco/issues/3877) (user report; no maintainer reply seen)
- **What each condim value means** (4.0.1 docs):
  - condim 1: frictionless.
  - condim 3: "regular frictional contact".
  - condim 4: adds "torsional friction torque opposing rotation around the contact normal, corresponding to a torque generated by a contacting surface patch. This is useful for modeling soft fingers". "Torsional friction coefficients have units of length which can be interpreted as the diameter of the surface contact patch."
  - condim 6: adds rolling friction. "Rolling friction coefficients also have units of length which can be interpreted as the depth of the local deformation within which energy is dissipated."
  - condim cannot be 2 or 5. — [Computation: condim](https://mujoco.readthedocs.io/en/latest/computation/index.html#condim)
  - ⚠️ The same docs contradict themselves. The 4.0.1 modeling tuning table says `friction[1]` and `friction[2]` "have units of length (contact-patch radii)", while the computation chapter says "diameter". — [Modeling: tuning table](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Friction cone (elliptic)**, for one contact with force vector f, where f₁ is the normal force and μ are the friction coefficients:

  f₁ ≥ 0 and f₁² ≥ Σᵢ fᵢ²/μᵢ₋₁²

  Pyramidal cones use 2(n−1) edges. — [Computation: Friction cones](https://mujoco.readthedocs.io/en/latest/computation/index.html#friction-cones)
- **How the two geoms' values combine.**
  - condim: the higher-priority geom wins; on equal priority, the maximum condim is used.
  - friction: the higher-priority geom wins; otherwise "the element-wise maximum of each friction coefficient over the two geoms is used".
  - solref/solimp: the higher-priority geom wins; otherwise a solmix-weighted average, except that the element-wise minimum is used if either solref is in direct (non-positive) format.
  - margin and gap: summed.
  - An explicit `<pair>` overrides all of the above. — [Modeling: Contact parameters](https://mujoco.readthedocs.io/en/latest/modeling.html)
  - 4.0.1 tuning table: "Set `priority="1"` on the geom whose parameters should govern". — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Default friction is `"1 0.005 0.0001"`**: sliding, then torsional about the normal, then rolling about both tangent axes. — [XML reference: geom friction](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom-friction)
- **Practitioner example of a torsional-friction fix.** In BiGym 2.0 (MuJoCo 3.8.1), grasped plates slowly rotated in the gripper, traced to torsional friction at the finger pads.
  - Pads had sliding friction 1.5 and priority 1.
  - Raising torsional friction from 0.01 to 0.05 cut 6 s of slip from 0.745 mm to 0.002 mm.
  - Copying the plate's `solref`/`solimp` onto the pads removed the rest. Contact priority had replaced the plate's values with MuJoCo defaults. — [BiGym 2.0, arXiv 2610.07594, Appendix F](https://arxiv.org/abs/2610.07594) (read by a sub-agent; not re-checked by me)

### Inferences
- **Contacts the current model actually generates** (own test, MuJoCo 4.0.1, constant 4 N per pad):
  - Each capsule-box pair gave exactly 2 contacts, with normal force 2.00 N each.
  - Contact positions were at the pad face edges, at z = ±18 mm when the stalk lies along the 36 mm length and y = ±6.5 mm when it lies across the 13 mm width.
  - The combined contact had `dim=6` and friction `[1.5, 1.5, 0.01, 0.001, 0.001]`. So the pad's torsional 0.002 and rolling 0.0001 are overridden by the stalk's 0.01 and 0.001 (element-wise max). If the pad values are the intended ones, they only take effect with `priority` on the pad, or with the stalk set to the same or lower values.
  - Turning multiccd off changed nothing (still 4 contacts), as expected for an analytic primitive pair.
  - A `cylinder` stalk also gave 4 contacts (2 per pad).
- **Torque one contact can resist about its normal.** From the elliptic cone, the torsional component alone is bounded by |τ| ≤ μ_torsion · f_n. Combined with sliding at a lever arm a from the pivot, the elliptic bound on one contact's resisting moment is f_n · √((μ_s·a)² + μ_t²). (Derived from the cone formula above.)
- **Theoretical pivot limits.** Same pad force F per pad, sliding μ, overlap length L, stalk radius r:

  | Pivot about pad normal | Real, uniform line contact (2 pads) | MuJoCo capsule-box: 2 end contacts of F/2 per pad, plus torsion |
  |---|---|---|
  | Formula | 2 · μ·F·L/4 = μFL/2 | ≈ 4 · (F/2) · √((μ·L/2)² + μ_t²) |
  | μ=1.5, F=4 N, L=13 mm | 0.039 N·m | 0.112 N·m (no torsion: 0.078) |
  | μ=1.5, F=4 N, L=36 mm | 0.108 N·m | 0.230 N·m (no torsion: 0.216) |

  | Spin about the stalk axis | Real (2 pads) | MuJoCo |
  |---|---|---|
  | Formula | 2·μ·F·r | about the same (sliding at radius r; rolling term negligible) |
  | μ=1.5, F=4 N, r=2 mm | 0.024 N·m | 0.024 N·m |

  - The uniform-line value μFL/4 per pad is the integral of μ·(F/L)·|x| over the line.
  - A real line contact is only uniform if the pad is flat, parallel and compliant enough. A stiff or tilted pad concentrates pressure towards one end and gives less.
  - Two point loads at the ends give μFL/2 per pad, twice the uniform value. **So capsule-on-box over-states the breakaway torque for pivoting, and under the same assumed μ the real pads would let go first.**
- **Gravity torque the grip has to hold** (estimate, not measured). Truss weight 0.69–0.98 N (70–100 g) times the horizontal offset of the truss centre of mass from the grip axis:
  - 20 mm: 0.014–0.020 N·m.
  - 40 mm: 0.027–0.039 N·m.
  - 60 mm: 0.041–0.059 N·m.

  These overlap both the real-pad pivot limit across the 13 mm face (about 0.039 N·m at μ=1.5) and the spin limit (0.024 N·m). **Both modes are marginal at 4 N even with μ=1.5, and worse at lower μ.** That makes measuring the real breakaway torque (bench test B6) the deciding experiment.
- **Measured breakaway brackets** (own test, MuJoCo 4.0.1, constant torque, noslip on so creep is removed):
  - Pivot across the 13 mm face: held at 0.10 N·m, broke at 0.15 N·m (theory 0.112).
  - Pivot along the 36 mm face: held at 0.2, broke at 0.5; the 0.3 run diverged (theory 0.23).
  - Spin with noslip 3: held at 0.02 N·m. With noslip 1 it broke at 0.025 N·m; with noslip 3 at 0.03 N·m it diverged (theory 0.024).
  - So the simulated cone limits match the formulas above.

### Gaps
- I did not find a published measurement of the torsional (pivot) friction of a plant stem, or of any 3–5 mm cylinder, pinched between rubber or TPU pads. The μ_t "patch size" has no published value for this case.
- The docs say both "diameter" and "radii" for the torsional coefficient, and I could not settle which is meant. For the grip it matters by a factor of 2.
- Why `mj_maxContact` reports 4 for capsule-box when the docs table says 2 was not resolved. In the pinch tests, 2 contacts per pair were observed both for a capsule stalk and for a cylinder stalk.

## A2. Elliptic vs pyramidal cones, impratio, solref/solimp, noslip, and slow creep of a pinched object

### Takeaway
MuJoCo's friction rows are purely damped and have no stick state. So any steady torque on the stalk makes it **creep at a constant rate, proportional to the torque, even far inside the friction cone**.

In 4.0.1, a 4 mm stalk with the current contact settings (elliptic, impratio 10, solref 0.004, 2 contacts per pad) creeps:
- **about 4 °/s about the pad normal** at 0.02 N·m (stalk across the 13 mm face);
- **far faster about its own axis**: about 9 °/s at only 0.002 N·m and about 44 °/s at 0.01 N·m, because the lever arm is only the 2 mm stalk radius.

These rates come from a test with a 0.09 kg stalk body. A lighter stalk body crept about twice as fast, so the rates in the full model will differ.

**Creep below the friction limit is enough to produce rotations of the reported 55–85° size within seconds**, especially for spin. Whether the full model's rotation is creep or a real friction-limit release has to be decided by measuring it.

How to tell which one is happening (a measurement, not tuning):
- Log rotation angle against time at a fixed gripper orientation, and each contact's force against its cone limit.
- **Steady drift that scales as 1/impratio** is regularisation (solver creep).
- **Sudden motion with contacts on the cone boundary** is a real friction limit.

### Cited Findings
- **Creep is expected behaviour.** 4.0.1 docs: "MuJoCo's regularized soft-contact model does not guarantee an exact zero-velocity stick state. In the recommended elliptic contact model, friction dimensions have no position residual and are purely damped, so a persistent tangential load can produce a nonzero steady slip velocity. This is expected behavior". — [Modeling: Preventing slip, Slow slippage](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)
- **How friction rows are modelled.** Friction dimensions of elliptic cones have r ≡ 0. "The impedance is always d₀ (solimp[0])"; "the stiffness k is always 0"; in the standard solref format "the time constant controls exponential velocity decay. The damping ratio is ignored". — [Modeling: Solver parameters, Friction](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Remedies listed in the 4.0.1 docs.**
  - (a) Elliptic cones and a larger impratio. This "makes the friction dimensions harder relative to the normal dimension without increasing the friction coefficient. It reduces slow slippage but does not guarantee exact sticking."
  - (b) `noslip_iterations` 1–3. "Exact zero slip is not guaranteed. NoSlip also adds computational cost, makes inverse dynamics ill-defined and can occasionally cause instabilities in complex multi-contact systems."
  - The tuning table quotes noslip at "roughly 2x step time" with "occasional chatter in overconstrained grasps", and recommends `impratio` "10 for manipulation". — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **impratio definition.** "ratio of frictional-to-normal constraint impedance for elliptic friction cones … it is not recommended to use high impratio values with pyramidal cones." — [XML reference: impratio](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-impratio)
- **What the 4.0.1 source does with impratio and the friction coefficients.** For elliptic contacts the regulariser is set as:
  - `R[1] = R[0]/impratio` for the sliding rows;
  - `R[j+1] = R[1]*friction[0]²/friction[j]²` for the torsional and rolling rows.

  So a torsional coefficient that is small relative to the sliding coefficient gets a much softer (larger-R) row. — [engine_core_constraint.c @ 4.0.1](https://github.com/google-deepmind/mujoco/blob/4.0.1/src/engine/engine_core_constraint.c) (lines ~2132–2142). On `main` the code differs slightly: it uses an "unclamped" normal R0.
- **solref clamping.** With `refsafe` enabled (the default), the solver uses max(solref[0], 2·timestep). So at a 2 ms timestep, the current solref 0.004 is already at the floor and cannot be made stiffer. The exception is the `discrete` integrator, where refsafe instead replaces rows that are too stiff. — [XML reference: refsafe](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-flag-refsafe)
- **solimp[0].** The tuning table says `solimp[0]` 0.95 "Halves rest penetration and creep, at the cost of more solver iterations". — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Issue #3328: a pinched capsule creeping out of a grip.** MuJoCo 3.8.1; 45 g capsule between pads with μ=0.7 at 12–60 N, which is 20–95x the force needed.
  - Creep was about 50–155 µm/s with elliptic cones and impratio 10, and about 1500–2300 µm/s with impratio 1 or pyramidal cones.
  - The reporter found creep "scales ≈ linearly with 1/impratio" and that noslip cut it about 110x.
  - Maintainer (yuvaltassa): "This is WAI" (working as intended). — [issue #3328](https://github.com/google-deepmind/mujoco/issues/3328)
- **Docs PR #3526 re-measured these cases on 3.12.1.** Slip rate in µm/s:

  | Setting | Slip rate (µm/s) |
  |---|---|
  | elliptic, impratio 10 | 155 |
  | elliptic, impratio 1 | 1552 |
  | elliptic, impratio 100 | 15.3 |
  | pyramidal, impratio 1 | 2270 |
  | elliptic, condim 6 | 153 |
  | elliptic, stiff solref | 51.3 |
  | elliptic, noslip 10 | 1.41 |

  These are translational slip only, not rotation. — [PR #3526](https://github.com/google-deepmind/mujoco/pull/3526)
- **No stiction flag exists in 4.0.1.** A proposed `stiction` flag explained that "MuJoCo's friction is a function of sliding velocity, not displacement … a grasped object sinks out of the gripper". The PR was closed without merging, and no such flag is in 4.0.1. — [PR #3349](https://github.com/google-deepmind/mujoco/pull/3349)
- **Older maintainer advice, still consistent.**
  - "reduce noslip_iterations to 3. More than 3 will just slow you down" (yuvaltassa) — [issue #698](https://github.com/google-deepmind/mujoco/issues/698)
  - Replace mesh collision with "a layer of (possibly invisible) box primitives" — [issue #239](https://github.com/google-deepmind/mujoco/issues/239)
  - "The collision geoms for the pads are implemented using multiple boxes to generate more contact points" (Robotiq 2F85) — [issue #786](https://github.com/google-deepmind/mujoco/issues/786)
- **Known 4.0.0 bug, fixed in 4.0.1.** A Newton divergence with elliptic cones, impratio 10, noslip and constraint islands was fixed in 9ae9bad01, which is in 4.0.1. — [issue #3901](https://github.com/google-deepmind/mujoco/issues/3901)
- **Possible early stop of the Newton solver (3.14.0).** With elliptic cones, impratio 10 and ls_iterations=20, the solver can stop before converging; 23 or more line-search iterations fixed it. — [issue #3628](https://github.com/google-deepmind/mujoco/issues/3628) (user report)

### Inferences
- **Creep is linear in torque and time, and scales as 1/impratio** (own test, MuJoCo 4.0.1). Stalk along the 36 mm face, rotation rate in °/s at constant torque:

  | Torque (N·m) | impratio 1 | impratio 10 | impratio 100 | impratio 10 + noslip 3 | pyramidal, impratio 10 |
  |---|---|---|---|---|---|
  | 0.01 | 4.8 | 0.48 | 0.049 | 0 | 1.7 |
  | 0.02 | 9.5 | 0.97 | 0.099 | 0 | 3.4 |
  | 0.04 | 18.6, accelerating | 1.95 | 0.20 | 0 | 7.0 |
  | 0.08 | — | 4.0 | 0.42 | 0 | — |
  | 0.15 | — | 7.7 | 1.05 | 0 | (diverged) |

  All of these torques except 0.25 are below the 0.23 N·m cone limit. At 0.25 N·m without noslip the stalk broke away fast.
- **Stalk across the 13 mm face, impratio 10** (own test). About 2.0 / 4.1 / 8.1 / 12.2 / 20 °/s at 0.01 / 0.02 / 0.04 / 0.06 / 0.10 N·m, and breakaway at 0.15 N·m.
  - With noslip 3: no rotation up to 0.10 N·m.
  - At 4 °/s, 55–85° takes about 14–21 s. The torque and the carry duration in the full model are not known here; both are assumptions.
- **What changes the creep rate.** Own test, 13 mm face, 0.02 N·m, impratio 10, °/s:

  | Change | Rate (°/s) | Note |
  |---|---|---|
  | Baseline | 4.1 | |
  | condim 3 on both geoms (no torsion rows) | 8.3 | |
  | Stalk torsional 0.002 instead of 0.01 | 8.0 | |
  | solimp d₀ 0.99 instead of 0.9 | 0.41 | |
  | Pad force 2 N instead of 4 N | 4.2 | Creep does not depend on normal force inside the cone. |
  | Sliding μ 0.5 instead of 1.5 (on both geoms) | 0.8 | ⚠️ Lower μ creeps *less*, because the torsional rows' R scales with μ_s²/μ_t². |
  | multiccd off | 4.1 | |

  So "higher friction" is not a creep fix in MuJoCo. Physical friction values and creep-suppressing solver settings are separate questions.
- **Splitting the pads cuts creep** (own test).
  - One rigid pad split into 4 boxes along the 36 mm length: 12 contacts, 0.56 °/s, against 0.97 for one box.
  - Pads split into independently sprung segments across the 13 mm face (each segment on its own slide joint, k = 500 N/m): load shared evenly.

    | Segments per pad | Contacts | Rate (°/s) |
    |---|---|---|
    | 1 | 4 | 5.2 |
    | 4 | 16 | 1.9 |
    | 8 | 32 | 1.0 |
- **Separating solver settings from physical values.** impratio, noslip_iterations and solimp d₀ are solver settings that suppress a documented numerical artefact. They are not physical constants. Friction coefficients, the patch size (μ_t) and pad stiffness are physical values that need a source or a bench measurement.
- **Spin about the stalk axis creeps much faster than pivot** (own test, stalk along z, 0.09 kg stalk body), in °/s:

  | Torque (N·m) | impratio 10 | impratio 100 | solimp d₀ 0.99 (impratio 10) | noslip 3 |
  |---|---|---|---|---|
  | 0.002 | 8.7 | 0.88 | 0.80 | 0 |
  | 0.005 | 21.8 | — | — | 0 |
  | 0.01 | 43.6 | — | — | 0 |
  | 0.02 | about 87 | — | — | 0 |

  - impratio 1 is about 10x faster again.
  - A cylinder-shaped stalk at 0.01 N·m gave the same 43.6 °/s.
  - So in sim, spin about the stalk is the fastest-creeping mode by far. Truss gravity torques of 0.01–0.06 N·m would spin a 4 mm stalk tens of degrees per second unless noslip is on.
- **Stalk body mass matters** (own test). With a 0.01 kg stalk body instead of 0.09 kg, pivot creep at 0.02 N·m rose from 4.1 to 8.4 °/s. This fits the regulariser being scaled by the constraint-space inertia (`efc_diagApprox` in the R computation of [engine_core_constraint.c @ 4.0.1](https://github.com/google-deepmind/mujoco/blob/4.0.1/src/engine/engine_core_constraint.c); I did not trace the exact formula). The full model's rates depend on its own masses and inertias, so measure them there.
- **Noslip instability** (own test). With noslip, torques at or above the spin limit diverged (BADQACC warning, then auto-reset), for example at 0.03 N·m with noslip 3. This is consistent with the docs' instability warning.

### Gaps
- No published creep measurement for MuJoCo 4.0.x. The numbers in #3328 and #3526 are from 3.8.1 and 3.12.1, and translational only.
- No published measurement of rotational creep (pivot) in a MuJoCo grip apart from BiGym's.
- GitHub Discussions were not searched, so there may be more maintainer advice there.
- The exact dependence of creep rate on body inertia was not derived from the source.

## A3. Modelling a compliant pad and a line or patch contact

### Takeaway
The established practice, from MuJoCo maintainers, MuJoCo Menagerie and robosuite, is **several primitive boxes per pad**, often with `priority="1"` on the pad and its own solref/solimp, elliptic cones and impratio 10. Deformable pads are possible in 4.0.1 with flexcomp (`dof="trilinear"`) and the `discrete` integrator; there is an official gripper example. They cost much more and change the integrator. The old elasticity plugins are gone in 4.x. I found **no published model with independently sprung pad segments**; my own test shows they spread the load evenly and reduce creep.

### Cited Findings
- **Docs advice on pad geometry.** "Improve the geometry of the contacting geoms in order to add more contact points, possibly with non-flat geometry (e.g., bumps), so slippage is prevented by the normal force and not only frictional components". The docs also say to keep multiccd and nativeccd on. — [Modeling: Preventing slip](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)
- **Cost of primitives.** The tuning table says primitive pairs are much cheaper than general convex collision and "far cheaper than sdf collisions". — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Menagerie Robotiq 2F-85.** The README lists "Broke up collision pads into two pads for more contacts", "Increased pad friction and priority" and "Added `impratio=10` for better noslip".
  - XML: `cone="elliptic" impratio="10"`.
  - Two box pads per finger with friction 0.7 and 0.6, `solimp="0.95 0.99 0.001" solref="0.004 1" priority="1"`, default condim 3.
  - The silicone pad mesh is visual only. — [2f85.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/2f85.xml), [README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/README.md)
- **Menagerie Franka Panda hand.** Five box pad geoms per finger: one 17 mm box plus four 6x4x6 mm corner boxes. Default friction and condim. — [hand.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/hand.xml)
- **Menagerie SO-101.**
  - `timestep="0.005" cone="elliptic" impratio="10" ls_iterations="20"`.
  - Gripper collision class `condim="6" friction="1 5e-3 5e-4" solref="0.01 1" priority="1"`.
  - Jaws built from several thin boxes, small sphere "tips" and a capsule. — [so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml)
- **Menagerie SO-ARM100.** Four 1 mm-thick box pads per jaw, `friction="1 0.005 0.0001"`, elliptic, impratio 10. — [so_arm100.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/trs_so_arm100/so_arm100.xml)
- **Menagerie UMI gripper.** The only Menagerie gripper found with noslip: `impratio="10" cone="elliptic" noslip_iterations="2"`, with mesh fingers. — [umi_gripper.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/umi_gripper/umi_gripper.xml)
- **Menagerie ALOHA.** Elliptic, impratio 10, `condim="6" friction="1 5e-3 5e-4"`, plus 13 tiny sphere collision geoms on the fingers. — [aloha.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/aloha/aloha.xml)
- **robosuite.**
  - Global `impratio="20" cone="elliptic"` — [base.xml](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/base.xml)
  - Panda gripper: finger pad box with `friction="2 0.05 0.0001" solref="0.01 0.5"` (sliding 2, torsional 0.05), and finger meshes with `condim="4"`. — [panda_gripper.xml](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/grippers/panda_gripper.xml)
- **MuJoCo Playground SO-101 pick-cube** randomises fingertip friction: sliding uniform 0.3–1.5, torsional log-uniform 1e-4 to 2e-2, rolling log-uniform 1e-5 to 5e-3. — [pick_cube task.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/tasks/manipulation/pick_cube/task.py), [so101 task.py](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/tasks/manipulation/pick_cube/so101/task.py)
- **ManiSkill uses SAPIEN/PhysX, not MuJoCo.** Different model:
  - Static and dynamic friction plus a torsional patch radius.
  - Panda and SO-100 finger material friction 2.0, `patch_radius=0.1`.
  - "If stable grasps are necessary >= 20 [solver position iterations] is preferred". — [ManiSkill types.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/structs/types.py), [so_100.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py)
- **Deformable (flex) pads in 4.0.1.** Official examples: [model/flex/gripper.xml](https://github.com/google-deepmind/mujoco/blob/4.0.1/model/flex/gripper.xml) and [gripper_trilinear.xml](https://github.com/google-deepmind/mujoco/blob/4.0.1/model/flex/gripper_trilinear.xml).
  - The trilinear example uses `cone="elliptic" impratio="10" integrator="discrete" solver="Newton"` and a `flexcomp dim=3 dof="trilinear"` with `elasticity young="1e4" poisson="0.1"`.
  - 4.0.0 made the Newton solver work with flex vertices attached to articulated bodies; `gripper_trilinear.xml` now uses Newton. — [changelog 4.0.0](https://github.com/google-deepmind/mujoco/blob/main/doc/changelog.rst)
  - The 4.0.1 modeling docs say "Models with flex elasticity or passive flex contact require discrete", meaning the discrete integrator. — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)
  - Speculative contacts do not support flexes or SDFs. — [Computation: Speculative contacts](https://mujoco.readthedocs.io/en/latest/computation/index.html#speculative-contacts)
- **Elasticity plugins are gone in 4.x.**
  - 4.0.0 "Removed the `cable` plugin (`plugin/elasticity`) and the `composite` element, which are superseded by 1D flexcomp with elasticity". Shell, solid and membrane plugins were removed earlier, in 3.2.x–3.3.x.
  - The 4.0.1 wheel's plugin folder contains only `libactuator`, `libsdf_plugin` and `libsensor`.
  - An experimental `ipc` contact mode exists (3.14+), but only for flex contact. — [changelog](https://github.com/google-deepmind/mujoco/blob/main/doc/changelog.rst)
- **Contact softness setting.** The 4.0.1 tuning table says to scale solimp's `width` to "the compliant layer or object size". Friction rows use only `solimp[0]`, so the width setting changes how far the pads sink in, not friction or creep. — [Modeling](https://mujoco.readthedocs.io/en/latest/modeling.html)

### Inferences
- **Two ways to get a line contact with the right breakaway torque for the 13 x 36 mm pads**, both using primitives:

  | | Option 1: rigid pad split into N boxes along the stalk | Option 2: independently sprung segments |
  |---|---|---|
  | What it gives | 2N capsule-box contacts | Contacts that share load evenly, like a compliant pad |
  | Pressure distribution | Set by the stiff contact softness | Set by the segment spring rate |
  | Own test, 4.0.1 | 4 boxes on a rigid pad: 12 contacts, creep 0.97 → 0.56 °/s | 4 / 8 segments across 13 mm: 0.5 / 0.25 N per contact, creep 5.2 → 1.9 → 1.0 °/s, stable at 2 ms |
  | Cost | Contact rows grow linearly with N | Adds N slide joints per pad |

  As N grows, the pivot breakaway torque falls from μFL/2 towards the uniform-line value μFL/4 per pad.

  Option 2's spring rate would have to come from a pad indentation measurement (bench test B6-c), not a guess.
- **Choosing the torsional coefficient.** If the docs' "patch diameter" reading holds, μ_t should be about μ_s × (contact-patch width across the stalk). For a 3–5 mm stalk pressed into a rubber or TPU pad, the patch width is probably 1–3 mm. **The current 0.01 m (10 mm) is larger than the stalk is wide.** This should be set from the measured indentation width, not tuned.
- **When a flex pad is worth it.** Mainly when the pad visibly wraps the stalk, which raises spin resistance by shape, not just friction. It forces `integrator="discrete"`, which also changes how refsafe treats stiff contacts. So it is a bigger change than geometry splitting and should come after B6 shows wrapping matters.

### Gaps
- No published work with independently sprung pad segments in MuJoCo was found.
- No published flexcomp gripper pad validated against a real pad's stiffness.
- No cost benchmark for a flex pad at a 2 ms step.
- SDF pads were not examined beyond the docs' cost statement.

## A4. How people have validated or identified friction parameters for simulated grasps

### Takeaway
Few published studies compare MuJoCo grasp holding against real measurements. The most direct one, a 2017 in-hand manipulation study, found that **MuJoCo's soft contacts mis-predict pivoting by a lot** (about 0.6 rad predicted against about 1 rad measured). More recent work either hardens contacts and turns on noslip to match a real non-slipping gripper, fits friction and solref/solimp by matching sim and real trajectories, or randomises friction across wide ranges. **No paper was found that identifies MuJoCo's torsional coefficient from measured pivot torques**, so a bench measurement is needed.

### Cited Findings
- **Experimental Validation of Contact Dynamics for In-Hand Manipulation** (2017).
  - Thousands of sliding, pivoting and rolling trials with a force-controlled parallel-jaw gripper.
  - Calibrated finger friction μ was 0.35–0.5.
  - MuJoCo's soft constraints "also limit the prediction accuracy": predicted pivot about 0.6 rad against about 1 rad measured. Rolling only matched with silicone μ "close to 2.5".
  - Repeat trials varied by up to 20% in force. — [arXiv 1702.07252](https://arxiv.org/abs/1702.07252) (read by a sub-agent)
- **Variable-friction in-hand manipulation.** "MuJoCo's soft contact model is hardened by adjusting solver parameters (solref, solimp), using an elliptical friction cone, Newton solver, and a high impratio. The Noslip solver (optional) ensures no slip, matching the real gripper's behavior. The MultiCCD flag is enabled".
  - Parameters were set by system identification against real trajectories, then used as domain-randomisation ranges. — [arXiv 2503.02738](https://arxiv.org/abs/2503.02738) (read by a sub-agent)
- **BiGym 2.0** fixed slow plate rotation with torsional friction 0.05 and matching pad solref/solimp (details in A1). Their target was "no visible slip", not a measured real value. — [arXiv 2610.07594](https://arxiv.org/abs/2610.07594) (read by a sub-agent)
- **Compliant fingers for connector insertion.**
  - Torsional friction 0.01–0.05, tangential 1.
  - Sim-to-real agreement only about 27% on one task, with offsets or even reversed trends on others. — [arXiv 2509.10012](https://arxiv.org/abs/2509.10012) (read by a sub-agent)
- **Roller Grasper V2.** "By default, Mujoco assumes a soft-contact model which leads to relatively slippery contacts, but the solver parameters (solref and solimp) were changed to have a harder contact model". — [arXiv 2004.08499](https://arxiv.org/abs/2004.08499) (read by a sub-agent)
- **solref direct format.** The docs recommend the direct (negative) solref format for system identification: "(-stiffness, -damping)". — [Modeling: Solver parameters](https://mujoco.readthedocs.io/en/latest/modeling.html)

### Inferences
- The 4.0.1 wheel ships a `mujoco.sysid` package (seen on disk at `site-packages/mujoco/sysid`). It may help fit parameters to bench data; not checked.
- The validation that fits this rig is a **matched pair**:
  1. A bench test measuring breakaway torque for pivot and spin at known clamp force (B6).
  2. The same load case in sim: a constant torque on the stalk, gravity off, spring pads as built.

  Compare the breakaway torque with noslip on, which sets μ and μ_t. Separately, check that creep at sub-limit torques over the carry time stays below an agreed angle with whatever solver settings are chosen.
- The real-world comparison for creep is zero. Real pads under a sub-limit static load show only small viscoelastic settling, not steady rotation. I did not find a source quantifying this for rubber pads on stems.

### Gaps
- Not found: a sim-to-real pivoting paper that identifies MuJoCo `friction[1]` against measured torques.
- Not checked: quantitative results of solref/solimp fitting papers. Search snippets only, not read: [arXiv 2311.07245](https://arxiv.org/abs/2311.07245) and [arXiv 2603.06218](https://arxiv.org/pdf/2603.06218).
- Not checked: what `mujoco.sysid` covers.

## B1. Fruit detachment (pull-off) force for cherry tomatoes on the truss

### Takeaway
The simulation's 0.58–2.46 N range matches one cherry-tomato study exactly, Xie et al. 2021. That study measured separation **at the calyx/fruit joint, with the knuckle (abscission zone) glued shut**, at chosen pull angles, with no split by ripeness. So it is a real measured range, but **for a different failure point than the knuckle**. I found no measured knuckle pull-off force for cherry tomatoes, by ripeness or otherwise. For large-fruited tomatoes, pedicel and knuckle pull forces are 8–72 N.

### Cited Findings
- **Cherry tomato, cv. "No.1 Hybrid", hydroponic plant factory (Xie et al. 2021).**
  - Separation force **0.58–2.46 N** across a 17-run Box–Behnken design. The three factors were grasping angle 30–90°, horizontal angle 45–135° and pitching angle 0–80°; each run averaged 10 fruits, 170 fruits in total.
  - Pull rate 3 mm/s; the fruit was gripped at 10 N.
  - The upper pedicel and fruit stalk were reinforced "by super glue" so that separation happened at the calyx/fruit joint, not at the separation layer.
  - Best case: 0.54 N measured (0.55 N predicted) at 68° grasp, 135° horizontal, 0° pitch.
  - Main-effect means: 0.67 N at 0° pitch against 2.00 N at 80° pitch.
  - The same paper (n=170) measured:
    - "ultimate breaking force of calyx branches": 0.57 ± 0.31 N;
    - fruit-stalk diameter: 2.10 ± 0.53 mm.
  - Source: [Xie et al. 2021, Agriculture 11:815](https://www.mdpi.com/2077-0472/11/9/815), DOI 10.3390/agriculture11090815. The super-glue detail and the 0.58–2.46 N range were checked in the full text.
- **Cherry tomato, commercial high-wire greenhouse, cultivar not stated (Singh et al., preprint).**
  - These are **fingertip loads during manual picking**, measured with a 6-axis sensor under a 2 mm silicone pad, 50 trials per pattern. They are not pull-off forces at the break point.

    | Picking pattern | Normal force | Torque |
    |---|---|---|
    | Grip | 3.92 ± 0.45 N | 6.48 ± 0.70 N·mm |
    | Twist | 5.09 ± 0.58 N | 19.20 ± 2.10 N·mm about the pedicel axis |
    | Axial pull | 0.22 ± 0.03 N | — |

  - The preprint's pedicel torsional stiffness (25–50 N·mm/rad) and bruise threshold (0.4–0.6 MPa) have no traceable source, so I have not used them.
  - Source: [Singh et al., arXiv 2610.18160](https://arxiv.org/abs/2610.18160), DOI 10.48550/arXiv.2610.18160
- **Large tomato, cv. Pintaro, 182–337 g fruit (Yang et al. 2026).**
  - Pedicel tensile breaking force 12.5 N on average (n=20), peak 16.3 N.
  - Tension combined with torsion: 8.36 N on average at 270° of rotation, 15.2 N maximum (n=60).
  - Source: [Yang et al. 2026, Front. Agr. Sci. Eng. 13(2):25643](https://journal.hep.com.cn/fase/EN/10.15302/J-FASE-2025643), DOI 10.15302/J-FASE-2025643
- **Large-fruited tomato, cv. Syngenta Spectrum (type not stated), early firm-ripe (Weng et al. 2024).**
  - Abscission-zone tensile force measured with a texture analyser at 1 mm/s:

    | Zone diameter | Tensile force |
    |---|---|
    | 5.52 mm | 40.3 N |
    | 6.27 mm | 44.8 N |
    | 7.75 mm | 72.0 N |

  - Source: [Weng et al. 2024, Agronomy 14:2274](https://www.mdpi.com/2073-4395/14/10/2274), DOI 10.3390/agronomy14102274
  - These values contradict Yang 2026 (12.5 N) several-fold. Cultivar and gripping method differ, so the two should not be averaged.
- **Medium to large tomato, cv. Jiangshu No. 1, 66–93 mm diameter (Zheng et al. 2022).** Pulling force about 11.7 N on average, 15.3 N maximum, n=50, measured with a dynamometer. — [Zheng et al. 2022, Front. Plant Sci.](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.974456/full), DOI 10.3389/fpls.2022.974456
- **Fruit loss from a gripped cluster (Kondo et al. 2010).** In a grip-and-cut cluster harvester, 1 of 20 clusters lost a fruit after harvest; the authors advise avoiding impacts on the cluster. — [Kondo et al. 2010, EAEF 3(1):20–24](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_article)

### Inferences
- **What 0.58–2.46 N is good for.** It can stand as "the force that tears a cherry off at the calyx under the measured pull angles". It should not be labelled as the knuckle force or as varying with ripeness. Its spread is mostly due to **pull direction** (pitch 0° → 0.67 N, 80° → 2.00 N), not fruit-to-fruit variation. So a per-fruit random draw over the whole range mixes the two up.
- **Static margin while carrying.** A 5–19 g cherry weighs 0.05–0.19 N. At rest the weakest measured joint (about 0.55 N) has a 3–10x margin. Fruit loss while carrying should therefore come from accelerations or impacts (pendulum swing, contact with the crate), consistent with Kondo's observation.
- Measuring knuckle and calyx pull-off on the cultivar actually used, at known pull angles, is bench test B6-d.

### Gaps
- Not found: cherry-tomato knuckle (abscission-zone) pull-off force; any cherry detachment force split by ripeness; bending or twisting detachment for cherry at the break point.
- Not read, paywalled or bot-walled:
  - Gao et al. 2024, Biosystems Engineering, DOI 10.1016/j.biosystemseng.2024.01.009. The abstract only ranks picking patterns.
  - Feng et al. 2018, IJABE, DOI 10.25165/j.ijabe.20181101.2853.

## B2. Peduncle (truss stalk) and rachis: diameter, stiffness, density, crushing and cutting force

### Takeaway
**There is no published measurement of a cherry-tomato peduncle's diameter, stiffness, density, crush limit or friction.** The available stem data is for large-fruited tomatoes:

| Large-fruit measurement | Value | Source |
|---|---|---|
| Truss peduncle diameter | 4.8–11.8 mm | Kondo |
| Pedicel shear force | 20–62 N, rising with diameter | Weng |
| Peduncle cutting force | about 60 N for the largest peduncle | Kondo |

The simulation's 3–5 mm stalk diameter is plausible for cherry, but **unsourced**.

### Cited Findings
- **Large-fruit tomato on a high-wire system; cultivar and sample size not given (Kondo et al. 2010).**
  - Peduncle diameter 4.8–11.8 mm, mean 7.0 mm.
  - Peduncle length 97.5–226.9 mm (mean 148.4); main stem to first fruit 33–132 mm (mean 79).
  - Cluster mass about 400 g.
  - "Cutting resistance" about 60 N for the largest peduncle; cutter type not described.
  - **Design values, not measured limits:** at least 5 N of frictional resistance needed; 40 N chosen for fast motion; spring grip 20 N at 8 mm (2.33 N/mm).
  - Source: [Kondo et al. 2010, Eng. Agric. Environ. Food 3(1):20–24](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_article). DOI not confirmed.
- **Cv. Syngenta Spectrum, large-fruited (Weng et al. 2024).** Blade type not described in the text extracted.
  - Pedicel shear at 90°, 1 mm/s, n=20:

    | Section | Diameter | Shear force | Energy |
    |---|---|---|---|
    | Proximal | 4.11 ± 0.40 mm | 62.1 ± 15.7 N | 0.256 J |
    | Distal | 3.33 ± 0.39 mm | 33.2 ± 8.0 N | 0.117 J |

  - Optimised shear for the distal pedicel (response-surface design, 1–5 mm/s, 30–90° blade angle):

    | Diameter | Shear force |
    |---|---|
    | 3 mm | 19.6 N |
    | 3.5 mm | 26.7 N (validated 28.8 N) |
    | 4 mm | 33.0 N (validated 32.8 N) |
    | 4.5 mm | 34.2 N |
    | 5 mm | 30.9 N |

  - Three-point bending of the abscission zone over a 12 mm span, n=5 per group:

    | Diameter | Break force | Apparent modulus |
    |---|---|---|
    | 5–6 mm | 22.4 ± 6.4 N | 7.1 MPa |
    | 6–7 mm | 37.4 ± 4.7 N | 3.8 MPa |
    | 7–8 mm | 41.6 ± 9.9 N | 2.8 MPa |

    Fitted line: F = −21.36 + 8.71·D (D in mm).

    ⚠️ This short-span test over a joint does not give a usable stem Young's modulus.
  - Moisture content about 76.5 %.
  - Source: [Weng et al. 2024, Agronomy 14:2274](https://www.mdpi.com/2073-4395/14/10/2274), DOI 10.3390/agronomy14102274
- **Cherry tomato, cv. "No.1 Hybrid" (Xie et al. 2021).** Fruit-stalk diameter 2.10 ± 0.53 mm (probably the pedicel, not the peduncle); "calyx branch" length 71.4 ± 4.0 mm; n=170. — [Xie et al. 2021](https://www.mdpi.com/2077-0472/11/9/815)
- **Analogue, strawberry peduncle, not tomato.** A grip of up to 10 N held the peduncle without crushing it; about 15 N cut it with a blade of 16.6° wedge angle at 30°. — [Rajendran et al., arXiv 2207.12552](https://arxiv.org/abs/2207.12552) (abstract only)
- **Peduncle skin abrasion.** In a robot study that pinched harvested truss tomatoes by the peduncle with foam-covered fingers, repeatedly grasping and releasing the same trusses caused "abrasion damage … on the skin of the peduncles". — [van den Bent et al., ICRA 2024 / arXiv 2309.17170](https://arxiv.org/abs/2309.17170), DOI 10.1109/ICRA57147.2024.10610089 (checked in the full text)

### Inferences
- **Peduncle crush risk at 4 N.** The current grip (about 4 N per pad on a 3–5 mm stalk) is an order of magnitude below the large-fruit bending-break forces (20–40 N) and below the strawberry "no crush at 10 N" figure. Outright crushing is therefore unlikely. Skin abrasion from repeated grasps is documented.
- **Cutting force for a cherry peduncle.** Scaling Weng's distal-pedicel shear values gives roughly 20–35 N for a 3–5 mm stalk, but this is an extrapolation from a different cultivar and tissue. Measuring it is bench test B6-f.

### Gaps
- Not found: for cherry-truss peduncles and rachis, diameter, bending stiffness (EI), Young's modulus, density, compressive crush force or pressure.
- Not found: a blade-specific cutting force for a cherry peduncle.
- Unread (paywalled): Liu, Li & Li 2021, a Springer chapter on tomato fruit and stem properties, which may contain some of this.

## B3. Truss mass, fruit count, fruit mass, pedicel length, truss length (cherry)

### Takeaway
The one measured cherry truss found weighs 168 ± 16 g with 15 ± 3 fruit, about twice the simulation's 70–100 g. Cherry fruit mass ranges from about 5 g to 19 g depending on cultivar. Use your own trusses' measured values (B6-e) rather than either number.

### Cited Findings
- **Cherry, commercial high-wire greenhouse, cultivar not stated (Singh et al., preprint, Table 1).** n not stated.
  - Truss weight **168.06 ± 15.50 g**; **15 ± 3** fruits per truss.
  - Fruit diameter 30.25 ± 1.30 mm, height 26.10 ± 1.12 mm.
  - Pedicel length 43.52 ± 6.44 mm.
  - Source: [arXiv 2610.18160](https://arxiv.org/abs/2610.18160)
- **Cherry, cv. "No.1 Hybrid" (Xie et al. 2021, n=170).** Diameter 29.8 ± 5.1 mm, height 34.6 ± 5.6 mm, mass about 18.75 g (the paper prints the unit as "mm"). — [Xie et al. 2021](https://www.mdpi.com/2077-0472/11/9/815)
- **Small cherry, light-red stage, greenhouse (Kabas et al. 2023, n=15).** Mass 4.82 ± 1.34 g (range 2.48–7.36 g); geometric mean diameter 18.96 mm. — [Kabas et al. 2023, Appl. Sci. 13:8906](https://www.mdpi.com/2076-3417/13/15/8906), DOI 10.3390/app13158906
- **Large-fruit cluster, for contrast (Kondo et al. 2010).** About 400 g; peduncle 97–227 mm long. — [Kondo et al. 2010](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_article)

### Inferences
- The simulation's 70–100 g fits a truss of about 6–8 medium cherry or cocktail fruits of 10–13 g. A 15-fruit commercial cherry truss would double the gravity torque in A1, making the pivot and spin margins worse.

### Gaps
- Unverified, search snippet only: Gao et al. 2024 report bunch length 69.3 ± 5.0 mm, peduncle 45.8 ± 7.5 mm, pedicel 17.3 ± 1.1 mm, fruit 10–18 g.
- Not found: cocktail-truss data; truss centre-of-mass position.

## B4. Friction of tomato peduncle and fruit skin against pad materials

### Takeaway
**No published friction coefficient exists for a tomato peduncle or stem on any material**, and none for tomato on silicone, TPU or PLA. The only measured values are whole fruit on tilted plates: about 0.2–0.5 on steel and aluminium and about 0.3–0.57 on rubber. These are bulk static values that vary by cultivar. **The simulation's 1.5 (pad) and 0.9 (stalk) are unsupported** and should come from bench test B6-b.

### Cited Findings
- **Cv. "Tompler", about 25 g fruit, about 35 mm (cocktail size) (Jahanbakhshi et al. 2019).**
  - Static μ on a tilting table, measured with a fruit-filled bottomless box (bulk, not single fruit):

    | Surface | μs |
    |---|---|
    | Galvanised steel | 0.249 ± 0.020 |
    | Aluminium | 0.212 ± 0.018 |
    | Rubber | 0.305 ± 0.019 |
    | Wood | 0.364 ± 0.020 |

  - ⚠️ The same paper reports an implausible density (0.47 g/cm³) and modulus (0.11 GPa), so treat it with care.
  - Source: [Jahanbakhshi et al. 2019, Food Sci. Nutr. 7:1473](https://pmc.ncbi.nlm.nih.gov/articles/PMC6475739/), DOI 10.1002/fsn3.986
- **Three large tomato cultivars, 110–182 g (Mahmoud et al. 2022).**
  - Static μ on a tilted plane, 100 fruits per cultivar:

    | Cultivar | Stainless steel | Rubber |
    |---|---|---|
    | Nesma | 0.488 | 0.566 |
    | Masa | 0.277 | 0.310 |
    | "2020" | 0.30 | 0.388 |

  - Source: [Mahmoud et al. 2022, CIGR Journal 24(4)](https://cigrjournal.org/index.php/Ejounral/article/download/7083/3959/38261) (no DOI found)
- **Silicone pad μs = 1.34**, cited second-hand by Zheng 2022 from Zhou et al. 2022, IJABE, DOI 10.25165/j.ijabe.20221502.6672. **Not verified**, and it is not clear what surface it was measured against. — [Zheng et al. 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.974456/full)

### Inferences
- Stem skin is drier and harder than fruit skin, and the cut peduncle may be wet with sap. Fruit values cannot be carried over to the stalk, so measure μ for stalk-on-pad, dry and wet.
- Because sliding μ sets both the pivot and spin breakaway torques (A1), it is the single most important number to measure.

### Gaps
- Not found: stem or peduncle friction on any material; kinetic friction; values for silicone, TPU or PLA.
- Not read: Li, Li & Liu 2011, J. Food Eng., DOI 10.1016/j.jfoodeng.2010.10.013, and Polat 2007 (cherry cv. Forme), both on fruit friction.

## B5. Bruise thresholds for cherry tomato fruit

### Takeaway
Cherry fruit at the light-red stage ruptures at about 23 N under a small probe (Kabas et al. 2023). Large-tomato skin punctures at 4.5–9 N under a 2 mm needle, dropping as the fruit ripens. One drop study found 100 cm onto concrete was the critical height for internal damage, with no visible damage. **No cherry-specific impact-energy or drop-height threshold was found.**

### Cited Findings
- **Cherry, light-red stage, n=15 (Kabas et al. 2023).**
  - Rupture force **23.38 ± 1.78 N** (20.8–27.6 N); firmness 4.97 N/mm; elastic modulus 2.59 ± 0.51 N/mm²; Poisson's ratio 0.29; deformation energy 56.3 N·mm.
  - Method: curved-end cylindrical probe at 10 mm/min.
  - Source: [Kabas et al. 2023, Appl. Sci. 13:8906](https://www.mdpi.com/2076-3417/13/15/8906), DOI 10.3390/app13158906
- **Large tomato, cv. Syngenta Sibede, 150–170 g, n=14 per stage (Zheng et al. 2023).**

  | Ripeness stage | Puncture force, 2 mm needle | Whole-fruit radial compression rupture |
  |---|---|---|
  | Green-ripe | 9.22 N | — |
  | Semi-ripe | 6.00 N | 109.7 N |
  | Early firm-ripe | 5.41 N | 98.9 N |
  | Mid-late firm-ripe | 4.52 N | 85.3 N |

  - Source: [Zheng et al. 2023, Agriculture 13:2201](https://www.mdpi.com/2077-0472/13/12/2201), DOI 10.3390/agriculture13122201
- **Drops onto concrete, tomato type not stated (Sophea et al. 2024).**
  - Heights 0, 10, 30, 50 and 100 cm, 100 fruit in total.
  - No visible damage straight after the drop.
  - 100 cm was the "critical level" for membrane damage at day 14.
  - Source: [Sophea et al. 2024, Biomolecules 14:1012](https://pmc.ncbi.nlm.nih.gov/articles/PMC11352672/), DOI 10.3390/biom14081012
- **Large cv. Pintaro.** Skin damage at 9.8 N with suction and clamping combined. — [Yang et al. 2026](https://journal.hep.com.cn/fase/EN/10.15302/J-FASE-2025643)

### Inferences
- Rupture thresholds are tens of newtons, far above anything in a 4 N stalk grip, and fruit is not gripped. The realistic fruit-damage route is fruit hitting the crate or each other while carried, where drop or impact thresholds apply, and no cherry-specific one was found.

### Gaps
- Unverified, snippet only: a finite-element estimate of about 237 mm critical drop height for cherry (Kabas et al. 2008, J. Sci. Food Agric., DOI 10.1002/jsfa.3246); "15.2–23.8 N damage force depending on variety" (Kabas & Ozmerzi 2008, J. Texture Stud. 39:199–209).
- Not found: cherry force at 5–10 % deformation; impact energy for bruising.

## B6. Simple bench tests for the missing quantities

### Takeaway
Six bench tests, using a kitchen scale, known weights (coins or a cup of water), calipers, a phone and printed jigs, can measure every number the grip model needs:

- **a.** Pivot and spin breakaway torque.
- **b.** Stalk-on-pad friction.
- **c.** Pad indentation and contact width.
- **d.** Fruit pull-off force.
- **e.** Truss mass and centre of mass.
- **f.** Stalk crush and cut force.

Test **a** decides whether real pads pivot, and directly calibrates μ and μ_t in the simulation. It must use the real pads at the real clamp force.

### Cited Findings
- Tilted-plane static friction (μs = tan α) is the standard method in the tomato friction papers above. — [Jahanbakhshi et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6475739/); [Mahmoud et al. 2022](https://cigrjournal.org/index.php/Ejounral/article/download/7083/3959/38261)
- Quasi-static pull tests at a few mm/s, with the break location recorded, are used in the detachment papers (3 mm/s in Xie 2021; 1 mm/s in Weng 2024). — [Xie et al. 2021](https://www.mdpi.com/2077-0472/11/9/815); [Weng et al. 2024](https://www.mdpi.com/2073-4395/14/10/2274)

### Inferences (proposed procedures; not taken from a source)

**General.**
- Use freshly cut trusses of the cultivar that will actually be handled. Note the hours since cutting, and test 5 stalks × 5 repeats per condition.
- Measure stalk diameter at the grip point with calipers, in two directions.
- Weigh the real pads' clamp force once: push each pad with the kitchen scale (a printed pusher on the scale pan) and read the force at the start of travel (preload) and at full travel.

**a. Breakaway torque for pivot and spin (the key test).**
- Set-up: fix the real jaw, or a printed copy holding the real pads and springs, in a vice. Clamp a stalk segment as the gripper would.
- **Pivot** (about the pad normal): tape or clip a light straight lever (a printed bar or a bamboo skewer) to the stalk, perpendicular to it and in the pad plane. Hang a cup at a known distance d (40–80 mm) and add water in 5 g steps, about 5 s apart.
- **Spin** (about the stalk axis): clip a small printed crank (a split collar with an arm) onto the stalk end and hang the cup from it.
- Record:
  - the mass m at which rotation starts, read by a marker line against a printed protractor and filmed with the phone; breakaway torque T = m·g·d;
  - the angle against time for 30 s at about 50 % and 80 % of m, to check whether real pads creep;
  - clamp force, pad material, stalk diameter, dry or sappy, and any marks on the stalk afterwards.
- Repeat at preload-only and full-travel clamp force, and with a 4 mm wooden or PLA dowel as a control.
- **Use in the simulation:**
  - Spin gives μ ≈ T_spin / (2·F·r).
  - Pivot then gives the pressure-distribution factor k in T_pivot = k·μ·F·L, which ranges from k = 1/2 for a uniform line contact to k = 1 for two end points (A1).
  - Run the same load case in simulation with noslip on and compare breakaway torques.

**b. Stalk-on-pad friction.**
- **Tilt method:** glue pad material to a board. Lay 3 parallel stalk segments under a small printed sled loaded with 50–200 g. Raise one end slowly and read the angle α at first slip with a phone inclinometer; μs = tan α.
- **Pull method:** pull the sled horizontally by a thread over a pulley into a cup, adding water until it slides; μ = cup weight / sled weight.
- Test dry and wet (a sap-smeared pad), and each candidate pad material.

**c. Pad indentation and contact width.**
- Press a stalk onto the pad with known weights (100–400 g, about 1–4 N).
- Measure the indentation with calipers or a dial gauge to get pad stiffness, which sets the segment springs in A3.
- Colour the stalk with marker or lipstick, press it onto paper laid over the pad, and measure the width of the print. This is the contact-patch width that μ_t should reflect.

**d. Fruit pull-off.**
- Hold the truss by its peduncle in the clamp. Loop thread around one fruit (or use a small sling) and hang a cup from it, axially along the pedicel, then at 90°. Add water until the fruit detaches; weigh the cup on the kitchen scale.
- Record force, break location (knuckle/abscission zone, calyx, or pedicel-rachis junction), ripeness stage (colour chart), and fruit mass.
- 20+ fruit per ripeness stage.

**e. Truss mass and centre of mass.**
- Weigh each truss.
- Hang it from the grip point on a thread at two different orientations, photograph it with a plumb line, and intersect the lines to find the centre of mass.
- The centre-of-mass offset × weight is the gravity torque used in A1.

**f. Crush and cut.**
- **Crush:** squeeze a stalk between two pad samples on the kitchen scale with increasing weights. Measure diameter before and after, and inspect after 1 h and 24 h for flattening or browning. Record the load at the first visible damage.
- **Cut:** press the actual blade through the stalk on the kitchen scale (slowly, by hand or with a screw jig) and record the peak reading.
- **Bending stiffness (optional):** clamp a stalk as a cantilever, hang weight W at length l and measure tip deflection δ. Then EI = W·l³ / (3·δ).

### Gaps
- None of these tests has been run.
- Kitchen-scale resolution (1 g ≈ 0.01 N) is fine for B6-b to B6-e.
- For the breakaway torque in B6-a, water-step size and how quickly weight is added will affect the onset reading. Add slowly and repeat.
