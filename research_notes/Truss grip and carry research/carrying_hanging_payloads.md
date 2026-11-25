# Carrying a hanging, pinch-held payload (truss) from cut to crate with a 5-joint arm, and post-cut handling in harvesting robots

Research date: 2026-11-25. Evidence tags on every cited bullet:
- **[HW]**: result measured on real hardware.
- **[SIM]**: simulation only.
- **[VENDOR]**: vendor or press claim, not independently checked.
- **[SNIPPET]**: taken from a search-result summary or abstract. I could not open the full text, so check it before relying on it.
- **[FETCHED]**: I read the full text or abstract page myself.

Everything under "Inferences" is my own reasoning and is not sourced. Each "Maps to" note turns a finding into via-poses, pose-selection rules, speed and acceleration caps, or crate placement for the 5-joint arm (pan, lift, elbow, wrist flex, wrist roll, jaw). Some methods generate paths (TSR/CBiRRT, trajectory optimisation, MPC). These are flagged, and only the pose filter or acceleration cap is taken from them.

---

## 1. Gravity-aligned transport: keeping the gripper's orientation relative to gravity constant from grasp to release

### Takeaway
The constrained-planning literature (Task Space Regions, constraint manifolds, waiter and upright-transport problems) solves "keep the object's orientation constant" by generating paths on a constraint manifold. That violates the hard constraint here. My own kinematic inference (unsourced; check against the MJCF joint axes) is that on this arm the constraint is almost free to express. Pan turns about the vertical axis, so gravity seen from the gripper should depend only on the summed pitch (lift + elbow + wrist flex) and on wrist roll. If both are held at one value at the cut pose, every via-pose and the release pose, the truss has no new gravity torque and no reason to pivot in the grip. Cut and release candidates should then be filtered so that one shared (pitch, roll) pair reaches both.

### Cited Findings
- **TSR / CBiRRT.** Pose constraints such as "keep upright" are expressed as Task Space Regions: a bounded box of allowed positions and orientations around a reference frame. CBiRRT2 builds a configuration-space path that stays on the constraint manifold by projecting samples onto it. It is probabilistically complete. Berenson, Srinivasa & Kuffner, "Task Space Regions: A framework for pose-constrained manipulation planning", IJRR 2011 — [CMU PDF](https://www.ri.cmu.edu/pub_files/2011/10/dmitry_ijrr10-1.pdf), [Brown copy](https://cs.brown.edu/courses/csci2951-k/papers/berenson12.pdf). [SNIPPET] **Path-generating.**
- **Why projection is needed.** Randomised planners almost never sample points that lie exactly on an orientation-constraint manifold, so samples must be projected onto it. Stilman, "Task constrained motion planning in robot joint space" — [CMU PDF](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2006_4/stilman_michael_2006_4.pdf). [SNIPPET] **Path-generating.**
- **Grasped objects and orientation constraints.** "Both grasped objects and forced orientation impose constraint manifolds in the robot configuration space." Learned planners (CoMPNet) target these manifolds. Qureshi et al., "Neural Manipulation Planning on Constraint Manifolds" — [arXiv 2008.03787](https://arxiv.org/pdf/2008.03787). [SNIPPET] **Path-generating.**
- **Waiter problem.** A time-optimal solution to the "waiter motion problem" (objects on a tray that must not slide) on an industrial robot. Letting the tray tilt shortens cycle time further. JKU Linz — [research.jku.at](https://research.jku.at/en/publications/a-time-optimal-solution-for-the-waiter-motion-problem-with-an-ind/). [SNIPPET] **Path-generating** (time-optimal path following).
- **Upright transport with MPC.** A mobile manipulator balances up to seven objects while avoiding obstacles, and a robust variant handles uncertain inertial parameters. The search snippet describes two separate works from the same group: (a) the MPC work on a mobile manipulator balancing up to seven objects, with code at [github.com/learnsyslab/upright](https://github.com/learnsyslab/upright); and (b) a robust planner for uncertain inertial parameters by Heins & Schoellig, RA-L 2023 — [PDF](https://www.dynsyslab.org/wp-content/papercite-data/pdf/heins-ral23.pdf). Which PDF holds which result was not verified. [SNIPPET] **Path-generating** (MPC / optimisation).
- **Pivoting as a virtual joint with the object hanging vertical.** One real-robot system models the pivoting object as a virtual joint between fingertips and object. A high-priority constraint minimises the angle between gravity and the grasp-point-to-centre-of-mass vector, so "the grasped object is always vertical during the planning process". Costanzo et al., "Manipulation Planning and Control for Shelf Replenishment", IEEE RA-L 2020 — [arXiv 1912.11018](https://arxiv.org/pdf/1912.11018). [FETCHED][HW] **Path-generating** (eTaSL constraint-based).
- **Same paper, simulation table.** Many start/goal object-angle pairs had no collision-free solution unless the object was allowed to pivot. With pivoting allowed, the planner picked the free angles itself, for example 0.35 rad and −1.38 rad. Costanzo et al. 2020 — [arXiv 1912.11018](https://arxiv.org/pdf/1912.11018). [FETCHED][SIM]
- **Arms with fewer than 6 DOF.** I found no paper on orientation-constrained transport for 5-DOF arms. See Gaps.

### Inferences
- **Gravity in the gripper frame on this arm** (own kinematics; check the URDF joint signs). Shoulder lift, elbow and wrist flex have parallel horizontal axes, pan is vertical, and wrist roll lies along the gripper axis. The gripper orientation is therefore R = Rz(pan) · Ry(φ) · Rx(roll), with φ = ±lift ± elbow ± flex + offset. Gravity in the gripper frame is Rᵀ·g = Rx(roll)ᵀ · Ry(φ)ᵀ · (0, 0, −g).
  - This does not depend on pan.
  - Holding φ and roll constant therefore keeps the truss's gravity load in the grip exactly constant, while pan, reach and height stay free: three position DOF at a fixed tilt.
  - With φ fixed, the lift/elbow/flex chain acts as a 2-link arm whose wrist point is offset by the fixed gripper vector. The fixed-tilt workspace is the 2R annulus shifted by that vector, so it is smaller than the full workspace, but it is a well-defined region in (radius, height).
- **Maps to via-poses.** Define one "carry tilt" (φ_c, roll_c). Then:
  1. At the cut pose, φ = φ_c and roll = roll_c.
  2. Every via-pose (retreat, lift, swing-over, above-crate, lowered) has the same φ_c and roll_c.
  3. The release pose has the same φ_c and roll_c.
  4. If the truss must later be tilted for lay-down (section 5), do it only after the lowest fruit touches the crate floor, never in free air.
  5. Interpolate in joint space. Tilt φ is a linear (signed) sum of joint angles, so synchronised linear joint interpolation between two via-poses with the same φ keeps φ exactly constant along the segment. The same holds if every joint passes through the same linear filter or shaper (section 3). No per-tick IK is needed. The real risks are unsynchronised per-joint profiles (different durations or velocity limits per joint) and servo tracking lag that differs between lift, elbow and flex. Check commanded and measured φ on hardware.
- **Maps to pose selection.**
  - Reject any (cut candidate, release candidate) pair that is not reachable at a common (φ, roll). Keep a short list of 2–4 allowed carry tilts, for example pads vertical with the gripper horizontal, or the gripper pitched 30° down. Accept a scene only if, for one tilt, the cut pose, all via-poses and one release candidate are inside the fixed-tilt workspace.
  - The cut candidates are then limited by that tilt. This is the "choose the grasp so the transport orientation is reachable all the way" rule.
- **What needs a 6th joint (flag).** With φ and roll fixed, the gripper's yaw in the world is set by pan. The pad normal, which is the truss's pivot axis and fixes its swing plane, therefore rotates with pan. Two things cannot be done on this arm with constant gravity alignment:
  - holding a full fixed world-frame orientation during a pan carry;
  - choosing the swing plane relative to the travel direction or to the crate wall independently of where the crate is.

  Both need a vertical-axis wrist yaw (a 6th joint). On 5 joints the release-time swing plane is decided by the crate's pan angle, so it is a layout choice (section 4). Changing roll to re-aim the swing plane breaks gravity alignment and makes the truss pivot. Do it only with the truss resting on the crate floor.
- **Pan is the safest carry axis.** A pure pan rotation leaves gravity in the gripper frame unchanged. It does add centripetal (ω²r) and tangential (αr) acceleration, which are handled in section 3. If the crate's drop point is at roughly the same radius and height as the cut points, the carry becomes "retreat radially, pan, lower" with only pan moving through the large angle.

### Gaps
- No source on orientation-constrained carrying with arms under 6 DOF, or on choosing grasps so that a fixed tool tilt is reachable from grasp to place. The kinematic argument above is my own and should be checked against the actual joint axes and signs in the MJCF/URDF.
- TSR/CBiRRT, waiter and upright-MPC methods all generate paths. I found no published "fixed via-pose plus constraint-check only" variant. Only the pose-filter idea carries over.

---

## 2. Modelling a pinch-grasped object as a pendulum or pivoting body for planning

### Takeaway
Pivoting-grasp papers agree that a pinch-held object with low torsional friction behaves like a pendulum: it rotates about the grasp axis until its centre of mass lies straight below the grasp point. The real-robot planning paper models this with a virtual joint and a constraint keeping the object vertical. For planning, the truss should therefore be modelled as hanging straight down from the pinch, centre of mass below the pad contact and free about the pad-normal axis, plus a swing envelope. It should not be modelled as rigidly fixed at its pre-cut angle. My own inference, not tested: that rigid assumption is the likely cause of the observed 3–20 cm planner errors.

### Cited Findings
- **Hanging equilibrium.** "During the gripper pivoting, the object behaves like a pendulum and stays in its equilibrium orientation, i.e., the vector pointing from the grasp point to the center of gravity (CoG) is aligned with gravity… If it is not above the CoG, the object will rotate so that the CoG goes to the equilibrium point below the grasp point." Costanzo et al., IEEE RA-L 2020 — [arXiv 1912.11018](https://arxiv.org/pdf/1912.11018). [FETCHED][HW]
- **Deviation after a motion sequence with mode switching.** Sensorised fingers ran slipping avoidance during translations, with the pivoting mode switched on automatically during rotations about the pivot axis. After the motion sequence (12 repetitions), the angle between object and fingers deviated by:
  - mean 0.104 rad (slow) and 0.112 rad (fast);
  - maximum 0.181 rad and 0.194 rad, about 10–11°.

  The authors call deviations under 0.2 rad acceptable. Costanzo et al. 2020 — [arXiv 1912.11018](https://arxiv.org/pdf/1912.11018). [FETCHED][HW]
- **Centre of mass from wrench.** The paper estimates the CoG offset from the grasp point by changing the gripper orientation and reading the measured wrench. A grasp exactly at the CoG cannot pivot because the gravity torque is zero. Costanzo et al. 2020 — [arXiv 1912.11018](https://arxiv.org/pdf/1912.11018). [FETCHED][HW]
- **Gravity-driven slip, sliding-mode control.** Gravity-driven in-hand slip is regulated with a sliding-mode controller, with the tool modelled as a link attached at the pinch. Viña et al., IROS 2015 — [KTH PDF](https://www.csc.kth.se/~fevb/publications/pdfs/iros15.pdf). [SNIPPET]
- **Torsional friction for pivoting.** Pivoting by modulating pinch force, with a soft-finger torsional friction model and the torsional friction coefficient adapted online from tactile sensing. Viña et al., ICRA 2016 — [KTH PDF](https://www.csc.kth.se/~fevb/publications/pdfs/icra16.pdf) (link returned 404 on fetch). [SNIPPET]
- **Tactile gravitational pivoting.** Gravitational pivoting on a Robotiq 2F-85 with tactile sensors, using an LSTM to predict pose and a dataset of 10 household objects. Toskov et al., CoRL 2022 — [PMLR PDF](https://proceedings.mlr.press/v205/toskov23a/toskov23a.pdf) / [arXiv 2210.05068](https://arxiv.org/pdf/2210.05068). [SNIPPET][HW]
- **Pivoting by lifting until slip.** Objects are pivoted by lifting so gravity causes slip, managed by adjusting gripper width, using force/torque, RGB-D and touch. [arXiv 2303.10865](https://arxiv.org/pdf/2303.10865). [SNIPPET]
- **Pivot angle from motor current (simulation only).**
  - Method: estimate the centre-of-mass direction of a pinched object from wrist-motor current during a slow wrist sweep.
  - Results: 0.009 rad RMS over 100 randomised objects; closed-loop pivot error 0.031 rad RMS over 40 objects. An open-loop schedule with nominal parameters gave 0.413 rad.
  - Side effect: wrist motion with a loose grip "ratchets the object downward".
  - Status: "All results are from simulation" (MuJoCo, soft-finger contact). Not peer-reviewed.

  Aidoo, engrXiv preprint, 2026-10-12, [doi:10.31224/8492](https://doi.org/10.31224/8492) — [engrXiv](https://engrxiv.org/preprint/view/8492). [FETCHED][SIM]
- **Larger CoG offset widens the pivot force window.** Grasping farther from the centre of gravity lengthens the moment arm, which widens the range of grip forces compatible with pivoting. MIT two-phase gripper — [DSpace PDF](https://dspace.mit.edu/bitstream/handle/1721.1/107782/A%20two-phase%20gripper.pdf?sequence=1&isAllowed=y). [SNIPPET]
- **Calibrated pendulum model for swinging packages.** Packages held by suction are modelled as an analytical pendulum. The effective length is calibrated from motion capture and predicted for new packages by Gaussian-process regression, with an optional uncertainty margin (mean ± κ·√Var). The model predicts the steady-state offset under acceleration and the peak swing per acceleration segment. Shukla, Moode, Wang, Mayya & Gupta, "Robot Trajectory Optimization for Safe Transport of Deformable Packages", CASE 2025 — [PDF](https://cdn.amazon.science/b3/58/2f1438b34288a50ec488ef3b3df6/case25-0617-fi.pdf). [FETCHED][HW]
- **Cart-pendulum model on a delta robot.** A controlled cart-pendulum-mass model approximates a delta robot carrying suctioned products (chicken fillets). Van der Kruk, Bindels, Bruyninckx & van de Molengraft, "Automatic control for swing-free control of suctioned products in robotic pick-and-place operations", Robotics and Autonomous Systems 184:104863, 2025, [doi:10.1016/j.robot.2024.104863](https://doi.org/10.1016/j.robot.2024.104863) — [TU/e page](https://research.tue.nl/en/publications/automatic-control-for-swing-free-control-of-suctioned-products-in/). [FETCHED][HW]
- **Double-pendulum crane models.** Modelling a crane as a double pendulum allows control of both position and tilt of the load, which matters when the payload is extended rather than a point. [Tower-crane double pendulum](https://www.researchgate.net/publication/224120075_Control_of_Tower_Cranes_With_Double-Pendulum_Payload_Dynamics); [bridge-crane double pendulum](https://www.researchgate.net/publication/245373387_Input_Shaping_Control_of_Double-Pendulum_Bridge_Crane_Oscillations). [SNIPPET]

### Inferences
- **Why the truss pivots 55–85°** (order-of-magnitude estimate, own numbers; please measure).
  - Grip side, assumed values: 4 N pad force, a 3–5 mm round stalk giving a narrow line contact (effective torsional radius ~1–2 mm), and μ ≈ 0.5–1. Torsional friction about the pad normal is then roughly τ_f ≈ 2 · μ · F · r ≈ 2 · 0.7 · 4 N · 0.0015 m ≈ 0.008 N·m.
  - Load side: a 0.085 kg truss with its centre of mass d = 5–8 cm from the pinch gives a gravity torque of m·g·d·sinψ, up to 0.04–0.067 N·m.
  - Result: the grip holds only while sinψ < τ_f/(m·g·d) ≈ 0.12–0.2, so about ±7–12° from hanging straight down. Larger offsets slip until the truss hangs. In practice the stalk in a flat-pad pinch is a hinge about the pad normal, which fits the 55–85° pivots observed when gripper tilt changes.
  - Loosely consistent with the literature: Costanzo et al. measured 6–11° residual deviation after a motion sequence that included deliberate pivoting phases, using tactile slip control. That is not a direct measure of passive torsional holding.
- **Planner model to adopt.**
  - Attach the truss to the gripper by a passive hinge whose axis is the pad normal (jaw closing direction) through the pad contact. Its angle is set by gravity: the centre of mass lies straight below the pinch, in the plane perpendicular to the hinge.
  - Recompute the truss pose at every via-pose from that rule, not from the cut-time relative pose.
  - Add a swing cone of ±θ_swing about the hanging direction. θ_swing comes from the acceleration caps in section 3, for example 10°. Inflate the fruit collision geometry by ℓ·sin(θ_swing), which is about 3 cm for ℓ = 17 cm at 10°.
  - This is the virtual-joint idea from Costanzo et al. used only as a collision-check model, not to generate motion.
- **Maps to pose selection.**
  - Prefer cut candidates where the truss already hangs at its equilibrium relative to the pads at the moment of the cut. With (φ, roll) held constant afterwards, nothing changes, so the pre-cut and post-cut relative poses agree.
  - A cut pose with a large equilibrium offset (pre-cut truss angle far from the vertical plane through the pinch) will swing by that offset at the instant of the cut. Add a short settle wait before retreat, or reject that candidate.
- **Maps to the jaw (hardware note, not a motion fix).** Pads that are longer along the stalk, or two separated contacts along the stalk, raise torsional resistance a lot. Wider contact means more torque per newton. This only helps on real hardware if the stalk is straight over the pad length. It is noted here because it changes the "hinge" assumption; it is not researched further.
- **Swing cone instead of rigid truss for the 13 % no-path scenes.** A hanging truss with a 10° cone is compact under the gripper. A rigidly fixed truss at a 55–85° stale angle sticks out sideways. Some of the scenes with no collision-free carry may be artefacts of the rigid model. Re-run the planner check with the hanging model before concluding they are infeasible.

### Gaps
- No paper measures torsional friction of a plant stalk (peduncle) in a flat-pad pinch. The τ_f above is an estimate.
- No paper models a cluster (stalk plus fruit on pedicels) as a multi-pendulum carried by a robot. The crane double-pendulum results are the nearest analogue.
- The ICRA 2016 KTH PDF returned 404, so its torsional-friction numbers are unverified.

---

## 3. Swing suppression for suspended loads: shaping, smooth profiles, and acceleration caps vs pendulum length (5–12 cm)

### Takeaway
At 5–12 cm the truss is a 1.4–2.2 Hz pendulum, so the period is 0.45–0.70 s, which is 13–21 control ticks at 30 Hz. The simplest robust recipe for a scripted joint-space mover has three parts:
1. Cap Cartesian acceleration of the pinch point so the steady lean tan⁻¹(a/g) stays a few degrees. This is the guarantee.
2. Smooth the commands with a jerk ramp, or a moving-average (FIR) window, lasting at least one pendulum period. Note: a constant-acceleration phase longer than a period does not by itself remove residual swing (see Inferences).
3. Optionally apply a ZVD input shaper tuned per scene to the truss's estimated equivalent length.

A plain ZV shaper is too fragile across the 1.55× frequency spread of this truss set. Real-hardware papers show that bounding the peak swing angle per acceleration segment is enough to plan safe transport.

### Cited Findings
- **Constant-acceleration segments with swing bounds.**
  - Trajectories were built from piecewise-constant acceleration segments. The pendulum model predicts swing displacement and velocity at segment boundaries, and the optimiser bounds them.
  - In real-robot tests up to 1.8 m/s and 4.0 m/s², with peak swing tracked by motion capture, trials with peak angle < 8° "seldom detached" and swings > 16° were strongly correlated with detachment. This is a suction-detachment threshold for packages, not a truss threshold.
  - The optimised trajectories cut transport time by up to 18 % against a conservative baseline with no drops.

  Shukla et al., CASE 2025 — [PDF](https://cdn.amazon.science/b3/58/2f1438b34288a50ec488ef3b3df6/case25-0617-fi.pdf). [FETCHED][HW] **Path-timing optimiser** (it optimises timing along a given path, so the bounding idea carries over to fixed via-poses).
- **ZV shaper on a delta robot.** Of the shapers tested, Zero Vibration "demonstrates the best overall performance for realistic variations (such as weight and pick-up pose), well within 10% of the variations of the natural frequency". The design procedure uses the ratio of vibration time to move time and finds the highest acceleration that keeps placement within tolerance. A passive-rotation gripper is also used. Van der Kruk et al., RAS 2025 — [TU/e](https://research.tue.nl/en/publications/automatic-control-for-swing-free-control-of-suctioned-products-in/). [FETCHED][HW]
- **Move time vs vibration period.** For fast pick-and-place of swinging products, input shaping significantly reduces overshoot "without settling time loss, when the move time equals more than 60 percent and less than 80 percent of the natural vibration time". ZV gave the best overall performance. [EasyChair preprint](https://1login.easychair.org/publications/preprint/hl7N3/open). [SNIPPET]
- **ZV and ZVD shapers defined.**
  - ZV convolves the command with two impulses separated by half the natural period. ZVD uses three impulses, each half a period apart, so ZVD lasts one full period.
  - 5 % insensitivity: I_ZV = 0.06 and I_ZVD = 0.28. Residual vibration stays under 5 % only while the true frequency is within 0.97–1.03 (ZV) or 0.86–1.14 (ZVD) of the design frequency.
  - Hardware result: up to 72 % lower peak-to-peak velocity error on a real cable-driven robot.

  Baklouti, Courteille, Lemoine & Caro, "Input-Shaping for Feed-Forward Control of Cable-Driven Parallel Robots", ASME J. Dyn. Sys. Meas. Control 2020, [doi:10.1115/1.4048354](https://doi.org/10.1115/1.4048354) — [arXiv 2010.11676](https://arxiv.org/pdf/2010.11676). [FETCHED][HW]
- **Shaped inputs on a manipulator carrying a pendulum.** Preshaped reference inputs gave a swing-free stop in simulation and experiment on a hydraulic manipulator carrying a compound pendulum. "On preshaped reference inputs to reduce swing of suspended objects transported with robot manipulators", Mechatronics — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957415899000926). [SNIPPET][HW]
- **Input shaping on a scaled crane.** Average peak swing fell from 13.1° to 10.7° (−18 %) during slewing on a scaled crane. [arXiv 2512.11228](https://arxiv.org/pdf/2512.11228). [SNIPPET][HW]
- **FIR and harmonic smoothers.** FIR (moving-average-type) filters for online minimum-time trajectories with residual-vibration suppression: Biagiotti & Melchiorri, Control Engineering Practice 2012 — [preprint](https://iris.unimore.it/bitstream/11380/903689/1/FIR4Optimaltrajectories08CEPSingleColumnRevised_bib.pdf). A "damped harmonic smoother" that cancels a known oscillation frequency and damping: IEEE TCST 2020 — [PDF](https://iris.unimore.it/bitstream/11380/1168970/1/Damped%20Harmonic%20Smoother%20for%20Trajectory%20Planning%20and%20Vibration%20Suppression.pdf). [SNIPPET]
- **Old swing-damping patent.** A patent describes cosine and cubic-spline acceleration profiles for oscillation-damped stops of suspended loads, and notes that "any initial oscillation of the swinging object will, in general, result in residual oscillation". [US4997095](https://patents.google.com/patent/US4997095). [SNIPPET]

### Inferences
- **Pendulum numbers** (own computation, point-mass T = 2π√(L/g); a uniform truss hanging from its top end has L_eq ≈ ⅔ of its length, so 10–17 cm gives L_eq ≈ 6.7–11.3 cm):

  | L_eq | Period T | Frequency | Half-period (ZV delay) | At 30 Hz |
  |---|---|---|---|---|
  | 5 cm | 0.45 s | 2.23 Hz | 0.22 s | 7 ticks |
  | 8 cm | 0.57 s | 1.76 Hz | 0.28 s | 8–9 ticks |
  | 12 cm | 0.70 s | 1.44 Hz | 0.35 s | 10–11 ticks |

  - The frequency spread from 5 to 12 cm is 2.23/1.44 = 1.55×, or about ±22 % around the middle. That exceeds both the ZV band (±3 %) and the ZVD band (±14 %).
  - So one fixed shaper for all trusses will not work. Either tune per scene from the truss model's L_eq, where ZVD needs L_eq right to within about ±28 % because f ∝ L^−½, or rely on acceleration caps and long ramps.
  - 33 ms ticks quantise a 0.22–0.35 s shaper delay to within about ±5–7 %. That is inside the ZVD band (±14 %) but outside the ZV band (±3 %), so ZV at 30 Hz is not viable even with a perfect length estimate.
- **Steady-state lean** (basic physics, consistent with the steady-state offset in Shukla et al.): tan θ = a/g.
  - 5° lean ↔ a = 0.86 m/s²; 10° ↔ 1.73 m/s².
  - An abrupt acceleration step from rest swings the pendulum to about twice that angle, so peak ≈ 2θ_ss.
  - Residual swing after a constant-acceleration phase of duration T_a, undamped, is about 2·θ_ss·|sin(π·T_a/T)|. It is zero only when T_a is an integer multiple of that truss's period T. A 0.7 s phase on a 0.45 s truss leaves close to the worst case, since sin(π·1.56) ≈ −0.98. A fixed "at least one period" phase duration therefore does not help across the 5–12 cm set.
  - What does help is smoothing the acceleration onset and offset: a jerk ramp, or a moving-average window, of length ≥ T. It cuts residual swing to roughly 20 % or less of the abrupt case (own estimate from the sinc-shaped spectrum of a rectangular window; not sourced). The acceleration cap remains the hard guarantee.
- **Maps to speed and acceleration limits** (scripted 30 Hz joint-space mover).
  - Cap the pinch-point Cartesian acceleration at about 0.8 m/s² (≈5° lean, ≤10° peak even with abrupt steps).
  - Smooth the joint-command stream with a moving average one pendulum period long, applied identically to every joint so tilt φ stays constant (section 1). For example, use 21 ticks for L_eq = 12 cm, or per scene from the truss's L_eq. A moving average of length T has spectral zeros at 1/T, 2/T…, so it cancels the swing at exactly that period and attenuates shorter periods. Cascading two windows gives broader robustness, similar in spirit to ZVD (own inference). Do not rely on choosing a constant-acceleration phase duration.
  - A 25 cm carry at peak 0.3 m/s with 0.7 s ramps takes about 0.83 s + 0.7 s ≈ 1.5 s. That is cheap next to typical harvest cycle times (section 4).
  - Check the pinch-point acceleration from the joint trajectory, not the joint accelerations. Pan contributes centripetal acceleration ω²r: at r = 0.25 m, ω = 1 rad/s gives 0.25 m/s² (1.5°), and ω = 2 rad/s gives 1.0 m/s² (5.8°).
- **Maps to via-poses.**
  - Add a dwell of at least one period (~0.7 s, 21 ticks) at two points: right after the cut, before retreat, so the cut-release swing decays; and at the above-crate pose before lowering.
  - Lower vertically, at constant (φ, roll), at low speed (≤0.1 m/s, own choice) so the vertical move adds no swing. Vertical acceleration does not excite a pendulum.
- **Path-generating flags.** The Shukla et al. optimiser and MPC approaches generate timing and paths. Only the "bound the per-segment swing" check carries over: compute each segment's predicted peak swing from a and duration with the pendulum formula, and reject or slow the segment if it exceeds the swing cone used in section 2.

### Gaps
- No paper on swing suppression for 5–12 cm pendulums on small hobby-servo arms. Servo tracking lag and backlash (not researched here) may filter or distort shaped commands.
- No source gives fruit-truss damping ratios. A plant cluster is probably more damped than a rigid pendulum, because the pedicels flex and fruits collide, but I found no measurement.

---

## 4. What harvesting robots do after the cut or pick

### Takeaway
Real harvesters mostly avoid a long, free-hanging carry. They use one of four patterns:
- catch the produce in a pocket, catching device or tube right at the end effector (sweet pepper, tomato truss, kiwifruit);
- store several items in the gripper (strawberry);
- carry a container on the robot itself (Ridder truss robot, Four Growers crate, strawberry punnets);
- hold the cluster by its stalk with a clip or gripper that also cuts.

For a bench arm with a crate beside the base, the transferable ideas are: put the crate where the carry is mostly pan at constant tilt, raise the crate rim toward the cut height, or (a hardware change) add a catch pocket under the truss.

### Cited Findings
- **Tomato truss, catch pocket (AHPPEBot).**
  - Design: the end effector has spike-shaped guide grooves and a saw. It rises along the peduncle with a "bottom-up wrapping" motion until the fruit are enclosed, rotates so the peduncle drops into the slot, and cuts. After the cut the truss collects in a mesh pocket.
  - Continuous autonomous test, 15 trusses: overall 86.67 % (13/15). Manual-pose test, 48 trusses: overall 89.58 % (43/48). Time per truss 21–34 s, arm start to cut done.
  - Failure: one failure occurred because "the end effector rotated, with its netted bottom touching the base of the cluster, displacing it".

  Li, Ma, Han, Yang & Zheng, "AHPPEBot: Autonomous Robot for Tomato Harvesting based on Phenotyping and Pose Estimation", arXiv 2405.06959 (2024) — [arXiv](https://arxiv.org/html/2405.06959). [FETCHED][HW] The paper reports an average of 32.46 s per success. Its own table sums to 422 s over 15 trials including the 2 failures, so the per-success figure may be mislabelled.
- **Tomato cluster, whole-cluster end effector.** Kondo et al. built an end effector for whole clusters with a 50 % success rate in high-density trials. A separate design whose scissors grasp and cut the peduncle at once reached 76.9 % (Ohmori et al., JSAM). [EAEF 3(1):20, 2010](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_article); [JSAM 77(2):113](https://www.jstage.jst.go.jp/article/jsamfe/77/2/77_113/_article/-char/en). [SNIPPET][HW] The two numbers come from different setups.
- **Tomato truss, flexible-transmission cut-and-grab.** One flexible-transmission end effector "could cut and grab at once", with 88.6 % success and 37.2 s per truss. [ResearchGate](https://www.researchgate.net/publication/272116773_Research_on_Key_Technology_of_Truss_Tomato_Harvesting_Robot_in_Greenhouse). [SNIPPET] Not verified in full text.
- **Ridder harvesting robot (commercial, trusses).** "Our robot transfers picked trusses into standard-sized, reusable packaging containers… Full containers are automatically stacked and packed on the towed harvesting trolley." [ridder.com](https://ridder.com/harvesting-robot). [VENDOR] No performance data.
- **Four Growers GR-200 (commercial).** A dual-cart setup where the robot manages its own crate changes. Peak 2.79 s per tomato, average depending on crop density. Beckhoff case study, July 2026 — [beckhoff.com](https://www.beckhoff.com/pt-br/company/applications-and-references/farm-to-fleet-engineering-the-future-of-robotic-greenhouse-harvesting.html). [VENDOR]
- **Harvested trusses stored piled in crates.** Trusses are "stacked in a crate with considerable clutter", which the authors call the common way they are stored and transported after harvest. Robots pinch-grasp the peduncle from such a pile with 100 % clearance and 93 % first-try grasps in lab tests. Van den Bent, Coleman & Babuška, "Robotic Grasping of Harvested Tomato Trusses Using Vision and Online Learning", ICRA 2024, pp. 13947–13953, [doi:10.1109/ICRA57147.2024.10610089](https://doi.org/10.1109/ICRA57147.2024.10610089) — [arXiv 2309.17170](https://arxiv.org/abs/2309.17170). [FETCHED][HW]
- **Cherry tomato, single fruit (two-jaw Fin Ray gripper).**
  - Grip-twist-pull strategy: 86.0 % picking success, 2.3 % fruit damage, 8.4 s per fruit.
  - Real hardware: 50 trials in a commercial greenhouse on a UR5 plus Husky base, plus Isaac Sim.

  Singh, Kaur, Adamu & Hussain, arXiv 2610.18160 (Oct 2026) — [arXiv](https://arxiv.org/abs/2610.18160). [FETCHED][HW] A search summary also reported a transport phase of ~2.2 s and a release of ~0.23 s inside the 8.4 s cycle. [SNIPPET] This phase breakdown was not in the abstract I read.
- **Tomato, stage timing incl. transport and release (2026).** A system timing approach, separation, cutting, grasping, transport and release stages reported a 24.3 s average cycle and ~80 % success in 10 lab trials. [Scientific Reports 2026](https://www.nature.com/articles/s41598-026-42227-2). [SNIPPET]
- **Sweet pepper, catching device (SWEEPER).**
  - Catch: "Almost all peppers that are cut will drop into the finger-type catching device". The fingers sometimes pushed the stem out of the knife's reach.
  - Damage and handoff: no fruit damage was seen from the drop. Dropping into the blue container was temporary, and later versions used a conveyor.
  - Performance: 4-week commercial trial, 262 fruit, 24 s average cycle, 61 % success in best-fit crop conditions and 18 % in current conditions.

  [WUR SWEEPER project/FAQ](https://wur.nl/en/project/SWEEPER-the-sweet-pepper-harvesting-robot.htm); Arad et al., JFR 2020 — [WUR record](https://research.wur.nl/en/publications/development-of-a-sweet-pepper-harvesting-robot/). [SNIPPET][HW]
- **Kiwifruit, gripper-through tube.**
  - Handoff: fruit drop through the gripper bottom into a flexible tube to a conveyor.
  - Losses: one test had 9 % of fruit dropped outside the chute and 23.4 % overall fruit loss in the original harvester. Causes included the hand striking target or neighbouring fruit, and fruit not fully enclosed before release.
  - Performance: the four-arm version harvested 51.0 % of fruit at 5.5 s per fruit.
  - Economics: break-even requires ≥95 % (green) or 97 % (gold) success.

  Williams et al. (Univ. of Auckland/Waikato) — [Waikato thesis/papers](https://researchcommons.waikato.ac.nz/entities/publication/d5f2cfe0-097d-422c-8ecf-ce4af49b6d43), [JFR, issue dated 16 July 2019, doi:10.1002/rob.21890](https://zendy.io/title/10.1002/rob.21890), [2017 poster](https://zenodo.org/records/1002294/files/ACPA%20Poster%20301.pdf). [SNIPPET][HW]
- **Cucumber (conflicting figures).** One source reports 80 % success and 45 s per cucumber (Autonomous Robots 2002, [doi:10.1023/A:1020568125418](https://www.doi.org/10.1023/A:1020568125418)). Another reports 74.4 % success, 65.2 s per successful cycle, and about 124 s per harvested cucumber once failures are counted (Biosystems Engineering field test, [WUR](https://research.wur.nl/en/publications/field-test-of-an-autonomous-cucumber-picking-robot)). The post-cut transport method was not found. [SNIPPET][HW]
- **Strawberry, multi-berry gripper storage.** A cable-driven gripper stores several berries, so the arm makes fewer trips to the container. [arXiv 1804.09771](https://arxiv.org/pdf/1804.09771.pdf). [SNIPPET][HW]
- **Strawberry, on-board punnets.** Octinion's Rubion is reported to place berries straight into a punnet at about one berry per 3 s. Dogtooth's robots carry each berry a few cm to an on-board inspection station, then into a punnet or waste chute; sources disagree on whether placement is direct. [profi.co.uk](https://www.profi.co.uk/news/first-autonomous-strawberry-picking-robot/); [Ingenia](https://ingenia.org.uk/articles/fruit-picking-robots/); [control.com](https://control.com/news/strawberry-season-is-here-denso-certhon-and-dogtooth-technologies-harvesting-robots-are-at-the-ready/). [VENDOR/press]
- **Grape, stalk-hanger clip.**
  - Patent idea: a clip holds the cluster by its stem through cutting, conveying and packing, so "the picker and the packer may hold the cluster by the hanger clip, and need not touch the grape berries" ([patent 7080498](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7080498); [patent 7669399](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7669399)). [SNIPPET]
  - Separately, a dual-arm grape robot reached 9 s per bunch and 83 % success ([Frontiers in Plant Science 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.881904/pdf)). [SNIPPET][HW]
- **Apple, bin-filling.** An orchard bin-filling device averaged 5.67 % damage. [SciELO](https://www.scielo.br/j/eagri/a/9kwTCDfvkz3WPfX6VxGTFky/). [SNIPPET][HW] Vacuum-tube apple robots claim reduced bruising qualitatively, but I found no tube-transport bruise percentage ([arXiv 2010.11296](https://arxiv.org/pdf/2010.11296)).

### Inferences
- **Patterns that shorten or remove the bench carry:**
  - **Container on the arm's "lap" (crate placement).** Ridder, Four Growers and the strawberry robots all keep the container close to the arm. On a bench, place the crate so its drop point sits at about the same radius from the pan axis as the cut points, inside the fixed-tilt workspace from section 1. Raise it on a stand so the rim top is roughly (cut height − truss length − margin). The carry then becomes: retreat radially about 5 cm, pan, short vertical lower. Swing comes only from pan (bounded in section 3), and the descent is short.
  - **Catch pocket or tray under the truss (hardware).** AHPPEBot's mesh pocket and SWEEPER's catching fingers show that a catcher under the cluster removes the free pendulum during transport. AHPPEBot also shows the risk: the catcher can push the cluster before the cut. This needs a hardware change to the jaw. I did not count it as a motion-only fix.
  - **Chute or tube (layout).** A short inclined chute beside the plant row turns the carry into "move a few cm, open". Kiwifruit data show 9 % misses when release is imprecise. For trusses (10–17 cm, irregular) a chute would need to be wide, and dropping a truss onto a slope risks rolling fruit. Weak fit (own judgement).
- **Maps to crate placement.**
  - Crate centre at the pan radius of the median cut point.
  - Crate long side tangential to the pan arc, so pan swing (tangential to the arc) has the 30 cm of room rather than 28 cm.
  - Crate on a stand so that "above-crate" is the same height as "after-retreat". The carry via-poses then differ only in pan, and the planner's collision check covers a single sweep.

### Gaps
- No source describes how any commercial truss robot (Ridder, others) moves the truss from the cut to its on-board container: catch, carry or tilt. Nor do they give drop or damage rates.
- No published drop or damage rate for any truss harvester's transport phase specifically.
- Root AI, Panasonic and other tomato robots were not reached in this pass.
- The 2.2 s transport and 0.23 s release split for the cherry-tomato system is snippet-only.

---

## 5. Placing a hanging cluster into a shallow container without knocking fruit; tomato drop and bruise thresholds

### Takeaway
I found no robotics paper on lowering a hanging cluster into a container, laying it down, or dropping it. Postharvest data on whole standard-size tomatoes dropped onto a surface show internal bruising already from 10 cm drops at breaker stage and 20 cm at mature-green, with cushioning reducing it greatly (the impact surface in the 10/20 cm study was not confirmed). Release should therefore be "lower until the lowest fruit touches, then release from ≈0 cm" rather than dropping, at least onto a hard crate floor. Trusses piled loosely in crates are the industry's normal storage, so truss-on-truss contact at low speed is acceptable.

### Cited Findings
- **Whole-fruit drops.**
  - Mature-green fruit: single drops from 20 cm onto opposite sides gave 5–45 % internal bruising, by cultivar.
  - Breaker-stage fruit: single drops from 10 cm gave 15–73 %.
  - A single 30 cm drop was enough to cause internal bruising.
  - Bruising is cumulative over handling steps.

  "Sensitivity of tomatoes at mature-green and breaker ripeness stages to internal bruising", J. Amer. Soc. Hort. Sci. 117(1):119 (1992) — [ASHS PDF](https://journals.ashs.org/downloadpdf/view/journals/jashs/117/1/article-p119.pdf) (403 on fetch), [AGRIS record](https://agris.fao.org/search/en/records/65df8e4db766d82b180260d0). [SNIPPET][HW] Fruit size is standard tomato, not cherry. The impact surface was not confirmed.
- **Cultivar threshold and repeated impacts.** Bruising first appeared at drops of about 10–15 cm, depending on cultivar. Repeated drops on the same spot caused much more internal bruising than a single drop. "Response of tomato fruit to consecutive impact loading" — [ResearchGate](https://www.researchgate.net/publication/382685202_Response_of_tomato_fruit_to_consecutive_impact_loading). [SNIPPET]
- **Pendulum-impactor energy threshold.** Almost all bruises occurred above 70 mJ, with a threshold between 60 and 80 mJ. "Factors that affect tomato bruise development as a result of mechanical impact" — [ResearchGate](https://www.researchgate.net/publication/229377330_Factors_that_affect_tomato_bruise_development_as_a_result_of_mechanical_impact). [SNIPPET] This uses a pendulum impactor on round tomatoes and should not be converted into a cherry-tomato drop height.
- **Steel-ball drops onto fruit.** Studies dropping a steel ball onto the fruit from 20, 40 and 60 cm found bruise area rising with height and with storage at 22 °C. [Horticulturae 7(5):113, 2021](https://www.mdpi.com/2311-7524/7/5/113); [Emerald AGJSR](https://www.emerald.com/agjsr/article/42/3/646/1218697/Influence-of-drop-test-impact-on-the). [SNIPPET] These are ball heights, not fruit drop heights.
- **Impact surface.** Metal surfaces caused the most damage and foam the least ([AU J.T. 10(4):271–275, 2007](https://funaab.edu.ng/wp-content/uploads/2010/11/An%20Assessment%20of%20Impact%20Damage%20to%20Fresh%20Tomato%20Fruits.pdf)). An older USDA report found that 1-inch foam rubber over the hard drop surface "greatly reduced bruising" ([USDA MRR 513](https://ageconsearch.umn.edu/record/313075/files/mrr513.pdf)). [SNIPPET]
- **Industry practice.** Harvested trusses are normally "stacked in a crate with considerable clutter". Van den Bent et al., ICRA 2024 — [arXiv 2309.17170](https://arxiv.org/abs/2309.17170). [FETCHED]
- **Short drop into a container.** SWEEPER saw no fruit damage from peppers dropping into a container, a short drop, which it treated as a temporary solution. [WUR SWEEPER FAQ](https://wur.nl/en/project/SWEEPER-the-sweet-pepper-harvesting-robot.htm). [SNIPPET][HW]
- **Release mistakes cause losses.** Kiwifruit drops were partly caused by fruit "not fully enclosed during closing, so it could be flung or miss the chute on release". [Waikato](https://researchcommons.waikato.ac.nz/entities/publication/d5f2cfe0-097d-422c-8ecf-ce4af49b6d43). [SNIPPET][HW]

### Inferences
- **Maps to via-poses for release** (own design, unvalidated):
  1. **Above-crate pose.** Same (φ, roll) as the carry. The lowest fruit is at least (wall height 5 cm + swing margin ℓ·sin θ + 2 cm) above the rim, and the pinch is over the crate centre or the chosen free spot. Dwell one pendulum period (~0.7 s).
  2. **Vertical lower** at ≤0.1 m/s until the lowest fruit is about 1 cm above the floor or above existing trusses, using the predicted hanging geometry from section 2.
  3. **Optional lay-down.** Continue down 2–4 cm while translating 2–4 cm horizontally toward the crate's empty side, so the truss folds onto its side, lowest fruit first, instead of the stalk standing upright on the fruit. Releasing a truss that stands on its lowest fruit lets it topple; the topple height is roughly the truss length, a drop of up to 10–17 cm for the top fruit.
  4. **Open the jaw** at near-zero speed, then retreat straight up at constant (φ, roll).
- **Maps to pose selection.** Choose the release candidate as the crate spot nearest the empty side, so the truss lies down toward free floor. Reject release candidates whose swing cone touches a wall or a previous truss.
- **Drop allowance.** Given 10 cm whole-fruit drops already bruising breaker-stage fruit (surface unconfirmed), I would keep any unsupported fall under about 2–3 cm. This is my own margin, not a sourced threshold. A foam or soft liner on the crate floor is cheap insurance on real hardware.

### Gaps
- No paper or patent on lowering, laying down or dropping a hanging fruit cluster into a container, and no comparison of "tilt and drag" against "drop from small height".
- No cherry-tomato-specific drop-height bruise threshold.
- No data on bruising from fruit-on-fruit contact inside a crate, or on pedicel detachment when a truss lands. The 0.6–2.5 N detachment force is from the task brief, not a source.

---

## 6. Scripted or planner-checked expert demonstrators for imitation learning: success rates, failure handling, filtering bias

### Takeaway
The standard practice is to generate attempts, check task success automatically, discard failures and retry, with optional resampling of the scene.
- MimicGen reports 82 % and 52 % generation success on real tasks, and as low as 7–8 % on hard simulated tasks.
- Later planner-assisted systems report 75–92 %.
- Policies trained on filtered data often beat the generator's own success rate.
- Filtering by success biases the dataset toward easy initial conditions. One measured case showed at most about 15 % per-region deviation. That is small but not zero, so it should be measured, and scene families that always fail should be reported rather than silently dropped.

### Cited Findings
- **MimicGen.**
  - "MimicGen checks for task success after executing all segments, and only keeps successful demonstrations… Attempts that did not achieve task success were discarded."
  - Limitation: "MimicGen only filters data generation attempts based on task success, so generated datasets can be biased."
  - Transit motions use "linear interpolation between human segments", which "does not guarantee collision-free motion". The authors chose this "to avoid the complexity of integrating a planner".
  - Real-robot generation rates: 82.3 % (Stack, 243 attempts) and 52.1 % (Coffee, 192 attempts).
  - Simulation example: Gear Assembly D1 generated at 8.2 %, but the trained agent reached 76.0 %.

  Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023, [doi:10.48550/arXiv.2310.17596](https://doi.org/10.48550/arXiv.2310.17596) — [arXiv HTML](https://arxiv.org/html/2310.17596). [FETCHED][HW+SIM]
- **Measured filtering bias.** "MimicGen works by initializing the environment, attempting to solve the task… and then reinitializing it once it exceeds some retry budget. This process can bias the dataset towards initial configurations where MimicGen is likely to succeed." Measured result: "minor bias… more data in the lower-right quadrant… within each tile, the deviation from an unbiased distribution is at most about 15%." Wilcox et al., "MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics", CoRL 2026, [arXiv 2611.05567](https://arxiv.org/abs/2611.05567), Appendix B.4 — [draft PDF](https://www.pair.toronto.edu/MESA/static/pdfs/draft.pdf). [FETCHED][SIM]
- **SkillMimicGen.** Reported generation success averaged 75.4 % against 40.7 % for MimicGen. The mechanism was not verified in full text. [arXiv 2410.18907](https://arxiv.org/pdf/2410.18907). [SNIPPET]
- **DemoStream.** Batched task-and-motion-plan optimisation, with 92 % success against 35 % for MimicGen in its comparison. [arXiv 2610.22020](https://arxiv.org/pdf/2610.22020). [SNIPPET]
- **Scaling Up and Distilling Down.**
  - An LLM writes a success-condition check for each task, so collection detects failures, retries and labels trajectories.
  - The dataset is "filtered for success" before distillation into a diffusion policy.
  - The distilled policy learned retrying behaviour, improved absolute success by 33.2 % on average over five domains, and transferred to a real robot with domain randomisation.

  Ha, Florence & Song, CoRL 2023 — [arXiv 2307.14535](https://arxiv.org/abs/2307.14535), [PMLR](https://proceedings.mlr.press/v229/ha23a.html). [SNIPPET]
- **Planner entropy.** Sampling-based planners (RRT) produce demonstrations with "unfavorably high entropy", which degrades contact-rich policy learning. Making the planner favour consistent demonstrations helps. [arXiv 2412.09743](https://arxiv.org/html/2412.09743v3). [SNIPPET]
- **Neural MP.** When the expert finds no exact solution, it relabels the reached final state as the goal instead of discarding the attempt. [arXiv 2409.05864](https://arxiv.org/pdf/2409.05864). [SNIPPET]
- **Demo-SCORE.** Training a classifier on policy rollouts to filter demonstrations raised absolute success by 15–35 % over training on all demonstrations. [arXiv 2503.03707](https://arxiv.org/html/2503.03707v1). [SNIPPET]
- **Discarding data is common.** "A significant fraction of collected data being discarded through the process of data curation is common" in large-scale collection. [arXiv 2510.19495](https://arxiv.org/pdf/2510.19495v1). [SNIPPET]
- **Real harvest economics.** Field harvesters need very high success: kiwifruit break-even was ≥95–97 %. Time on failed attempts must be counted in cycle time ([Waikato](https://researchcommons.waikato.ac.nz/entities/publication/d5f2cfe0-097d-422c-8ecf-ce4af49b6d43); [review, MDPI Agronomy 2025](https://www.mdpi.com/2073-4395/15/11/2650)). [SNIPPET]

### Inferences
- **The 13 % of scenes with no collision-free carry** are where filtering bias would concentrate: likely crowded trusses, trusses close to neighbours, or awkward stalk angles. Before filtering:
  1. Re-check those scenes with the hanging-truss model and swing cone (section 2) and with the fixed-tilt candidate sets (section 1), since some may become feasible.
  2. For those that stay infeasible, log the scene parameters and compare the retained-scene distribution with the sampled distribution per parameter bin, as in MESA's per-tile frequency check. Flag bins with more than about 15 % deviation.
  3. Do not resample the same scene with different random seeds until it succeeds without recording it; that is the MimicGen retry-budget mechanism.
- **Failures during execution** (pivot, swing hit, drop) are best discarded for training but counted. The demonstrator's 67–70 % is the analogue of MimicGen's generation rate. Policies can exceed it (MimicGen Gear Assembly: 8.2 % generation → 76.0 % agent), but only if the retained data covers the hard scenes.
- **Consistency** (from the entropy finding): the fixed via-pose plus candidate-choice structure already yields low-entropy demonstrations. Keep tie-breaking rules deterministic, for example always choose the lowest-swing-risk release candidate rather than a random feasible one.

### Gaps
- No paper reports generation success for a scripted demonstrator with a pendulum-like grasped payload, or for any agricultural harvesting demonstrator.
- No quantitative study links the amount of success-filtering bias to downstream policy failure on the under-represented initial conditions, beyond MESA's distribution measurement.

---

### Gaps across all questions (searched and not found)
- **Tomato-specific transport after cut.** No published method or data on how truss robots move a truss from the cut to a crate (swing, pivot, damage). Commercial systems (Ridder, Four Growers) give no technical detail.
- **Fewer than 6 DOF.** No literature on orientation-constrained carrying with arms under 6 DOF. Section 1's pan-invariance argument is my own.
- **Stalk torsional friction.** No measurements of plant-stalk torsional friction in a pinch grasp, and no truss damping or pendulum parameters.
- **Placement method.** No studies comparing lowering, laying down or dropping clusters into containers.
- **Cherry-tomato bruising.** No cherry-tomato drop-bruise thresholds.
- **Unopened full texts.** Several items are snippet-only, marked [SNIPPET]: the hydraulic-manipulator compound-pendulum paper, Biagiotti FIR, SkillMimicGen, DemoStream, Kondo/Ohmori, SWEEPER JFR, and the kiwifruit JFR paper. The KTH ICRA 2016 PDF returned 404 and the ASHS 1992 PDF returned 403.
