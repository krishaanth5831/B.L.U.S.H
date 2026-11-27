# Rod pinch-grip mechanics: when a pinched stalk turns, and which pad geometry and material stop it at 4-10 N

Written 2026-11-25. Evidence tags used throughout:
- **[MEASURED]**: numbers measured on real hardware in the cited source.
- **[THEORY-CITED]**: a formula stated in the cited source.
- **[SNIPPET]**: taken from a search-result summary; the full text was not read in this session. Verify before relying on it.
- Inferences are my own derivations from Coulomb friction and statics, with assumptions stated. They are not cited.

Conventions for every number below:
- F is the clamp force at each pad (4 N or 10 N). With two pads, the rod has two contacts and carries 2F of total normal force.
- Rod radius r = 2 mm (4 mm stalk).
- L is the contact length along the rod: 13 mm (current pad width) or 36 mm (pad turned 90° so the stalk lies along its length).
- The load to resist is 0.04-0.08 N·m (given). A check: 0.1 kg × 9.81 × 0.08 m = 0.078 N·m.
- μ values used: 0.3 (hard plastic, from reed-on-hard-material data 0.12-0.37), 0.5 (TPU/PLA-like, from measured 0.36-0.54), and 1.0 (rubber on reed, measured 0.81-1.02). Sources are in the friction section.

---

## 1. Contact mechanics: holding torque of a pinch grip about the three axes

### Takeaway
Flat 13 mm pads hold about μ·F·L/2 against pivoting: 0.013 N·m at 4 N and 0.033 N·m at 10 N with μ = 0.5. Neither reaches the 0.04-0.08 N·m load unless μ is about 1 at 10 N. That is not inconsistent with the simulation, where 9.5 N reduced the turn but did not stop it, but the simulator's friction and condim settings are unknown here (item F). The two levers that change this by large factors are:
- a longer stance along the rod (the torque is linear in L, and pushing the pressure to the ends doubles it);
- geometry that turns pivoting into "climbing out of a groove", so the clamp springs resist it instead of friction.

Pad softness changes the torque mainly by making the contact real over the full length and by letting the stalk sink in. It does not change the basic μ·F·L scaling.

### Cited Findings
- **[THEORY-CITED]** Hertzian circular soft-finger contact:
  - pressure p(ρ) = p0·(1 − ρ²/a²)^½, with p0 = 3F/(2πa²);
  - maximum friction force f_max = μF;
  - maximum torsional friction moment m_max = (3π/16)·μ·a·F ≈ 0.59·μ·a·F;
  - contact radius a = (3RF/(4E*))^⅓, so the torsional moment grows as F^(4/3) while the tangential friction grows as F.

  Source: [Ghazaei Ardakani, Bimbo, Prattichizzo, "Quasi-static Analysis of Planar Sliding Using Friction Patches", arXiv:1904.06677 (Eq. 53-57)](https://ar5iv.labs.arxiv.org/html/1904.06677)
- **[THEORY-CITED]** The coupled force-torque friction limit surface of a patch contact is approximated well by an ellipsoid. This was confirmed for soft-finger models by Xydas & Kao and by Fakhari et al. Source: [arXiv:1904.06677](https://ar5iv.labs.arxiv.org/html/1904.06677)
- **[SNIPPET]** Xydas & Kao derived a relation between normal force and contact radius for soft fingers (linear and nonlinear elastic materials). Combined with the pressure distribution, it gives a normalised friction limit surface. Source: [Xydas & Kao, "Modeling of contact mechanics and friction limit surfaces for soft fingers in robotics, with experimental results", IJRR 18(8), 1999, DOI 10.1177/02783649922066673](https://www.osti.gov/biblio/20001201). I did not read the equations themselves (see Gaps).
- **[MEASURED]** A sensorised parallel gripper with flat, rigid, 15 mm-radius pads (3D-printed PLA or TPU95) was used to estimate the torsional friction as an equivalent "rim radius" r_eq, defined by τ_max ≈ μ_c·r_eq·F_n.
  - Estimates ranged from 3.9 to 7.9 mm across cardboard, a phone case, plastic and wood. Typical values were 5-7.6 mm.
  - The uniform-pressure theory value is 10 mm (that is, 2/3 × 15 mm).
  - Measured Coulomb μ_c was 0.36-0.54, and static μ_s was 0.40-0.57.
  - TPU95 gave more friction than PLA.
  - The grip force during exploration was 5 N.

  So real flat pads gave about 40-80 % of the uniform-pressure torsional capacity. Source: [Waltersson & Karayiannidis, arXiv:2410.19660 (Tables 8-10)](https://arxiv.org/html/2410.19660)
- **[THEORY-CITED]** That gripper's controller uses ellipsoidal coupling between force and torque:
  - minimum normal force for a torque load: f_n,min = |τ|/(μ_c·r_eq·γ_τ);
  - minimum normal force for a tangential load: f_n,min = f_t/(μ_c·γ_t);
  - the γ factors share the capacity between force and torque.

  Source: [arXiv:2410.19660 (Eq. 16-21)](https://arxiv.org/html/2410.19660)
- **[THEORY-CITED, real prototype]** The two-phase gripper patent models the frictional torque of a small contact patch as τ = μ·r·F_grip. Its bounds are:
  - no linear slip: F ≥ Mg/μ;
  - the part pivots under gravity: F ≤ M·g·L·cos φ/(μ·r).

  The prototype used μ ≈ 0.6 (estimated for its finger-object pair), a contact patch about 3 mm in diameter, and grip forces limited to 5-30 N. Source: [US9808936B2, "Two-phase gripper to reorient and grasp"](https://patents.google.com/patent/US9808936); paper: [Chavan-Dafle, Mason, Staab, Rossano, Rodriguez, "A two-phase gripper to reorient and grasp", IEEE CASE 2015](https://dspace.mit.edu/handle/1721.1/107782)

### Inferences (my derivations; Coulomb friction, rigid rod unless stated)

**A. Pivot about the pad normal (rod turns in the pad plane), flat pads.**

Model a contact line of length L with pressure q(x), rotating about its centre. The friction moment per contact is τ = μ∫q(x)|x|dx. With two contacts:

| Pressure along the line | Per contact | Total (2 contacts) |
|---|---|---|
| Uniform, q = F/L | μ·F·L/4 | **μ·F·L/2** |
| Centre-peaked parabolic (soft pad, slightly curved stalk) | μ·F·3L/16 | 0.375·μ·F·L, i.e. 25 % less |
| Two ridges at spacing D (pressure forced to the ends) | μ·F·D/2 | **μ·F·D**, about 2× uniform for D ≈ L |

Worked numbers, in N·m. For ridges, D = L − 2 mm (ridge centres 1 mm in from the edges).

| L | μ | Uniform at 4 N | Uniform at 10 N | Ridges at 4 N | Ridges at 10 N |
|---|---|---|---|---|---|
| 13 mm | 0.3 | 0.008 | 0.020 | 0.013 | 0.033 |
| 13 mm | 0.5 | 0.013 | 0.033 | 0.022 | 0.055 |
| 13 mm | 1.0 | 0.026 | 0.065 | 0.044 | 0.110 |
| 36 mm | 0.3 | 0.022 | 0.054 | 0.041 | 0.102 |
| 36 mm | 0.5 | 0.036 | 0.090 | 0.068 | 0.170 |
| 36 mm | 1.0 | 0.072 | 0.180 | 0.136 | 0.340 |

Required friction to hold a torque τ:
- uniform pressure: μ_req = 2τ/(F·L);
- ridges: μ_req = τ/(F·D).

For the 0.04-0.08 N·m load:
- **13 mm, uniform:** at 4 N, μ_req = 1.5-3.1 (not achievable). At 10 N, μ_req = 0.62-1.23 (only rubber-like faces, and only for the lighter or shorter trusses).
- **13 mm, ridges (D = 11 mm):** at 4 N, μ_req = 0.9-1.8. At 10 N, μ_req = 0.36-0.73 (achievable with rubber).
- **36 mm, uniform:** at 4 N, μ_req = 0.56-1.1. At 10 N, μ_req = 0.22-0.44.
- **36 mm, ridges (D = 34 mm):** at 4 N, μ_req = 0.29-0.59. This is achievable at the current 4 N with rubber (0.8-1.0), with a safety factor of about 1.4-3.

Measured flat pads reach only about 40-80 % of the uniform theory (rim-radius data above), so derate every uniform number by about 0.6.

**B. Coupling with the weight.**

The pads also carry the truss weight by friction: 0.7-1.0 N against a capacity of 2μF (4 N at μ = 0.5 and F = 4 N). With the ellipsoidal limit surface, (f/f_max)² + (τ/τ_max)² ≤ 1, so the torque capacity drops to about √(1 − 0.25²) ≈ 0.97 at μ = 0.5, and to about 0.91 at μ = 0.3. This is a minor effect.

**C. Rocking (rod tilts out of the pad plane, about the in-plane axis perpendicular to the rod).**

There is no friction term here. With rigid pads guided parallel, the rod can only rock by pushing the pads apart, so the centre of pressure on the two pads moves to opposite ends. The maximum couple is about F·L:
- L = 13 mm: 0.052 N·m at 4 N and 0.13 N·m at 10 N;
- L = 36 mm: 0.144 N·m at 4 N and 0.36 N·m at 10 N.

**Caveat:** if each pad sits on a single central spring and can tilt, this resistance largely disappears. Pads must be guided parallel, for example with two springs at the pad ends or a linear guide.

**D. The patent pivot bound with our numbers.**

Using F_pivot ≤ M·g·L/(μ·r) with μ = 0.6 and r = 1.5 mm (a 3 mm patch):
- M = 0.085 kg, L = 5 cm: about 46 N;
- range M = 0.07-0.1 kg, L = 3-8 cm: about 23-87 N.

So a small-patch grip lets the truss pivot at any force the servo can produce. The 13 mm line contact is about 2-4× better than a 3 mm patch, and it is still marginal. The factor is 4× if the patent's μ·r·F is the two-contact total and 2× if it is per contact; the patent text as read does not say which.

**E. Consistency with the simulation (assumes μ = 1).**

At 9.5 N, uniform pressure, μ = 1 and L = 13 mm, the capacity is 0.062 N·m. That holds the light or short trusses but not 0.08 N·m, which is not inconsistent with "reduced but did not stop". At 4 N the capacity is 0.026 N·m or less, so the grip always turns. The simulator's real friction model may differ (item F), so this is not a validation.

**F. Unverified note on the simulator (from my prior knowledge, not checked this session).** In MuJoCo, torsional resistance at a single contact exists only with condim ≥ 4. Its coefficient (the second friction value, default about 0.005, a length) multiplies the normal force. So the pivot resistance in simulation may come mostly from how many contact points the pad-stalk collision generates and how far apart they are, not from the formula above. Check the pad and stalk geom friction and condim against the [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) before trusting simulated turn angles.

**G. Pad compliance (a rough Winkler-layer model; my assumptions).**

For a rod of radius R pressed into an elastomer layer of thickness t and modulus E by a line load q = F/L, the contact half-width is b ≈ (1.5·q·R·t/E)^⅓ and the sink depth is δ ≈ b²/(2R). With t = 3 mm, L = 13 mm and R = 2 mm:
- **E ≈ 1 MPa (roughly Shore 30A):** b = 1.4 mm and δ = 0.5 mm at 4 N; b = 1.9 mm and δ = 0.9 mm at 10 N. The stalk sinks 25-45 % of its radius, so the pad forms its own shallow groove.
- **E ≈ 30 MPa (roughly TPU 95A):** b = 0.45 mm and δ = 0.05 mm at 4 N. This is effectively a flat line contact.

Shore-to-modulus conversions are rough and from memory. The stalk's own compliance is ignored.

What this means:
- A soft face mostly helps by producing full-length contact on an irregular stalk and by adding wrap, which resists spin and pivot through partial form closure.
- It does not increase the μ·F·L ceiling for pure pivoting, except through a higher μ.
- A centre-peaked pressure from a crowned or soft pad lowers the pivot capacity by about 25 %. Prefer stiff backing with soft edge ridges.

### Gaps
- I did not read the full Xydas & Kao (1999) or Howe & Cutkosky limit-surface papers, so I could not quote their exact soft-finger exponent or normalisation constants.
- The KTH ICRA 2016 pivoting paper (Viña et al.) could not be fetched (404).
- No source gave a measured torsional capacity for a rod-on-pad line contact (as opposed to a disc). The line-contact formulas above are derivations only.

---

## 2. Spin about the rod's own axis: flat pads vs V-groove, round groove, serrations

### Takeaway
The lever arm for spin is only the 2 mm stalk radius:
- flat pads hold 2·μ·F·r, which is 0.008-0.04 N·m across 4-10 N and μ = 0.5-1;
- V-grooves on both jaws multiply that by at most 1/sin(β), where β is the groove half-angle (×2 for a 60° V). That is the ideal case where flank friction does not oppose seating. If the stalk seats against flank friction, the factor is 1/(sin β + μ cos β) per jaw. That is 0.73 for 60° and μ = 1, or 0.71 for 90° and μ = 1, which is worse than flat pads. So a V does not reliably help spin when μ is high.

That holds only a fruit-mass offset of about 1 cm (flat, μ = 0.5, 4 N) to at most about 5-10 cm (best case, V upper bound, μ = 1, 10 N) from the stalk axis, at the cost of high line pressures on a soft stalk and possible wedge sticking on release. Spinning until the fruit hang below is the natural equilibrium. Allowing it, or gripping in that pose from the start, is cheaper than fighting it.

### Cited Findings
- **[THEORY-CITED, design]** Two-phase gripper V-groove design rules:
  - the cylinder seats in a V-groove that centres it;
  - groove depth H and V half-angle θ must satisfy H ≤ R/sin θ, so the jaws themselves do not touch while holding;
  - an elastic strip must extend at least δ = 2H(1 − sin θ)/cos θ to seat the part;
  - strip stiffness bounds: F_pivot/(2δ_pivot) ≤ K ≤ F_grasp/(2δ_grasp).

  The prototype used 3D-printed V fingers, a pre-loaded rubber band as the strip, and a cured liquid-rubber dot as the point contact. No numeric groove angle is given. Source: [US9808936B2](https://patents.google.com/patent/US9808936)
- **[SNIPPET, industrial practice]** Configurations for gripping round parts:
  - a V finger on one side and a flat finger on the other gives a 3-line contact that works across a range of diameters;
  - V fingers on both sides give a 4-line contact "to hold parts even more securely for high-speed applications" and keep the part centreline constant across diameters;
  - roughing or knurling finger faces is recommended for parts slick with coolant.

  Source: [American Machinist, "Let's give robots a hand"](https://www.americanmachinist.com/cutting-tools/article/21895163/lets-give-robots-a-hand)
- **[Industrial guidance]** Flat fingertips on a parallel gripper make 2 contact lines on a cylinder, which "can create instability". A V-shaped fingertip gives 4 contact points, which is more stable and centred. Silicone or high-density rubber is suggested to raise friction. No angle or durometer is given. Source: [Robotiq, "How to design gripper fingertips"](https://blog.robotiq.com/bid/73286/how-to-design-gripper-fingertips)

### Inferences (my derivations)

**Flat pads.** Spinning a rod between two fixed flat pads slips at both contacts, so the resisting torque is τ_spin = 2·μ·F·r. Rolling on one pad still requires slip at the other with lever 2r, giving the same bound. With r = 2 mm:

| μ | 4 N | 10 N |
|---|---|---|
| 0.5 | 0.008 N·m | 0.020 N·m |
| 1.0 | 0.016 N·m | 0.040 N·m |

**V-groove of included angle 2β (both jaws grooved, 4 line contacts).**
- Upper bound (no flank friction opposing seating): each flank carries N = F/(2 sin β), so each jaw's normal-force sum is F/sin β, and τ_spin = 2·μ·r·F/sin β. With one V and one flat jaw: μ·r·F·(1 + 1/sin β).
- Lower bound (stalk seated against flank friction): F = 2N(sin β + μ cos β), so τ_spin = 2·μ·r·F/(sin β + μ cos β). For 60° and μ = 1 at 10 N this gives about 0.029 N·m, below flat pads' 0.040.
- The table below is the upper bound. The true value depends on loading history, for example whether the clamp force rises after the stalk has settled.

| Included angle | μ | 4 N | 10 N |
|---|---|---|---|
| 90° (β = 45°, factor 1.41) | 0.5 | 0.011 N·m | 0.028 N·m |
| 90° (β = 45°, factor 1.41) | 1.0 | 0.023 N·m | 0.057 N·m |
| 60° (β = 30°, factor 2) | 0.5 | 0.016 N·m | 0.040 N·m |
| 60° (β = 30°, factor 2) | 1.0 | 0.032 N·m | 0.080 N·m |

**Spin load.** The load is m·g·e, where e is the horizontal offset of the truss centre of mass from the stalk axis. For m = 0.085 kg, the largest offset that can be held is e_max = τ/(m·g):
- flat pads, μ = 0.5, 4 N: about 1 cm;
- two 60° V-grooves, μ = 1, 10 N: about 9.6 cm at the upper bound, and about 3.5 cm at the lower bound.

The real e of a truss hanging from its stalk is unknown (see Gaps).

**Costs of narrow V-grooves:**
1. Line pressure on the stalk rises by 1/sin β per flank. A 60° V puts F on each flank line, which raises the crushing risk on a green stalk.
2. The wedge self-locks when tan β < μ. For a 60° V that means μ > 0.58, which is true for rubber, so the stalk can stay stuck in the groove when the jaws open over the crate.

A 90° V with rubber (tan 45° = 1 ≈ μ) is near that boundary. A 90-120° V is the safer compromise if release matters.

**Round groove (radius slightly larger than the stalk).** This behaves like a V with distributed wrap. It conforms only for one diameter, and a 3-5 mm range breaks that. The V's main virtue is that it self-centres any diameter.

**Serrations, knurls and teeth.** These add form closure by indenting the stalk. There is no general formula; the gain depends on tooth depth relative to stalk tissue strength. Use them only if stalk damage behind the cut is acceptable. It may be, since the stalk stub is waste, but this is not verified.

### Gaps
- No source gave a recommended V-jaw included angle from a gripper maker. The Festo gripper white paper (403) and the Robotiq ebook were not readable.
- No measured spin-torque data for rods in V-grooves.
- The centre-of-mass offset e of a hanging truss relative to its peduncle axis is not in any source I found.

---

## 3. Geometries that lock a rod against pivoting at low force

### Takeaway
At 4-10 N, friction alone on a 13 mm flat line contact cannot reliably hold 0.05-0.1 N·m. Three geometries can, on paper:
1. **Stance length.** Rotate the pads so the stalk lies along the 36 mm length, and put rubber ridges at both ends. This gives 0.07-0.14 N·m at 4 N with μ = 0.5-1.
2. **V-grooves along the stalk on both jaws, with parallel-guided pads.** Pivoting then forces the stalk to climb the flanks, which is resisted by the springs. The frictionless estimate is F·L·cot β: 0.09 N·m at 4 N and 0.23 N·m at 10 N for a 60° V and L = 13 mm.
3. **Form closure.** A fork or hook under the truss takes the moment directly.

Option 2 is the most force-efficient. It is also the least certain, because stalk compliance erodes it.

### Cited Findings
- **[THEORY-CITED + real prototype]** The two-phase gripper uses exactly the pivot-vs-lock switch:
  - **pivot phase:** a small rubber point contact on an elastic strip over a V-groove gives low torsional friction, so the part swings under gravity;
  - **lock phase:** a higher grip force pushes the strip into the cavity and the part seats in the V-groove ("multi-point contact with higher friction"), which forms a firm grasp.

  The prototype values are μ ≈ 0.6 (estimated) and a grip-force range of 5-30 N. The pivot and lock conditions themselves are formulas, not measurements. Source: [US9808936B2](https://patents.google.com/patent/US9808936); [Chavan-Dafle et al., CASE 2015](https://dspace.mit.edu/handle/1721.1/107782)
- **[Real prototype]** A follow-up builds a fingertip-based two-phase gripper with a spring-and-shaft mechanism that switches between pivot and grasp mode by grip force alone, with no extra motor. Source: [arXiv:1912.02752, "Reorienting Objects in 3D Space Using Pivoting"](https://arxiv.org/pdf/1912.02752) (summary from search; full text not read)
- **[SNIPPET]** Industrial V-on-both-jaws 4-line contact is preferred for high-speed holding of round parts. Source: [American Machinist](https://www.americanmachinist.com/cutting-tools/article/21895163/lets-give-robots-a-hand)

### Inferences (my derivations)

**V-groove against pivot.**
- Pivoting by a small angle moves the stalk ends ±(L/2)θ sideways across the groove.
- Set-up: two V jaws of half-angle β, frictionless, rigid stalk, and pads that stay parallel with F/2 per end.
- The lateral force each end can take before the jaws separate is Q = 2(F/2)·cot β = F·cot β.
- Pivot capacity: M ≈ Q·L = F·L·cot β, plus a friction term.

| L | Included angle | 4 N | 10 N |
|---|---|---|---|
| 13 mm | 90° | 0.052 N·m | 0.13 N·m |
| 13 mm | 60° | 0.090 N·m | 0.23 N·m |

These are upper bounds. A green 3-5 mm stalk will bend and flatten, so the effective L and β shrink. The real value is unknown until it is measured.

Conditions for this to work:
- the pads must not be free to tilt;
- the groove must be deep enough that the smallest (3 mm) stalk still sits below the jaw faces. For a 60° V, the stalk centre sits 2R from the apex: 3 mm for R = 1.5 mm. A groove depth of about 2 mm leaves a jaw-face gap of 2 mm for a 3 mm stalk and about 6 mm for a 5 mm stalk.

**Wide stance.**
- Two short rubber ridges, or two separate pinch points, spaced D apart along the stalk give μ·F·D (table in section 1).
- Doubling D doubles the hold at the same servo force.
- The 36 mm pad dimension is the cheapest lever if the stalk has at least about 30 mm of straight length between the cut and the first fruit branch (unknown, see Gaps).

**Cross-grooves perpendicular to the stalk.** These resist sliding along the stalk, not pivoting. They do little for this failure mode.

**Pins, teeth and forks (form closure).**
- A fork or hook that the truss's main stalk rests against, a few cm from the pinch point, converts the gravity moment into a contact force at that hook.
- For 0.08 N·m and a 2 cm lever, the hook force is 4 N. That force rests on the stalk, not on the fruit.
- It needs no clamp force at all. This is the industrially "correct" answer for an off-axis load, since a form-fit grip does not depend on friction.

The specific statement that gripper makers prefer form-fit for this case is from general knowledge; the Festo source was not readable.

### Gaps
- The straight peduncle length available between the cut and the first fruit branch is unknown. The 36 mm stance option needs about 30 mm or more. Turning the stalk along the 36 mm pad dimension also changes how the jaw has to approach the stalk.
- No V-jaw patents for cables or tubes with numeric angles were read.
- No cable-harness, test-tube, pipette or needle-holder gripper guides were read. Needle holders use cross-hatched (diamond) serrated jaws, but I have no fetched source for that.
- No measured pivot-hold numbers for V-grooved jaws.

---

## 4. Robotics literature: gravity pivoting with parallel grippers (pivot vs stay fixed)

### Takeaway
The literature treats gravity pivoting as a deliberate primitive. Gripping force acts as a brake on rotation, and the switch between pivot and hold is set by the ratio of gravity torque to torsional friction capacity, μ·r_eq·F. Every hardware system that wants a fixed grasp after pivoting adds a groove, which is evidence that friction alone at low force is not a robust hold.

### Cited Findings
- **[THEORY-CITED; prototype parameters]** The two-phase gripper's pivot condition F ≤ M·g·L·cos φ/(μ·r) and slip condition F ≥ Mg/μ. These are formulas; the prototype ran at 5-30 N with an estimated μ ≈ 0.6. Source: [US9808936B2](https://patents.google.com/patent/US9808936)
- **[Real hardware]** Gravity pivoting is controlled by loosening the grip on a Robotiq 2F-85 fitted with PapillArray tactile sensors. The approach is model-free: an LSTM estimates the angle and velocity from tactile data, ten household objects were used, and it "embrace[s] slip". It notes that earlier model-based work needed the object's mass, shape and friction. Source: [Toskov, Newbury, Mukadam, Kulić, Cosgun, "In-Hand Gravitational Pivoting Using Tactile Sensing", CoRL 2022, arXiv:2210.05068](https://arxiv.org/pdf/2210.05068)
- **[Real hardware]** Pivoting boxes of 0.88-1.72 kg against a surface with a Robotiq 2F-85, PapillArray sensors and an FT-300. The grip is closed until the tactile normal force exceeds a threshold that satisfies μ_s·F_N > f_p (the initial pivot force). This prevents translation while allowing rotation. No N values were reported. Source: [arXiv:2303.10865, "Rotating Objects via In-Hand Pivoting using Vision, Force and Touch"](https://arxiv.org/html/2303.10865v3)
- **[Real hardware, MEASURED]** The sensorised parallel gripper above estimated μ_c of 0.36-0.54 and equivalent torsional radii of 3.9-7.9 mm on a 15 mm pad. It used ellipsoidal force-torque coupling to command the minimum normal force against slip, at grasp forces of 5 N up to about 45 N. Source: [arXiv:2410.19660](https://arxiv.org/html/2410.19660)
- **[Summary only]** A KTH ICRA 2016 paper uses gravity with the grip force acting as a brake, with tactile-feedback adaptive control that tolerates friction-coefficient errors. It builds its torsional model on limit surfaces and soft-finger mechanics results. Source: [Viña et al., ICRA 2016](https://www.csc.kth.se/~fevb/publications/pdfs/icra16.pdf) (the PDF returned 404, so this rests on the search summary only)
- **[Summary only]** Costanzo, De Maria and Natale control the pivoting angle and sliding velocity with a force/tactile-sensorised parallel gripper. They use a limit-surface plus LuGre soft-contact model in which "soft contacts allow torsional frictions as well as friction forces". Source: [Costanzo thesis / UniCampania repository](https://iris.unicampania.it/handle/11591/481950); [CORE copy](https://core.ac.uk/download/521169272.pdf) (not read in full)

### Inferences
- With the 13 mm line contact (effective torsional radius of about L/4 per contact for uniform pressure), the gravity-pivot threshold for the truss is roughly F_hold ≈ 2τ/(μ·L):
  - 6-12 N at μ = 1;
  - 12-25 N at μ = 0.5.

  That is at or above the servo's 4-10 N, so the truss is in the pivoting regime of this literature by default.

### Gaps
- No paper found pivots a slender compliant rod (stalk) specifically. All examples are rigid boxes or cylinders.
- No quantitative torsional model for a line contact in the pivoting literature I could read.

---

## 5. Friction of candidate pad faces on stems and waxy or moist surfaces

### Takeaway
On a plant stalk:
- rubber faces measured about 0.8-1.0 (reed, dry);
- hard plastics and metals measured 0.12-0.37, so bare PLA is the worst choice;
- TPU95 against everyday objects measured about 0.4-0.55.

Wet or sap-covered tomato stalk data were not found. Textured soft silicone gives more stable (not higher) friction when wet.

### Cited Findings
- **[MEASURED]** Common reed (*Phragmites australis*) stalks:
  - against rubber: μ = 0.81 (transverse) to 1.02 (longitudinal);
  - against hard materials (metals and plastics): μ = 0.12-0.37, with no significant effect of stalk diameter for hard materials;
  - moisture not stated in the abstract.

  Source: [Vronskis & Kakitis, "Friction properties of reed stalks", Engineering for Rural Development 2023, pp. 427-432, DOI 10.22616/ERDev.2023.22.TF091](https://agris.fao.org/search/fr/records/65142f5d925d9e099bc6f9e2)
- **[MEASURED]** 3D-printed flat pads (PLA or TPU95) on cardboard, plastic, wood and a phone case: μ_c 0.36-0.54 and μ_s 0.40-0.57. TPU95 gave higher friction than PLA. Source: [arXiv:2410.19660](https://arxiv.org/html/2410.19660)
- **[MEASURED]** Two-phase gripper rubber fingertip on its test object: μ ≈ 0.6. Source: [US9808936B2](https://patents.google.com/patent/US9808936)
- **[SNIPPET]** Silicone gripper pads of Shore 10A and 30A were tested wet and dry. Textured pads gave more stable friction, especially when wet, but somewhat lower peak friction. Source: [Int. J. Adv. Manuf. Technol. 2025, s00170-025-16625-1](https://link.springer.com/article/10.1007/s00170-025-16625-1)
- **[SNIPPET]** Textured silicone rubber showed a significantly higher coefficient of friction than smooth silicone under wetting at small normal force. Source: [Fretting behaviour of biomimetic-textured silicone rubber under varying wetting conditions, PMC12387959](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12387959/)
- **[SNIPPET]** A kiwifruit soft-gripper thesis found silicone gave inadequate friction compared with a high-friction flexible material. Source: [University of Waikato research commons](https://researchcommons.waikato.ac.nz/bitstreams/4fb77e6f-5b4a-40b1-9091-78ff8cee531a/download)
- **[SNIPPET]** Pinecone static friction was measured on spring steel, rubber and silicone plates, with moisture content as a factor. Values were not read. Source: [BioResources 19(1) 766](https://bioresources.cnr.ncsu.edu/wp-content/uploads/2023/12/BioRes_19_1_766_Ren_TGZLD_Frictional_Props_Red_Pinecones_Harvest_22852.pdf)

### Inferences
- For design, use μ ≈ 0.8 for a rubber or soft-TPU face on a dry stalk, and plan for a wet or sap derating to perhaps 0.5. That derating is a guess.
- Only geometric holds (V-groove climbing, ridge stance, hook) keep working when μ collapses. This argues for geometry first and material second.
- Bare PLA (μ ≈ 0.12-0.37 on stalks) should not touch the stalk.

### Gaps
- No measured μ for any pad material on a tomato peduncle, wet or dry.
- No data on the effect of tomato sap or of the waxy or trichome-covered stalk surface.
- No Shore-hardness-to-friction curves on plant tissue.

---

## 6. Hang the centre of mass below the grip, or resist the torque?

### Takeaway
Gripping so the centre of mass already hangs below the grip removes the gravity torque on both the pivot and spin axes. That brings the required holding torque toward zero instead of 0.04-0.08 N·m. Given the numbers above, this is the most robust option at 4-10 N. The cost is that any later change in gripper orientation re-creates the torque, unless the truss is allowed to swing freely and is damped.

### Cited Findings
- **[THEORY-CITED]** In the two-phase gripper, gravity torque scales with M·g·L·cos φ. It vanishes when the centre of mass is directly below the contact (φ = 90°), and that alignment is exactly where a pivoting part settles. Source: [US9808936B2](https://patents.google.com/patent/US9808936)
- **[Real hardware]** Loosening the grip lets gravity rotate a held object about the grip point. That work controls the rotation to a target angle. Source: [Toskov et al., CoRL 2022](https://arxiv.org/pdf/2210.05068)

### Inferences
- An uncontrolled loosened grip ends with the centre of mass hanging below the grip. That is the zero-torque equilibrium (my inference from statics, not a claim of the cited papers).

Two strategies follow.

**(a) Pre-align, then hold.** Grasp or reorient so the stalk is vertical above the truss centre of mass, and keep the wrist orientation fixed relative to gravity during the carry. The residual torque is then only from accelerations: m·a·d, for example 0.085 kg × 2 m/s² × 0.05 m ≈ 0.009 N·m. Flat rubber pads at 4 N handle that (0.026 N·m). Arm accelerations should be limited; the 2 m/s² figure is an assumed value.

**(b) Two-phase.** Pinch lightly and let the truss swing to hang below, then raise the force or seat the stalk in a V-groove to lock it. This is the published two-phase-gripper pattern.

Accept the spin about the stalk axis, which ends with the fruit below. It is the minimum-energy state, and the spin-holding capacity at 4-10 N is small anyway (section 2).

### Gaps
- I found no source with a quantitative design rule like "grip within X of the centre-of-mass vertical". The industrial "grip close to the centre of gravity" guidance in gripper-maker sizing documents (for example Festo) could not be fetched (403).
- No published data on swing damping of a hanging fruit cluster during robot carry.
