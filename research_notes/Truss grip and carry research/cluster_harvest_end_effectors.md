# Cluster-harvest end effectors: how real robots hold a cut truss / bunch and get it into a container

Compiled 2026-11-25. Scope: tomato truss / cherry-tomato bunch clamp-and-cut end effectors first, then grape, litchi, kiwifruit and banana where the lessons transfer.

Evidence labels used in every bullet:
- **[HW-field]**: real hardware, in a greenhouse or field.
- **[HW-lab]**: real hardware in a lab or on a test bed.
- **[SIM]**: simulation only.
- **[PATENT]**: patent text. It shows design intent, not test results.
- **[MKT]**: a marketing or press claim.
- **[SNIPPET]**: the number came from a search-engine summary. I could not verify it in a full text I read. Treat it as unverified, and do not tie it to a specific paper without checking.
- **[INFERENCE]**: my own reasoning. It is not a sourced fact.

**Read directly as full text (PDF-to-text):**
- Kondo et al. 2010, EAEF
- Kondo et al. 2009, EAEF
- Kondo 1995, JARQ
- AHPPEBot (arXiv 2405.06959)
- de Haan et al. 2021 (arXiv 2103.01272)
- Kok 2024, TU Delft MSc thesis
- the Dutch vocational module "Oogsten onder glas"

**Read through a web-fetch summariser (full page, but I saw a model's extraction, not the raw text):**
- the Denso patent NL2026342B9
- Jiang et al. 2022 (grape)
- Fu et al. 2024 (kiwifruit)
- Zheng et al. 2022
- van den Bent et al.
- the Certhon, Ridder and HortiDaily pages
- the two Chinese abstracts (Li & Sun 2025; Yuan et al. 2013)

Quotes from this group are the summariser's quotes of the source. Where I was unsure whether wording was verbatim, I give it as a paraphrase without quote marks.

Everything else comes from search summaries and is labelled [SNIPPET].

**Blocked sites:** IJABE (bot protection), MDPI (HTTP 403), the DENSO newsroom (HTTP 403) and IEEE Xplore (empty page).

---

## 1. Which robots harvest whole tomato trusses / cherry-tomato bunches (and grape bunches) by clamp-and-cut, and how is the peduncle held after the cut?

### Takeaway
Almost every clamp-and-cut cluster harvester holds the cut cluster by pinching the peduncle in a spring-loaded grasper next to the blade, and lets the cluster hang below the grip. Two designs do something more:
- **Denso patent:** uses acute "biting" protrusions plus a hard-stop gap set slightly below the stalk diameter, so the stalk is not crushed and does not slip on its own juice.
- **AHPPEBot:** does not hold the stalk at all. The cut truss drops into a mesh pocket under the blade.

Rubber-plated or foam-covered pinch fingers are the norm in tomato research. Grape tools add a rubber block and, in one case, a pusher that presses on the top of the bunch.

### Cited Findings

#### Tomato truss / cluster: research prototypes

**Kondo, Yata, Iida, Shiigi, Monta, Kurita, Omori 2010, "Development of an End-Effector for a Tomato Cluster Harvesting Robot"** (Engineering in Agriculture, Environment and Food 3(1):20-24) [HW-field: greenhouse, 20 clusters on a SCARA arm] — [J-STAGE PDF](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_pdf)
- **Holding mechanism:** The upper fingers surround the main stem. A ball screw and DC motor move them down onto the peduncle. Inside the upper finger, a spring-loaded "peduncle grasper" slides to grasp the peduncle, and a cutter in the same finger cuts it. The grasp is therefore spring-loaded and sits right next to the cut.
- **Lower fingers:** They carry two strain gauges that sense the peduncle as the tool slides up the main stem.
- **Clamp force:** "The end-effector can grasp the peduncle between the peduncle grasper and lower finger with a force of 20 N for a peduncle diameter of 8 mm, because its spring constant is 2.33 N/mm." An air cylinder opens and closes both fingers through a linkage.
- **Force requirements:** "Since the larger fruit cluster mass was 400 g, a frictional resistance of at least 5N was necessary to collect the fruit cluster. It was, however, determined that 40 N was necessary as peduncle grasping force of fingers for robot's quick movement with a harvested fruit cluster."
- **Measured plant dimensions (high-wire system):**
  - Peduncle diameter: 4.8–11.8 mm (mean 7.0)
  - Peduncle length from main stem to first fruit: 33.0–131.6 mm (mean 79.0)
  - Entire peduncle length: 97.5–226.9 mm (mean 148.4)
  - Fruit diameter: 43–63 mm
  - Peduncle angle: 5–139° (mean 69.5)
- **Cutting resistance:** "The cutting resistance of peduncles was around 60 N for the largest peduncle size."
- **Scale caveat:** These are large-fruit clusters of 4–6 fruit, up to 400 g, on 4.8–11.8 mm peduncles. A 3–5 mm, 70–100 g cherry truss is a much smaller load.

**Kondo, Tanihara, Shiigi, Shimizu, Kurita, Tsutsumi, Chong, Taniwaki 2009, "Path Planning of Tomato Cluster Harvesting Robot for Realizing Low Vibration and Speedy Transportation"** (EAEF 2(3):108-115) [SIM + HW-lab: 1-DOF slide with 5 real clusters] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/2/3/2_3_108/_article/-char/en)
- In the real-cluster experiment, "The end effector has a 34-Nm [sic] grip force and hold top of cluster." The unit is as printed. It is most likely 34 N, but that is unverified.
- The cluster hangs from that top grip, and the paper models it as a chain of rigid links with rotational springs and dampers.

**Yuan, Li, Ji, Zhang, Yu, Yang 2013, 温室串番茄采摘机器人关键技术研究 ("Research on key technology of truss tomato picking robot in greenhouse")** (China Science Paper Online preprint, DOI 10.12383/SciencepaperOnline.201302.0478) [HW-field: greenhouse, abstract only] — [abstract page](https://sinoxiv.napstic.cn/article/5595751)
- A "scissor-type end-effector based on flexible transmission" cuts and grasps at the same time.
- Harvest success was 86.67%, with an average of 37.57 s per successfully picked truss.
- **Probably the same group in English** ([INFERENCE] from the matching title and the overlapping author Ji Chao): Ji et al. 2014, "Research on key technology of truss tomato harvesting robot in greenhouse," Applied Mechanics and Materials 442:480-486. As cited by de Haan et al.: "After cutting, the truss is pinch-grasped with two rubber-plated fingers at the end of the peduncle. As a result, the grasped truss hangs almost vertically downwards, making accurate placement difficult. Kondo et al. used a similar approach" — [de Haan et al., arXiv 2103.01272](https://arxiv.org/abs/2103.01272)
- **Conflicting secondary citation:** Zheng et al. 2022 cites "Ji et al. 2013" as a rigid shear-and-grip end effector with "37.2 s per fruit" and notes that the blade can damage fruit — [Zheng et al. 2022, PMC9465300](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465300). The abstract says 37.57 s per truss. The secondary citation appears to garble it.

**AHPPEBot: Li, Ma, Han, Yang, Zheng, arXiv 2405.06959 (May 2024)** [HW-field: commercial greenhouse] — [arXiv](https://arxiv.org/abs/2405.06959)
- **End effector:** It has "guiding grooves and a saw blade. The groove size is tailored based on the width of the tomato vine and truss peduncle, ensuring that only the peduncle can fit and be cut… once the peduncle is severed, the tomato truss temporarily collects in a mesh pocket."
- **Motion sequence ("bottom-up wrapping"):**
  - (a) start below the truss;
  - (b) rise slowly along the peduncle curve until all fruits are enveloped;
  - (c) bring the blade edge to the stem point (SP), where the peduncle meets the main stem;
  - (d) rotate the end effector so the peduncle falls into the slot and is cut;
  - (e) keep pressing by rotation;
  - (f) "the peduncle is severed and falls into the collection mesh."
- **Holding:** Nothing grips the peduncle after the cut. The truss is caught from below.

**de Haan, Kulkarni, Babuška 2021, "Geometry-Based Grasping of Vine Tomatoes"** (arXiv 2103.01272, TU Delft) [HW-lab: post-harvest pick-and-place from a table, plastic tomatoes] — [arXiv](https://arxiv.org/abs/2103.01272)
- They used an Interbotix PincherX 150, a 5-DOF hobby arm with a 50 g payload, which is close to our arm class.
- The fingers are L-shaped parallel-gripper fingers "covered with deformable foam… rubber knobs were attached to these fingers such that they can enclose the peduncle."
- This is a caging grasp: the rigid parts enclose the stalk and the soft parts apply only a small reaction force.

**van den Bent, Coleman, Babuška, "Robotic Grasping of Harvested Tomato Trusses Using Vision and Online Learning"** (ICRA 2024; arXiv 2309.17170) [HW-lab: post-harvest, Franka Panda] — [arXiv HTML](https://arxiv.org/html/2309.17170v2)
- Custom 3D-printed slim fingers "covered with grippy, deformable foam to minimize potential damage to the stem and tomatoes"
- A top-down pinch grasp on the peduncle.
- Kok 2024 states that van den Bent et al. "applies a force of 40 N" — [Kok 2024 thesis](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)

**Kok 2024, "Specialised Hook Gripper For Robotic Grasping of Vine Tomato Trusses"** (TU Delft MSc thesis) [HW-lab: post-harvest crate emptying] — [TU Delft repository](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)
- **Why a hook:** "The gripper proposed… grasps the truss with a hook around the peduncle often optimally at the centre of mass." The thesis says this improves success, stability and damage avoidance.
- **Form closure from below:** "This hook is strategically positioned underneath the peduncle, effectively encaging it to prevent slippage… Unlike a pinch grip which requires overcoming both the weight and friction between the peduncle and fingers, the hook's design only needs to counteract the weight of the peduncle and truss."

**Li Xiaoping & Sun Bingyu 2025, 圣女果串采收末端执行器的设计与实验 ("Design and Experiment of Cherry Tomato Bunch Picking End Effector")** (Journal of Agricultural Mechanization Research 47(3):103-110, DOI 10.13427/j.issn.1003-188X.2025.03.016) [HW-lab: indoor tests, abstract only] — [abstract](https://search.napstic.cn/literature/periodical/010njhyj202503017)
- A planar four-bar linkage drives a combined clamp-and-shear mechanism. The clamp has a "self-compensating" design, and screw theory is used to check force closure.
- Whole-bunch clamping success was 100%; shearing success was 86.7%.
- The abstract gives no clamp force, pad material or post-cut holding details.

**Fujinaga et al. 2021: a "rigid tip for clusters"** with 50% success, called too bulky — as cited by [Zheng et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465300)
- Bhourji et al. 2026 instead cite Fujinaga et al. 2021 as a suction-cutting device with 52.4% success — [arXiv 2610.18160](https://arxiv.org/abs/2610.18160).
- The two secondary sources disagree. I did not read the primary paper.

**Feng et al. 2018, "Design and test of robotic harvesting system for cherry tomato"** (IJABE 11(1):96-100, DOI 10.25165/j.ijabe.20181101.2853) — [IJABE](https://ijabe.org/index.php/ijabe/article/view/2853)
- [SNIPPET] The end effector is "based on the stalk's mechanical features" for mature bunches. "Each successful harvest averagely needed 1.4 times attempt," and a successful cycle took 8 s excluding travel. The full text was blocked by bot protection, so I could not verify the clamp design.
- [SNIPPET] A figure of 83% for a scissor-style shear-and-clamp bunch gripper appeared in a search summary. I could not confirm that it belongs to this paper.

#### Tomato truss / cluster: commercial and patents

**Denso Corp patent NL2026342B9, "End-effector for crop harvesting and crop harvesting system"** (filed 2020-08-26, priority 2019-09-06) [PATENT] — [Google Patents](https://patents.google.com/patent/NL2026342B9/en)

How the tool works:
- The cluster passes up through a rectangular passage inside a rigid annular "moving member", which is metal or resin, frame about 3–5 mm thick, with rollers so that fruit do not catch.
- A servo, screw and ball nut retract the moving member, which shrinks the passage and presses the stalk onto plate cutters.

How it holds the stalk:
- A gripping member pushed forward by a coil spring clamps the stalk against a receiving portion.
- Both the grip and the receiver carry protrusions with acute profiles. Each protrusion "does not cut off the fruit stem but cuts into the fruit stem."
- Locking portions hold the grip and receiver at a gap L1, which "is preferably set to be a value greater than 0 and slightly less than an average value of outer diameters of fruit stems."
- The stated reason for the gap: when a stalk is crushed, "water (fruit juice) leaks from the fruit stem… the fruit stem becomes slippery," so it can "slip off" and "the harvesting target is likely to fall."

Release and other details:
- After the cut, the arm carries the cluster "while gripping the fruit stem" and "releases the fruit stem at the position immediately above the collection box." Opening the passage ends the grip.
- A load detector flags entanglement and triggers a release-and-retry routine.
- No mm values for L1, no stalk diameters and no spring force are given.
- The patent does not discuss swinging or rotation of the cluster.
- [INFERENCE] I assume this patent relates to Denso's commercial truss robot, but nothing I read confirms it.

**Denso EP4140286, "Crop harvesting method, crop harvesting device, and crop harvesting system"** [PATENT, SNIPPET] — [Google Patents](https://patents.google.com/patent/EP4140286)
- It targets cluster crops such as cherry tomatoes and grapes, with a separation mechanism that moves along the stem without gripping it.
- I did not read the full text.

**Artemy (Denso and Certhon), cherry-truss tomato robot, commercial orders in Europe from May 2024** [MKT]
- [SNIPPET, paraphrase] The scissors cut the peduncles and place the fruit in crates — search summary of [DENSO newsroom 2024-05-13](https://www.denso.com/global/en/news/newsroom/2024/20240513-g01/). That page returned HTTP 403 to me.
- [SNIPPET, paraphrase] The scissors are multifunctional: they grab and cut the truss in one action, in the correct position. When all six crates are loaded, the robot swaps all crates at once at a trolley — search summary of trade-press coverage, e.g. [ProduceGrower](https://producegrower.com/news/artemy-cherry-truss-tomato-harvest-robot-denso-certhon-produce-greenhouse-growing) and [FreshPlaza](https://www.freshplaza.com/europe/article/9625556/a-fully-automated-cherry-truss-tomato-harvesting-robot-to-hit-the-market/). The FreshPlaza page I fetched did not contain these sentences.
- **Verified on Certhon's page:**
  - a "cluster and peduncle detection LED";
  - paraphrase: when all crates on the robot are full, it drives to a cart of empty crates and replaces them all at once;
  - no gripper details and no performance numbers.
  
  Source: [Certhon](https://certhon.com/denso-and-certhon-introduce-artemy-a-fully-automated-cherry-truss-tomato-harvesting-robot/)

**Ridder Harvesting Robot** (cherry-truss, cocktail-truss, mini-plum and vine tomatoes) [MKT] — [Ridder](https://ridder.com/harvesting-robot)
- **Hardware (paraphrase):** a smart-cutting shear mechanism that finds the cutting position, cuts and secures the truss; dual arms.
- **Crate handling (paraphrase):** an onboard boxing system moves trusses into standard reusable containers, which are stacked automatically on a towed trolley.
- **Claims:** up to 80% labour saving.
- **Missing:** gripper details, speed and success rate.

**inaho snack-tomato robot:** summer 2026 pilots at Dutch and Belgian growers [MKT] — [HortiDaily 2026-11-11](https://www.hortidaily.com/article/9890382/harvest-robot-completes-summer-pilots-at-two-european-growers/)
- The article does not say whether the robot cuts whole trusses or picks loose fruit. It mentions a "more compact gripper" planned for 2027.

#### Grape and other cluster crops

**Kondo 1995, "Harvesting Robot Based on Physical Properties of Grapevine"** (JARQ 29:171-177) [HW-lab + HW-field] — [JIRCAS PDF](https://www.jircas.go.jp/sites/default/files/publication/jarq/29-3-171-177_0.pdf)
- **Force sizing:** The end effector grasps and cuts the rachis (bunch stalk). From grasp-force/friction and cross-section/cutting-resistance tests, "the grasping force was 10 N and the cutting force was 100 N."
- **Single drive:** One DC motor plus two springs drive both the finger and the cutter.
- **Pusher:** "a function of pushing of bunch was added to this end-effector, to enable the end-effector to grasp a very short rachis also at harvest time, to reduce the swinging of the bunch at the time of carrying, and to orient the bunch at the time of release." The pusher is a separate DC motor with rack and pinion that pushes straight.
- **Design goals:** the bunch must not shatter, and the bloom (white powder) must not be rubbed off.

**Jiang, Liu, Wang, Li, Peng, Shan 2022, "Development of a dual-arm rapid grape-harvesting robot for horizontal trellis cultivation"** (Frontiers in Plant Science 13:881904, DOI 10.3389/fpls.2022.881904) [HW-field + HW-lab] — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.881904/full)
- A "cut-clip" end effector: motor-driven fingers, three sets of blades that cut the stem as the fingers close, and (paraphrase) a rubber block on the lower part of the fingers holds the cluster while the stem is cut. The article states that "The end effector must grip the whole grape cluster when cutting the stem."
- Curved finger shapes widen the tolerance to position errors in x and y.

**Retractable four-point flexible clamping end effector for grape** (Agronomy 15(12):2813, 2025) — [MDPI](https://www.mdpi.com/2073-4395/15/12/2813)
- [SNIPPET, SIM] Simulated peak clamping force was 11 N, "below the grape's critical rupture force of 24.79 N." The same article reportedly also gives 25.79–34.54 N elsewhere, so it is inconsistent.
- [SNIPPET] Average picking speed was 6.18 s per bunch.
- The clamp-force figures are simulation, not hardware.

**Autonomous grape harvester** (Electronics 10(9):1056) [SNIPPET] — [MDPI](https://www.mdpi.com/2079-9292/10/9/1056)
- A gripper with scissors and 3D-printed fingers, "designed for cutting and holding the grape cluster from the stem."

**Fu, Guo, Chen, Cheng, Cui 2024, "Design and experimentation of multi-fruit envelope-cutting kiwifruit picking robot"** (Frontiers in Plant Science 15:1338050, DOI 10.3389/fpls.2024.1338050) [HW-lab: test bed and trellis model] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10875067)
- An envelope-type picking bin goes under the cluster: carbon-fibre plates with a hexagonal port of about 90 × 135 mm, holding 6–8 fruit.
- An internal rotary cutter severs the stalks.
- Fruit rest on curved rods covered with polyethylene foam, which (paraphrase) buffer and guide them to a collection end and then a hose. Nothing grips the stalk.

**Banana:** Chen et al. 2024, Agriculture 14(8):1389 [SNIPPET] — [DOAJ](https://doaj.org/article/b35ee26cf2ad48dcbf857dac0da62164)
- An "integrated end-effector capable of clamping and cutting tasks on the banana stalks continuously."

**Litchi** [SNIPPET]: litchi work generally treats the fruit-bearing branch, not the fruit, as the object to grasp and cut. The search summary drew this from [Agronomy 13(11):2795, 2023](https://mdpi.com/2073-4395/13/11/2795).

#### Per-robot summary table

| System | Evidence | Post-cut hold | Pad / face | Clamp force | Cluster after cut | Into container |
|---|---|---|---|---|---|---|
| Kondo 2010 tomato cluster | HW-field | Spring-loaded grasper beside the cutter, in the upper finger | not stated | 20 N at 8 mm (k = 2.33 N/mm); 40 N judged necessary for quick moves | Hangs below the grip | The motion list opens both fingers, then moves back; the method text says it "release[s] the tomato cluster into a harvest tray". How the stalk is held between cut and tray is not explained |
| Kondo 2009 transport | HW-lab + SIM | Holds the top of the cluster | not stated | "34-Nm [sic]" | Hangs; modelled as a 3-link pendulum | 0.5 m transfer to a basket |
| Ji / Yuan 2013–14 truss | HW-field (abstract) + secondary | Scissor with simultaneous grasp; rubber-plated fingers at the peduncle end | rubber | not found | "hangs almost vertically downwards" | not found |
| AHPPEBot 2024 | HW-field | No stalk hold | — | — | Drops into a mesh pocket | not described |
| Denso NL2026342 | PATENT | Spring grip plus acute biting protrusions, hard-stop gap < stalk diameter | resin/metal; protrusions | not given | Carried by the stalk | Released directly above the box; can drop onto a conveyor |
| de Haan 2021 | HW-lab, post-harvest | Caging pinch near the centre of mass | foam plus rubber knobs | small (caging) | Lifted near the CoM to avoid tilt | Placed back on a table |
| van den Bent 2024 | HW-lab, post-harvest | Top-down pinch | 3D-printed slim fingers with grippy foam | 40 N (as stated by Kok) | — | — |
| Kok 2024 hook | HW-lab, post-harvest | Hook under the peduncle near the CoM | not stated | weight only (form closure) | Stable lift | Placed while still supported |
| Kondo 1995 grape | HW-lab/field | Finger plus cutter on one motor, plus a pusher on the bunch | not stated | 10 N grasp, 100 N cut | Pusher reduces swing and orients the bunch | Oriented at release |
| Jiang 2022 grape | HW-field | Cut-clip fingers | rubber block | not stated | not stated | Moved to a "grape-putting position" |
| Fu 2024 kiwifruit | HW-lab | No stalk hold | PE-foam rods | — | Fruit fall into a bin | Hose to a box |

### Inferences
- [INFERENCE] Clamp-and-cut tomato tools split into two holding philosophies:
  1. Pinch the stalk and let the cluster hang under gravity: Kondo, Ji/Yuan, the Denso patent, Ridder (whose shear, per its page, also secures the truss), and probably Artemy.
  2. Do not hold the stalk; catch the cluster from below: AHPPEBot's mesh pocket, the kiwifruit bin.
  
  None of the tomato tools I found tries to hold the truss rigidly in a fixed pose by the stalk alone. Kondo's grape pusher is the only hardware example of adding a second contact to stop swing and set the orientation.
- [INFERENCE] The Denso patent is the most directly copyable detail for a 3D-printed jaw:
  - small acute ridges or teeth on both pads that bite into the stalk skin;
  - a mechanical hard stop that leaves a gap slightly smaller than the stalk diameter.
  
  Together these give a geometric grip that does not depend on squeezing harder, which avoids the crush, juice, slip failure.

### Gaps
- No clamp force, pad material or pad dimensions were found for any commercial truss robot (Artemy, Ridder, inaho). Only marketing descriptions are public, and the Denso newsroom page blocked automated access.
- I did not read full texts for the Chinese cherry-tomato bunch end effectors. Only abstracts were available, and one site was behind bot protection. Their clamp forces, jaw materials and post-cut hold details are therefore unknown.
- Search summaries gave these cherry-tomato bunch figures, which I could not tie to any paper I read. All are unverified:
  - a gear and five-link "bottom-up enveloping" bunch manipulator: 87% success, 7% fruit damage, 10.43 s per string;
  - a single-actuator combined cutting-and-gripping end effector: 93.33% cutting success, 1600 g gripping capacity;
  - an integrated bunch-harvesting system: 12.23 s per bunch, 70.77% success;
  - measured stem diameters of 2.48–5.12 mm.
- Two unread leads from the first search may hold some of these:
  - IEEE Xplore document 10021390, "Design and Experiment of a Novel Manipulator for Autonomous Harvesting Tomato Clusters" (https://ieeexplore.ieee.org/document/10021390/), which returned an empty page to my fetch;
  - a preprint titled "Design, Development, and Field Testing of a Tomato Bunch …" on a Frontiers cloudfront host, which would not resolve (DNS failure).
- A search summary described a lychee cluster tool that pairs dual gripping fingers with an eccentrically actuated cutting blade, with 78% orchard success. I could not tie it to a specific paper. The candidates were MDPI Agriculture 15(14):1537 and 16(6):686, both blocked.
- A search summary gave banana 91.69% success / 33.28 s (Chen et al. 2024). This was not verified in the full text.
- I did not obtain the full texts of the Ji et al. 2014 paper or the PicknPack reports, which describe the rubber-plated pinch fingers.

---

## 2. After the cut: does the cluster hang, is it cradled from below, or dropped into a chute? What stops swing, rotation in the grip, or fruit shedding?

### Takeaway
In the tomato hardware I found, the cut cluster either hangs from a stalk pinch or drops into a pocket or bin under the cutter. Three things were used specifically against swing:
- feed-forward input shaping of the arm's acceleration, tuned to each cluster's natural frequency, which is about 1.5–1.9 Hz (Kondo 2009);
- a pusher that presses on the bunch during transport (Kondo 1995, grape);
- grasping near the centre of mass, or using a hook under the stalk, to stop tilt (de Haan 2021; Kok 2024; post-harvest only).

No source I found measures or names rotation of the cluster within the grip as a failure mode. The documented problems are slip-out, swing/wobble, tilt and vertical hanging.

### Cited Findings

**Hanging from a stalk pinch**
- **Ji et al., as cited by de Haan:** "the truss is pinch-grasped with two rubber-plated fingers at the end of the peduncle. As a result, the grasped truss hangs almost vertically downwards, making accurate placement difficult" — [de Haan et al. 2021](https://arxiv.org/abs/2103.01272)
- **Kondo 2010 sequence:** "The peduncle is grasped and cut… Both upper and lower fingers open. Manipulator moves back to release the harvested fruit cluster." In the experimental method: "The manipulator moves the end-effector to release the tomato cluster into a harvest tray." [HW-field] — [Kondo et al. 2010](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_pdf)
- **Kondo 2009 definition:** "Transportation" means "the movement from the position where a tomato peduncle of a tomato cluster is cut with an end effector to a basket for storing." "Both speedy transportation and vibration damping are required when a tomato cluster is transported into a box or a basket," and "damping vibration of a tomato cluster is also desired for preventing the fruit from spoiling." — [Kondo et al. 2009](https://www.jstage.jst.go.jp/article/eaef/2/3/2_3_108/_article/-char/en)

**Kondo 2009: the swing problem, quantified** [SIM + HW-lab]

- **Measured natural frequency:** first-mode frequency of the whole hanging cluster, with one end fixed, from free decay of an accelerometer near the cluster's centre of mass, across 8 clusters:
  - max 1.93 Hz
  - mean 1.70 Hz
  - min 1.53 Hz
- **Model:** a point mass m at distance l (grip point to centre of mass) on a peduncle with bending spring k1:
  - f_e = (1/2π)·√((k1 + m·g·l)/(m·l²))
  - k1 = 323·d − 0.16, where d is peduncle diameter, fitted on 10 peduncles; units are not stated clearly.
  - Measured vs modelled: f_l = 0.0701·f_e + 1.3204.
  - Estimation error: 3.15% on average, 8.69% at most. Using the fixed mean of 1.70 Hz instead gives 4.96% average and 11.9% maximum.
- **Simulation:**
  - The model is 3 links, with modes at 1.70, 2.84 and 5.00 Hz and a damping ratio of 0.06.
  - Input shaping with 3 acceleration pulses moves 0.5 m in about 1.8 s, and swing settles "within 2.0˚" during constant-speed motion.
  - "The effect of damping control may be lost at about 25% of the error, and vibration may be conversely amplified by more than 25%" (error in the assumed natural frequency).
- **Experiment:**
  - A 1-DOF slide carried real clusters 0.5 m in about 2.5 s.
  - Table 4, headed "Reduction ratio [%]", lists 46.7, 10.2, 67.8, 80.5 and 74.8 for 5 clusters. The text defines the figure as "the ratio of the peak magnitude of the input shaping method for the peak magnitude of the experimental result of a trapezoid wave input". These could be either the reduction or the remaining ratio; the paper is ambiguous. The text does say that "in each case… the effect of damping vibration was greater using the input shaping method than the trapezoid wave."
  - The authors attribute the spread to "individual specificity of the physical properties… the slight vibration at the beginning of the transportation, and the variation at the an initial phase angle."
- **Note:** The model rotates only in the plane of travel. Rotation about the stalk axis is not modelled.

Source: [Kondo et al. 2009](https://www.jstage.jst.go.jp/article/eaef/2/3/2_3_108/_article/-char/en)

**Cradled or caught from below**
- **AHPPEBot:** "the peduncle is severed and falls into the collection mesh" [HW-field] — [arXiv 2405.06959](https://arxiv.org/abs/2405.06959)
- **Kiwifruit (Fu 2024):** fruit fall onto polyethylene-foam-covered curved rods that (paraphrase) buffer and guide them into a collection end and a retractable hose (0–0.9 m) to a box [HW-lab] — [PMC10875067](https://pmc.ncbi.nlm.nih.gov/articles/PMC10875067)

**Second contact or pusher against swing:** the grape end effector's pushing device exists "to reduce the swinging of the bunch at the time of carrying, and to orient the bunch at the time of release." The results say it "also worked effectively except for a rachis that was … short." The scanned text is garbled there, so the exact length is unreadable. — [Kondo 1995](https://www.jircas.go.jp/sites/default/files/publication/jarq/29-3-171-177_0.pdf)

**Where to grip to stop tilting (post-harvest pick-and-place, de Haan 2021)** [HW-lab, plastic fruit]
- Grasp rules:
  - "Space condition: the target grasp location lies at least at a distance L from the junctions," where L is larger than the end-effector width.
  - "Balance condition… introduced to prevent undesired tilting of the truss. The selected grasping point is the candidate location which lies closest to the center of mass."
- Even so, "Undesired tilting of the second model was frequently observed."

Source: [de Haan et al.](https://arxiv.org/abs/2103.01272)

**Hook vs pinch (Kok 2024)** [HW-lab]
- **Pinch prototypes:** rigid pinch grippers "did not meet the design criteria for a robust grip… as tomato trusses tended to swing and wobble during pick-and-place actions. Maintaining a grasp on higher-weight trusses often led to slipping out of the end-effector or dirtying of the gripper by the tomatoes… reducing the effectiveness of the grip."
- **Fin-ray / deformable pinch tips:** "A slight tilting or motion of the truss and the tips might miss align, causing the truss to fall out."
- **Hook:** "ensures a stable grip even during the robot arm's movement, facilitating faster operational speeds." It also allows "accurately positioning trusses at a desired location, unlike the pinch grip, which requires precise pre-positioning of the tomato to avoid accidental dropping upon release."

Source: [Kok 2024](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)

**Denso release** [PATENT]: the stalk is released "at the position immediately above the collection box," or the cluster drops onto a conveyor if there is one. The stated purpose of the bite-plus-gap grip is that the cluster does not fall after the cut. — [NL2026342B9](https://patents.google.com/patent/NL2026342B9/en)

**Post-harvest crate state:** trusses are "stacked on top of each other with the peduncle facing upwards, which is how they are commonly stored in a crate after harvesting" — [van den Bent et al.](https://arxiv.org/html/2309.17170v2). Kok describes the industrial setting the same way: "The tomatoes are harvested and stacked in a crate, with an optional sheet of paper between each layer" — [Kok 2024](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)

### Inferences
- [INFERENCE] **Why our truss turns about the jaw-closing axis.** Two flat pads pinching a round 3–5 mm stalk give a narrow line or patch contact. Rotation about the closing axis (the pad normal) is resisted only by torsional friction in that patch.
  - Rough estimate, assuming:
    - µ ≈ 0.5;
    - 4 N per pad, with both pads acting;
    - an effective contact radius of a few mm along the stalk.
  - Holding torque ≈ 2·µ·N·r_eff ≈ 2 × 0.5 × 4 N × 0.003 m ≈ 0.012 N·m.
  - Gravity torque of an 85 g truss whose centre of mass is 50–80 mm from the grip and off the vertical: up to 0.085 × 9.81 × 0.06 ≈ 0.05 N·m.
  - So friction alone cannot hold the truss at an arbitrary angle about the closing axis. It will rotate until the centre of mass is below the grip, which is what Ji et al. observed ("hangs almost vertically downwards").
  - The same logic applies to spin about the stalk axis, where the lever arm is only the stalk radius, about 2 mm.
  - All numbers here are my assumptions, not measurements.
- [INFERENCE] Given that, the documented fixes fall into four families:
  1. Accept the hang. Orient the jaw so the truss already hangs centre-of-mass-down at the moment of the cut, then carry it with slow or input-shaped accelerations. Kondo's measured 1.5–1.9 Hz is for 300–400 g clusters. A simple pendulum with l = 0.06–0.08 m gives about 1.8–2.0 Hz for our size; this is an estimate, not a measurement.
  2. Add geometric (form) closure on the stalk: Denso's biting protrusions, de Haan's rubber knobs that enclose the peduncle, Kok's hook. On a jaw, the equivalent is a V-groove or ridges running across the stalk.
  3. Add a second contact away from the grip: Kondo's grape pusher. On a 1-DOF jaw this would be a passive, spring-loaded rest or fork that touches the peduncle or the first fruit below the pads.
  4. Do not hold it at all: AHPPEBot's mesh pocket, or a basket under the jaw.
- [INFERENCE] Lengthening the pad contact along the stalk raises torsional resistance about the closing axis roughly in proportion to contact length. If the 36 mm pad dimension currently runs across the stalk rather than along it, turning the pad 90° is a cheap test.
- [INFERENCE] Juice or dirt on pads (Kok) and crushed, juicy stalks (Denso) both cut friction. A pad that relies on friction alone will get worse over a shift.

### Gaps
- No source reports measured rotation of a truss within a stalk grip (degrees), or rotation about the stalk axis, as a failure mode.
- No tomato source reports a pusher, fork or second contact on a truss tool. Only the 1995 grape tool has one.
- Kondo 2009 considers only planar swing in the direction of travel, not twist about the stalk.
- I found no description of how Artemy or Ridder physically place trusses in crates: orientation, drop height, or whether the stalk is held until contact.

---

## 3. Reported real-hardware results: success rate, cycle time, drop / shatter / damage, and failure modes (especially slip, rotate, swing, hit during transport and placing)

### Takeaway
Real-hardware tomato-cluster harvest success is about 50–90%, and the failures are mostly in approach, perception and cutting. The only primary-source transport loss is Kondo 2010: one cluster in 20 lost a fruit, and the authors call for manipulator control that avoids impacts on the cluster. Post-harvest studies add slip-out and tilt of pinch-gripped trusses.

### Cited Findings

#### Summary of success rates and cycle times

**Tomato:**

| System | Success | Time |
|---|---|---|
| Kondo 2010 | 50% | ~15 s per cluster |
| AHPPEBot | 86.67% | 32.46 s per successful truss |
| Yuan/Ji 2013 | 86.67% | 37.57 s per truss |
| Li & Sun 2025 | 100% clamp, 86.7% cut (indoor) | — |

**Grape:** Jiang 2022 harvested 13 of 15 clusters (86.7%; the abstract says 83%) at 8.76 s per cluster.


**Kondo et al. 2010** [HW-field, 20 clusters, high-density system] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_pdf)
- About 15 s per cluster.
- Success 50% (10 of 20).
- Failures:
  - 35% (7 of 20): the tool could not reach into the plant; its top hit the upper cluster.
  - 10%: insufficient air pressure to grasp peduncles repeatedly.
  - 5% (1 of 20): "caused by the loss of a fruit from the harvested cluster. It is important to control the manipulator to prevent impacts on the tomato cluster."

**AHPPEBot** [HW-field, commercial greenhouse] — [arXiv 2405.06959](https://arxiv.org/abs/2405.06959)
- **With manual intervention (48 trusses, Table III):**
  - stem-point prediction: 85.41% (41/48);
  - bottom-up wrapping: 93.75% (44/48);
  - detach: 95.56% (43/45);
  - overall harvesting: 89.58% (43/48).
- **Autonomous continuous test (15 trusses, one attempt each, unpruned):**
  - 13 successes in 422 s, an average of 32.46 s per success;
  - stem detachment 92.86%;
  - overall 86.67%.
- **Failure causes:**
  - "Interference from the actuator leading to target displacement";
  - pose-estimation errors from occlusion;
  - end-effector limits. In one case, "the end effector rotated, with its netted bottom touching the base of the cluster, displacing it. Consequently, the stem did not fall into the guiding slot."
- Left-facing and inward-facing trusses were excluded as outliers.

**Yuan et al. 2013** [HW-field, abstract] — [sinoxiv](https://sinoxiv.napstic.cn/article/5595751)
- Success 86.67%, an average of 37.57 s per truss.

**Li & Sun 2025** [HW-lab, abstract] — [napstic](https://search.napstic.cn/literature/periodical/010njhyj202503017)
- Whole-bunch clamping 100%, shearing 86.7%.

**Feng et al. 2018** [SNIPPET] — [IJABE](https://ijabe.org/index.php/ijabe/article/view/2853)
- 1.4 attempts per successful harvest; 8 s per successful cycle, excluding travel.

**Jiang et al. 2022, grape** — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.881904/full)
- [HW-field] Trellis test: 13 of 15 harvested (86.7%), 8.76 s per cluster, two clusters with berry damage (4.39% and 2.02% grape damage rate).
- The abstract states 83% success, which conflicts with Table 3.
- [HW-lab] With 0–5%, 6–20% and 21–40% leaf shade: 93.3%, 86.7% and 73.3%.
- Failures were mainly perception: leaf occlusion and small grape size affecting coordinates. In the lab, the plastic grape model shook.
- Bunch slip, swing and drop are not reported as failure modes.

**Fu et al. 2024, kiwifruit** [HW-lab] — [PMC10875067](https://pmc.ncbi.nlm.nih.gov/articles/PMC10875067)
- 25 clusters (125 fruit): 88.0% success, 9.7 s per cluster.
- Damage 7.3%, rising to 12.0% at 6 fruit per cluster.
- Failures:
  - premature cutting when leaves fooled the colour sensor;
  - fruit-to-fruit collision damage on overripe fruit;
  - a fixed bin size that missed fruit.

**de Haan et al. 2021** [HW-lab, plastic fruit, PincherX 150] — [arXiv](https://arxiv.org/abs/2103.01272)
- Pick-and-place, 20 trials per truss.
- Simple models: 8% failure (5 of 60). Realistic models with a real peduncle: 17% (10 of 60).
- Realistic-model failures were all "pedicels blocking the end effector."
- "Undesired tilting of the second model was frequently observed."
- Servo overload and overheating occurred and was excluded from the counts.

**van den Bent et al. 2024** [HW-lab, post-harvest] — [arXiv HTML](https://arxiv.org/html/2309.17170v2)
- Failure rates by grasp-pose method (isolated trusses unless noted):

  | Method | Failure rate |
  |---|---|
  | random grasp pose | 47.6% |
  | pose nearest the bounding-box centre | 20.0% |
  | proposed ranking | 7.2% (5.2% gripper, 2.0% perception) |
  | proposed ranking, cluttered trusses | 4.4% |

- Gripper failures (paraphrase): the truss slipped out of the fingers during lifting or mid-air. The authors read this as a weakness of pinch grasping.
- Repeated grasping abraded the peduncle skin.

**Kok 2024** [HW-lab, post-harvest] — [TU Delft](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)
- **Hook gripper at crate edges:** 8 of 10. One truss scraped against the crate side during lifting and was counted as a failure. One was too tilted to hook.
- **Stacks:** 6/6, 6/6, 11/11 and 10/11. The one failure: a supermarket truss "lost a tomato, causing instability and slipping out of the gripper."
- **Deformable pinch prototype:** could not lift a 500 g steel test truss (maximum 294.1 g). When the robot moved, "it often resulted in dropping the held weight."

### Inferences
- [INFERENCE] The single best-documented transport failure is fruit shed from impacts or jerks (Kondo 2010, Kok). Combined with the input-shaping data, this supports limiting acceleration and jerk during the carry and keeping clearance from walls. Crate-wall contact is a counted failure in Kok's tests.
- [INFERENCE] The success rates above are for whole harvest cycles, which are dominated by approach and cut failures. No study isolates "truss arrives in the crate without fruit loss" as a metric. The user's problem, rotation during the carry, is under-measured in the literature.

### Gaps
- No real-hardware drop or shatter rate per carried truss (fruit lost per truss in transit) was found for any tomato truss robot. The only data point is 1 of 20 clusters in Kondo 2010.
- No commercial robot publishes success, damage or cycle-time figures. Ridder gives labour-saving percentages only.
- I could not verify the bunch-robot figures that appeared only in search summaries, listed in section 1.

---

## 4. Where do they grip relative to the cut point and the first fruit, and at what gripper orientation relative to gravity?

### Takeaway
At harvest, the clamp-and-cut tomato tools I could check grip the peduncle immediately next to the cut, on the plant side of the first fruit. These are Kondo 2010, the Denso patent, and Ji et al. as cited by de Haan. Kondo has the grasper and cutter inside the same upper finger. The Denso patent clamps the stalk between grip and receiver across the same passage where the cutters sit.

The cluster therefore hangs from a point that is far (tens of mm) from its centre of mass. Kondo's data put the main-stem-to-first-fruit distance at 33–132 mm (mean 79 mm). Tools approach from below or alongside the main stem, with the passage plane horizontal and the cluster passing upward through it.

Gripping near the centre of mass to stop tilt has only been done in post-harvest handling (de Haan, Kok).

### Cited Findings
- **Kondo 2010** [HW-field] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_pdf)
  - The upper fingers surround the main stem; the end effector slides up until the lower-finger strain gauges sense the peduncle; the upper finger comes down and grasps and cuts.
  - The grip is therefore at the peduncle root, beside the main stem.
  - Peduncle length from main stem to first fruit: 33.0–131.6 mm (mean 79.0); peduncle angle 5–139° (mean 69.5).
- **Kondo 2009** [HW-lab] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/2/3/2_3_108/_article/-char/en)
  - The end effector "hold[s] top of cluster."
  - The distance l "between the position where the end effector held the tomato peduncle and the mass center" drives the swing frequency.
- **Denso NL2026342B9** [PATENT] — [Google Patents](https://patents.google.com/patent/NL2026342B9/en)
  - The arm lifts the tool so the whole cluster passes up through the passage from below, with the stalk inside the opening, so the cutter does not strike the main stem.
  - The passage plane is kept horizontal, or perpendicular to the path.
  - In one embodiment the path is offset horizontally by a distance D from the main stem.
- **AHPPEBot** [HW-field] — [arXiv](https://arxiv.org/abs/2405.06959)
  - The blade edge goes to the stem point (SP), "the junction of the truss peduncle and the main stem… the traditional cutting point."
  - The first-fruit point (FP) and the peduncle beyond it are "to be avoided contact or cutting."
  - The tool approaches from below and envelops the fruit.
- **de Haan 2021** [HW-lab, post-harvest] — [arXiv](https://arxiv.org/abs/2103.01272)
  - The grasp lies on the peduncle at least L (more than the gripper width) from any junction, and as close as possible to the centre of mass.
  - The gripper is aligned with the peduncle, approaching vertically from above onto a truss lying on a table.
- **van den Bent 2024** [HW-lab, post-harvest] — [arXiv HTML](https://arxiv.org/html/2309.17170v2)
  - The target is the peduncle, with as much clearance as possible from other stem parts and tomatoes (paraphrase); pedicels and calyxes are weaker.
  - Top-down 4D grasp pose: 3D position plus wrist roll.
- **Kok 2024** [HW-lab, post-harvest] — [TU Delft](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)
  - The hook goes under the peduncle, "often optimally at the centre of mass."
  - The hook tilts in the pre-grasp phase so it fits under the peduncle.
- **Dutch vocational module (2001), "Oogsten onder glas":** "Bij het oogsten van trostomaten knip je vlak bij de stam. Als je een stompje laat zitten, kan dit doorrotten van de stengel veroorzaken." In English: when harvesting truss tomatoes, cut close to the main stem; a stub left behind can make the stem rot. — [OC-27062 Oogsten](https://maken.wikiwijs.nl/bestanden/76496/OC-27062-2%5B1%5D%20Oogsten.pdf)

### Inferences
- [INFERENCE] There is a geometric conflict. Agronomy wants the cut flush with the main stem, which puts the grip at the plant end of the peduncle. Stability wants the grip near the centre of mass, which is among the fruit.
  - Harvesting tools resolve it by accepting a hanging truss, or by catching it from below (AHPPEBot).
  - Post-harvest tools grip at the centre of mass because they do not need to cut.
  - For a 1-DOF jaw, a second stalk contact further along the peduncle (towards the first fruit) is a way to get both. A long pad along the stalk or a second finger segment would do it. No source has tested this on a truss.
- [INFERENCE] With the grip at the peduncle root and the truss hanging, the truss's natural rest orientation is centre-of-mass straight down. If the jaw axis at the cut leaves the truss held sideways or upward, it will swing toward vertical during the carry. Rotating the wrist so the truss is already "hanging" when the jaw closes avoids storing that potential energy.

### Gaps
- No paper states a grip-to-cut distance in mm on the peduncle, or a grip-to-first-fruit distance.
- No paper specifies the gripper's angle relative to gravity during transport for tomato trusses.

---

## 5. What clamp force is used or recommended on a tomato peduncle, and is there a force above which it is crushed?

### Takeaway
Hardware values in use:

| System | Clamp force | Notes |
|---|---|---|
| Kondo 2010 | 20 N on an 8 mm peduncle | spring 2.33 N/mm |
| Kondo 2010 | 40 N judged necessary for quick motion | 400 g clusters; 5 N friction suffices to merely collect them |
| Kondo 2009 | "34-Nm [sic]" | |
| van den Bent, per Kok | 40 N | |
| Kondo 1995, grape rachis | 10 N | |

No source gives a numeric crush threshold for a tomato peduncle. Kok's lever pinch tests found that every load tried caused bruising or surface damage, worse at higher loads and on thinner stalks.

The Denso patent avoids a force rule altogether. It uses a geometric hard stop (gap slightly below the mean stalk diameter) plus biting ridges, because crushing releases juice that makes the stalk slip.

### Cited Findings
- **Kondo 2010** [HW-field; peduncles 4.8–11.8 mm, clusters up to 400 g] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/3/1/3_1_20/_pdf)
  - 20 N grasp at 8 mm (spring 2.33 N/mm);
  - "frictional resistance of at least 5N was necessary to collect the fruit cluster" (400 g);
  - "40 N was necessary as peduncle grasping force of fingers for robot's quick movement";
  - peduncle cutting resistance about 60 N for the largest peduncles.
- **Kondo 2009:** "34-Nm [sic] grip force," holding the top of the cluster [HW-lab] — [J-STAGE](https://www.jstage.jst.go.jp/article/eaef/2/3/2_3_108/_article/-char/en)
- **Kok 2024, pinch test** [HW-lab] — [TU Delft](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)
  - **Method:** "This setup involves a lever… and a set of blocks that load the peduncle in a manner similar to a pinch grasp. The applied loads range from 200 grams to 6 kilograms, with a 30-second timer." The lever ratio is not stated in what I read, so these masses cannot be converted to newtons at the peduncle.
  - **Result:** "any applied weight resulted in bruising or even surface damage… The extent of damage was reduced for lower loads and in areas where the peduncle exhibited greater thickness."
  - **Padding:** "While incorporating padding on the gripper's surface and adopting rounded edges may alleviate certain effects, pinch gripping still imposes a greater force on the peduncle… as it aims to counteract gravity through friction."
  - **Rigid pinch prototypes:** with "grasping forces ranging from 500 g to 4 kg and peduncle diameters from 3.3 to 6.1 mm," trusses "did not meet supermarket standards due to the incurred damage on the peduncles… Immediate post-testing observations indicated surface bruising or perforation, with the peduncle's condition worsening significantly the next day."
  - **van den Bent:** "Since the work of van den Bent et al. (2023) applies a force of 40 N" (Kok's statement).
- **Denso NL2026342B9** [PATENT] — [Google Patents](https://patents.google.com/patent/NL2026342B9/en)
  - The gap L1 is "greater than 0 and slightly less than an average value of outer diameters of fruit stems."
  - Crushing makes juice leak, so "the fruit stem becomes slippery," it slips, and the cluster may fall.
  - The protrusions "cut into" the stalk without severing it.
- **van den Bent 2024:** repeated grasps caused abrasion on the peduncle skin; no visible damage to tomatoes [HW-lab] — [arXiv HTML](https://arxiv.org/html/2309.17170v2)
- **Kondo 1995, grape rachis:** grasping force 10 N, cutting force 100 N, sized from friction and cross-section tests. The rachis had to be gripped "tightly" without shattering the bunch or rubbing off the bloom. — [JARQ](https://www.jircas.go.jp/sites/default/files/publication/jarq/29-3-171-177_0.pdf)
- **Pedicels, not peduncles:** the fruit pedicel (the small stalk to a single fruit) shear force depends on position, and proximal pedicels need more force than distal ones — [Weng et al. 2024, Agronomy 14:2274](https://doi.org/10.3390/agronomy14102274). This is about single-fruit pedicels and is given for context only.

### Inferences
- [INFERENCE] Scaling Kondo's numbers by load: 40 N for 400 g clusters is about 10 N per 100 g. Our 4 N for 70–100 g is about 40% of that ratio. Kondo's 5 N static / 40 N dynamic split suggests he needed a large margin, about 8×, for quick moves with a friction-only pinch. A friction-only 4 N pinch is therefore low for dynamic carrying. But raising force is exactly what Kok found damages peduncles. That points to geometry (ridges, a groove, a hook or a second contact) rather than more force.
- [INFERENCE] A hard stop at a gap a little below the stalk diameter (Denso) is easy to print on a PLA jaw. With 3–5 mm stalks, a gap of about 2.5–3 mm plus shallow ridges would bite without fully crushing. These numbers are my extrapolation; the patent gives none.

### Gaps
- No numeric crush or bruise threshold (N or N/mm²) for a tomato peduncle was found. Kok's data are qualitative and in lever-load units.
- No cherry-tomato peduncle (3–5 mm) clamp force from any hardware study was found. The Kondo values are for 4.8–11.8 mm peduncles.
- No friction coefficients for peduncle against rubber, silicone or foam pads were found in the sources read.

---

## 6. How do human pickers handle a cut truss? Grower or packing guidance on handling trusses to avoid fruit drop

### Takeaway
Sources on how a human holds a cut truss are thin:
- Dutch vocational material says trusses are cut flush to the main stem with curved shears, and crates ride on harvest trolleys on the pipe rail.
- Post-harvest robotics papers describe crates of trusses stacked peduncle-up, sometimes with paper between layers.
- Extension guidance (search-summary level) stresses gentle handling and minimal drop height.

I found no written instruction on whether pickers let the truss hang by the stalk or lay it in, or on preventing fruit drop from the truss.

### Cited Findings
- **Dutch "Oogsten onder glas" module (2001)**, vocational horticulture course material — [OC-27062 Oogsten](https://maken.wikiwijs.nl/bestanden/76496/OC-27062-2%5B1%5D%20Oogsten.pdf)
  - Trusses are cut, not picked: "Bij trostomaten gebruik je een schaar waarvan het knipgedeelte is gebogen, zodat je gemakkelijk kunt knippen." (For truss tomatoes you use shears with a curved cutting part, so cutting is easy.)
  - Cut "vlak bij de stam" (close to the main stem) to avoid stem rot from stubs.
  - On the harvest rack, "bij trostomaten de lege kisten of dozen" (empty crates or boxes for truss tomatoes); full crates are stacked on it.
  - Most harvest equipment "loopt over een buisrail" (runs on a pipe rail).
- **Industrial crate state:** "The tomatoes are harvested and stacked in a crate, with an optional sheet of paper between each layer of tomatoes" — [Kok 2024](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558). "Stacked on top of each other with the peduncle facing upwards, which is how they are commonly stored in a crate after harvesting" — [van den Bent et al.](https://arxiv.org/html/2309.17170v2)
- **Damage from handling:** "grasping an entire truss by a single fruit would damage the tomato or detach it from the truss" — [de Haan et al.](https://arxiv.org/abs/2103.01272). The PicknPack project's pinch grasp on fruit flesh in the crate "proved unsuitable, as it causes damage to the tomato skin" (as cited by de Haan).
- **Kok:** a truss that "lost a tomato" became unstable and slipped out of the gripper — [Kok 2024](https://repository.tudelft.nl/record/uuid:d81da13c-650f-4b86-be9c-d579cdba7558)
- **Human picking of single cluster-tomato fruit** (not trusses) — [Zheng et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465300)
  - Five growers picked 200 fruit: semi-enclosed grip 85%, fingertip pinch 10%, full coverage 5%.
  - Pickers rotate the fruit (77.5% needed 360–540°) and then pull 3–5 cm.
  - This concerns detaching single fruit, not carrying a cut truss.
- **Extension guidance** [SNIPPET, paraphrased: from a search summary over University of Florida IFAS, University of Arizona CEAC and UC Davis postharvest pages; the exact page for each line was not verified]
  - for cluster types, remove the entire cluster as a unit when about 80% of the fruit show colour;
  - choose varieties whose fruit stay attached to the cluster;
  - use truss supports or cluster clips to stop larger tomato-on-the-vine clusters from crimping;
  - keep drop heights to a minimum to reduce bruising;
  - train and supervise the harvest crew to ensure gentle handling.
  
  Candidate sources: [UF IFAS VH079](https://edis.ifas.ufl.edu/publication/VH079/pdf), [UA CEAC Ch8](https://ceac.arizona.edu/sites/default/files/Ch8.pdf), [UC Davis postharvest](https://postharvest.ucdavis.edu/node/8181).
- **Packing rules:** how products must be laid in a box or crate is set by the auction's packing regulations ("verpakkingsvoorschriften… Hoe je de producten in de doos of kist moet leggen, staat ook in deze voorschriften") — [OC-27062 Oogsten](https://maken.wikiwijs.nl/bestanden/76496/OC-27062-2%5B1%5D%20Oogsten.pdf). The regulations themselves were not found.

### Inferences
- [INFERENCE] Several things together suggest the target end state for a cut truss is "laid flat, peduncle up, in a single layer":
  - the crate convention (peduncle up, stacked, paper between layers);
  - the commercial robots' release directly above the crate (Denso);
  - the extension advice to keep drop height low.
  
  For our carry, this argues for lowering the truss into the crate while still held and releasing low, rather than dropping it from height.
- [INFERENCE] Human pickers cutting with shears in one hand very likely support the truss with the other hand. No source I found documents this, so it should be checked by watching pickers or asking a grower.

### Gaps
- No written grower or packer instruction was found on how a picker holds a cut truss (by the stalk, hanging, or cradled) or lays it in the crate.
- No data were found on fruit drop-off rates from trusses during harvest or handling, or on what causes it (for example impact versus jerk).
- I could not retrieve the Dutch auction packing regulations (verpakkingsvoorschriften) for truss tomatoes.
- No video-based or ergonomic study of manual truss harvesting was found.
