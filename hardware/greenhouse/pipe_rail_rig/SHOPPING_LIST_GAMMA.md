# Pipe-rail test rig v2: Gamma shopping list

Prices read from gamma.nl product pages on 2026-11-19. Check them again before you buy: prices and offers change.
The list covers the rig in `pipe_rail_rig.fs` (v2: 0.99 x 0.44 x 0.76 m, knock-down). Only Gamma, cheapest option per part.

| # | Part | Gamma product | Qty | Each | Total | Notes |
|---|---|---|---|---|---|---|
| 1 | Frame, sleepers, camera masts | [Bouwhout vuren geschaafd 44x44 mm 270 cm](https://www.gamma.nl/assortiment/bouwhout-vuren-geschaafd-44x44-mm-270-cm/p/B736850) | 4 | €8.89 | €35.56 | €4.79 each with the free GAMMA Voordeelpas until 30-11-2026 (= €19.16) |
| 2 | Cart deck (2 strips side by side = 238 mm wide) | [Bouwhout vuren geschaafd 18x119 mm 240 cm](https://www.gamma.nl/assortiment/bouwhout-vuren-geschaafd-18x119-mm-240-cm/p/B270673) | 1 | €12.99 | €12.99 | €9.99 with Voordeelpas |
| 3 | Pipe rail tubes | [Martens elektrabuis slagvast grijs 19 mm 2 m](https://www.gamma.nl/assortiment/martens-elektrabuis-slagvast-grijs-3-4-19-mm-2-meter/p/B461108) | 2 | €1.99 | €3.98 | ⚠️ store only (no delivery, no click & collect in every store) |
| 4 | Stems | [Ronde stok beuken Ø 8 mm 100 cm](https://www.gamma.nl/assortiment/ronde-stok-beuken-8-mm-100-cm/p/B203982) | 8 | €1.69 | €13.52 | ⚠️ online only in packs of 50: buy singles in the store |
| 5 | Corner brackets (tie joints) | [Waelbers versterkingshoek 50x50x40 mm verzinkt 4 st](https://www.gamma.nl/assortiment/waelbers-versterkingshoek-50-x-50-x-40-mm-verzinkt-4-stuks/p/B105042) | 2 | €4.29 | €8.58 | 8 brackets |
| 6 | Bolts (take-apart joints) | [GAMMA slotbout M6 x 60 mm verzinkt 8 st](https://www.gamma.nl/assortiment/gamma-slotbout-m6-x-60-mm-verzinkt-8-stuks/p/B458208) | 1 | €5.99 | €5.99 | online "almost sold out" |
| 7 | Wing nuts | [GAMMA vleugelmoer M6 verzinkt 4 st](https://www.gamma.nl/assortiment/gamma-vleugelmoer-m6-verzinkt-4-stuks/p/B458307) | 2 | €2.69 | €5.38 | |
| 8 | Washers | [GAMMA sluitring M6 verzinkt 50 st](https://www.gamma.nl/assortiment/gamma-sluitring-m6-verzinkt-50-stuks/p/B458373) | 1 | €2.89 | €2.89 | stops the wing nuts biting into the wood |
| 9 | Long screws (row frames: rail into post) | [Rotadrill schroef platkop T25 verzinkt 5x80 mm 16 st](https://www.gamma.nl/assortiment/rotadrill-schroef-universeel-platkop-t25-verzinkt-5x80-mm-16-stuks/p/B365998) | 1 | €8.39 | €8.39 | 2 per joint x 8 joints |
| 10 | Short screws (brackets, deck, sleepers) | [GAMMA schroef cilinderkop verzinkt 4.0x30 mm 30 st](https://www.gamma.nl/assortiment/gamma-schroef-universeel-cilinderkop-verzinkt-4-0x30-mm-30-stuks/p/B465540) | 1 | €5.69 | €5.69 | |
| | **Rig without wheels** | | | | **€102.97** | ≈ €83.57 with Voordeelpas |

## Wheels (pick one)

| Option | Gamma product | Cost | Notes |
|---|---|---|---|
| **A, cheapest:** 3D-print 4 U-groove wheels + brackets, M6 bolt axles | +1 pack slotbout M6x60 (B458208, €5.99) + [GAMMA borgmoer M6 8 st](https://www.gamma.nl/assortiment/x/p/B458291) (€2.99, search-result price, not checked on its page) | **€8.98** | groove sized for the 19 mm tube; lock nuts so the wheels don't work loose |
| B, buy ready-made | [StoreMax Basic schuifdeur wielset R-20 aluminium 2 st](https://www.gamma.nl/assortiment/storemax-basic-schuifdeur-wielset-r-20-aluminium-2-stuks/p/B391699) x 2 | €37.18 | ⚠️ made for top-hung sliding doors on a 20 mm round rail (20 kg per wheel); fit on 19 mm conduit and mounting under a cart are unverified |

**Total with option A: €111.95** (≈ €92.55 with Voordeelpas before 30-11-2026).

## Cut list (same codes as the Onshape model and its CUT PLAN)

Measure every mark from the LEFT end of the stock and cut on the right of the line (3 mm saw kerf assumed).

| Stock | Pieces in cutting order (code, length mm) | Marks from the left end (mm) | Offcut |
|---|---|---|---|
| [1] batten 1 | B1-1 string rail A 988, B1-2 floor rail A 988, B1-3 post A closed end 610 | 988, 1979, 2592 | 105 |
| [1] batten 2 | B2-1 string rail B 988, B2-2 floor rail B 988, B2-3 post B closed end 610 | 988, 1979, 2592 | 105 |
| [1] batten 3 | B3-1 post A open end 610, B3-2 post B open end 610, B3-3 top tie 444, B3-4 top tie 444, B3-5 floor tie 356 | 610, 1223, 1670, 2117, 2476 | 221 |
| [1] batten 4 | B4-1..B4-4 sleepers 240, B4-5 + B4-6 mast poles 410, B4-7..B4-10 mast feet 180 | 240, 483, 726, 969, 1382, 1795, 1978, 2161, 2344, 2527 | 170 |
| [2] deck plank | D-1 + D-2 deck strips 730, D-3 + D-4 cleats 238 | 730, 1463, 1704, 1945 | 452 |
| [3] conduit x 2 | T1 / T2 rail tube 1180 | 1180 | 817 each |
| [4] dowel x 8 | S1..S8 stem 610 | 610 | 387 each |

Hardware actually used: 6 of the 8 [5] angle brackets (4 top joints, 2 floor joints), 6 [6] bolts at the joints + 4 as
wheel axles (= the 2 packs), 6 of the 8 [7] wing nuts, 6 [8] washers, 16 [9] 5x80 screws (2 per row-frame joint),
[10] 4x30 screws for the brackets, wheel brackets, deck strips to cleats and tubes to sleepers.

## Cheaper still

- **Twine instead of the dowels:** pull tomato twine taut between the floor rail and the string rail instead of using 8 dowels. That saves about €10. The printed truss clips must then grip the twine.
- **Skip the washers** (−€2.89). Wing nuts on bare wood will dent it over time.

## The Onshape model

- Done in v3 (2026-11-19): the model now uses exactly these parts: 19 mm tubes, two 119 mm deck strips on two cleats, wooden camera masts, 50x50x40 brackets, M6 bolts, printed wheels.
