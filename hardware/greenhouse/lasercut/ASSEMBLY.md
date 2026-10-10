# SO-101 greenhouse: laser-cut version

Replaces the 3D-printed frame (2026-10-10). It has the same layout:
- 3 vine rows at 400 mm pitch, 740 mm between the end frames
- crop rails at 520 mm
- the SO-101 mid-aisle, with nothing above the floor inside its reach

It's built from **6 mm birch plywood** with slot-together joints and wedges. The camera mast stays 3D-printed: the `gh_cam_*` STLs in `hardware/greenhouse/` (see its README).

## Which folder to send

Ask the laser shop for their **bed size**, then send the matching folder. Each one is complete: DXF + SVG per sheet, `PARTS.csv`, `README_FOR_LASER_SHOP.md` and `sheets_preview.png`.

| Folder | Sheets | Pieces | Glue joints |
|---|---|---|---|
| `cut_order_1300x900/` | 2 | 30 | Only rail web to flange |
| `cut_order_1000x600/` | 4 | 32 | + each end frame is 2 pieces (dovetails) |
| `cut_order_600x400/` | 10 | 53 | + end frames in 6 pieces, rails spliced (works on any shop's bed) |

Bigger bed = fewer glue joints = stiffer frame.

**Ask for at least 1000 × 600.** On a 600 × 400 bed, the posts are glued end to end with dovetails halfway up. That joint is the weakest point of the whole frame. If you ever reinforce it with a glued splice plate, put it on the **outside** face only: the inside face has just 7.2 mm of clearance to the arm's reach.

For any other bed or sheet thickness, regenerate (bottom of this note).

## Size

| Dimension | Value |
|---|---|
| Footprint | about 940 mm across the rows (incl. clamp ears) × 796 mm along them (incl. tenons) |
| Height | 705 mm to the apex |
| End frame | 860 mm wide between the outer post edges |

## What the parts are

| Part | Role |
|---|---|
| **EF** end frames (×2) | Gable panels: posts, base tie with clamp ears, crop-tie band, rafters, king post |
| **CW + CF** crop rails (×3) | T-beam at 520 mm: keyhole flange underneath, web on top |
| **AW + AF** anchor rails (×3) | Upside-down T on the table: web on the table, keyhole flange on top |
| **RG** ridge | Along the apex |
| **WG** wedges (×14) | Lock every rail tenon outside the end frame |

**Keyholes:** both sides of each web, 15 mm off the row line, every 20 mm. They're the same entry hole + 1.8 mm slot as before. Engraved ticks mark each keyhole, with numbers every 100 mm (in cm from the end frame).

## Assembly

**You need:** PVA wood glue, masking tape or small clamps, a mallet (or a block of wood), fine sandpaper, and 4 small clamps for the table.

0. **Fit test first.** Push a strip of the real sheet into the slots on `TEST`.
   - The laser makes every slot about one kerf wider than drawn. So on a true 6 mm sheet, the strip goes into `6.0` with a little play, and the **snug** slot is usually labelled `5.8`. That's expected; the default files are meant to have that small clearance (glue and wedges take it up).
   - **Trust the comb, not the calipers.** If it doesn't go into `6.0` at all, the sheet is thick. Regenerate with `--t` set to the smallest label it fits.
   - For tighter joints, regenerate with `--t` set to the label of the snug slot.
   - Also check your twine against the three keyhole slots (1.5 / 1.8 / 2.2). The rails use 1.8.
1. **Dry-fit everything before any glue.** Lightly sand any char that stops a tab going in.
2. **Splices** (only if the end frames or rails came in pieces):
   - Lay the pieces flat on baking paper, glue the dovetails, and weigh them down flat.
   - For the end frames, check the posts are square to the base tie.
   - Let dry.
3. **T-beams:**
   - Glue the web tabs into the flange slots, web centred on the flange.
   - Crop rails: web **on top** of the flange.
   - Anchor rails: web **under** the flange, standing on the table.
   - Sight along each rail to check it's straight, then let dry.
4. **Stand the end frames up.** Either face can point in: the three rows are identical, and R1 / R2 / R3 only name them. Then:
   - Push the anchor-rail and crop-rail tenons through the end-frame mortises: web tenon in the tall slot, flange tab in the flat slot.
   - Push the ridge tenons through the apex.
5. **Wedges:**
   - Push a wedge narrow end first through the hole in each tenon, outside the end frame.
   - Tap it until the rail shoulder pulls tight against the frame. Snug, not hammered, or the tenon splits.
   - Re-tap if the frame loosens after a few days.
6. **Fix to the table:**
   - Each base tie has 4 holes of 4.5 mm: 2 in the ears, 2 at y = ±200. That's 8 holes in all.
   - Bolt a small **steel angle bracket** (corner brace, ~30 mm legs) to each, on the **outside** face, with M4 × 16 + washer + nut. Then screw the bracket's other leg to the table.
   - If your table edge is close enough, clamping the ears works too.
   - Tape-mark the frame's footprint and the arm base. The frame must not move when the gripper closes.
7. **Arm and camera:** both stay where they were in the printed design.
   - **Arm:** SO-101 base mid-aisle, 370 mm from each end frame's inner face, centred on y = ±200 (halfway between two rows).
   - **Camera mast:** base 120 mm outside the end frame at y = 151, as in `../README.md`.

## Hanging vines

These work exactly as before:
1. Take knotted twine. Drop the top knot into a crop-flange keyhole; it sits on top of the flange beside the web. Slide it into the slot.
2. Do the same in the anchor-flange keyhole directly below; the knot sits underneath the flange, beside the web. Pull the twine taut.
3. Keep neighbouring vines at least 60 mm apart. Clip fruit between 100 and 350 mm above the table.

## Stiffness: not tested yet

- **Across the rows** the end-frame panels are very stiff.
- **Along the rows**, nothing braces the bays. Diagonal braces would cross the arm's reach. Only the wedged rail joints, the 6 mm panels and the table brackets resist racking.
- That direction is probably weaker than the printed 20 mm tubes, and it's untested.
- **After the dry fit:** push the ridge sideways along the rows. If it sways more than a few mm, brackets at all 8 holes fix most of it. Glue the tenon shoulders as a last resort, but then the frame no longer comes apart.

## Checked in the generator (every build)

- **Reach:** every part above 40 mm clears the SO-101's full-reach sphere by **7.2 mm** or more, the same margin as the printed design. The only parts inside it are floor-level (base ties, anchor rails), as before.
- **Fit:**
  - No two parts overlap, so every tenon passes cleanly through its mortise.
  - The wedges bite into their tenons slightly, by design.
- **Camera:** from the printed mast's default pose (lens 423 mm, tilt 20°), all 96 sampled pick-zone points are visible through the end-frame opening. The margin is 13° horizontal and 15° vertical inside the field of view. The mast base clashes with nothing.
- **Sheets:**
  - Every piece is one closed outline and fits the bed.
  - Pieces sit at least 5 mm apart.
  - Every DXF re-opens cleanly.
  - At least 3 mm of wood remains around every dovetail and keyhole.

## Regenerate

```bash
cd hardware/greenhouse/lasercut      # from the repo root
python greenhouse_lasercut.py check                                  # checks + preview/assembly.png, parts.png, assembly.stl
python greenhouse_lasercut.py build --bed 1000x600 --t 5.8 --out cut_order_custom
```

- **Needs:** `pip install numpy shapely trimesh manifold3d mapbox-earcut ezdxf matplotlib`. A throwaway venv works.
- **Other dimensions:** at the top of `greenhouse_lasercut.py`. The build refuses to write cut files if a check fails.
