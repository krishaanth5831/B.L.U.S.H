# SO-101 test greenhouse — fully printed

A tabletop greenhouse for the SO-101 hardware track: three rows of hanging cherry
tomato vines, aisles the arm stands in, and nothing overhead or at the ends that the
arm can hit at full stretch. Every piece is printed; a 256 × 256 mm bed fits all of them.

Onshape (private, parametric): **SO-101 Greenhouse** —
<https://cad.onshape.com/documents/54620f37090a501ebf4a1a7f>. Double-click the
*SO-101 Greenhouse* feature to change any dimension or fit; the source is
`so101_greenhouse.fs` here.

## Size and layout

| | |
|---|---|
| Footprint | 760 × 820 mm (along rows × across) |
| Height | 692 mm to the ridge |
| Rows | 3, at 400 mm pitch (y = −400 / 0 / +400) |
| Aisles | 2, at y = ±200, so every vine the arm works is 200 mm away |
| Crop rails (vine tops) | 520 mm, above the arm's ~490 mm vertical reach |
| Arm position | middle of an aisle, 370 mm from each end frame |

```
      end frame            end frame
   y=+400  ===rail==============   row 1   (vines hang here)
   y=+200   . . . [SO-101] . . .   aisle A  - arm picks rows 1 and 2
   y=   0  ===rail==============   row 2
   y=-200   . . . . . . . . . .    aisle B  - move the arm here for rows 2 and 3
   y=-400  ===rail==============   row 3
```

**Free articulation was checked in the model**, not assumed. A sphere of the arm's
full reach (400 mm from the shoulder incl. the jaw, shoulder 117 mm up, from the
SO-ARM100 URDF) sits in the aisle. ⚠️ Only table-level pieces (feet, anchor rails,
base ties) fall inside it — they're the floor. Every post, tie, rafter and crop rail
stays at least 7 mm outside. At 720 mm long the four nearest posts grazed the
sphere by 1–2 mm, which is why the house is 740 mm between end frames.

## Parts (print these)

| STL | Qty | Size (mm) | Notes |
|---|---|---|---|
| `gh_fit_coupon.stl` | 1 | 76 × 46 × 26 | **Print first** — see Fit below |
| `gh_pin_x86.stl` | 86 | 40 × 16.5 × 16.5 | Every joint; print 20–30 per plate |
| `gh_rail_240.0_x18.stl` | 18 | 240 × 20 × 20 | Crop + anchor rails, keyholes every 20 mm |
| `gh_tube_240.0_x3.stl` | 3 | 240 × 20 × 20 | Ridge |
| `gh_tube_220.0_x12.stl` | 12 | 220 × 20 × 20 | Posts (2 per post) |
| `gh_tube_156.5_x24.stl` | 24 | 156.5 × 20 × 20 | Base ties, crop ties, rafters (2 each) |
| `gh_tube_101.6_x2.stl` | 2 | 101.6 × 20 × 20 | King post, top |
| `gh_node_F_out_x4.stl` | 4 | 80 × 40 × 20 | Outer feet, with a table tab (4.5 mm screw/clamp hole) |
| `gh_node_F_mid_x2.stl` | 2 | 87 × 40 × 20 | Middle feet |
| `gh_node_C_out_x4.stl` | 4 | 91 × 80 × 20 | Eave corners: post, tie, rafter, crop rail |
| `gh_node_C_mid_x2.stl` | 2 | 70 × 60 × 20 | King post crossing at crop height |
| `gh_node_R_x2.stl` | 2 | 72 × 40 × 20 | Ridge |

160 pieces, about **2.5 kg PLA** as printed (3.3 kg if everything were solid), roughly
50–60 h on a Bambu-class printer.

**Print settings:** PLA, 0.4 mm nozzle, 0.2 mm layers, **4 walls** (the tubes are
1.6 mm thick, so this prints them solid), 15 % infill. All parts are already in
print orientation and need **no supports**. Nodes lie flat in their own plane;
the square holes print as short bridges. Every edge is rounded.

## Fit (print the coupon first)

- **Pin clearance**: three sockets, 1 / 2 / 3 dots = bore −0.1 / nominal / +0.1 mm.
  Push a printed pin into each. The right one goes in by hand and doesn't wobble.
  If that's not the 2-dot socket, change *Pin clearance* in Onshape and re-export.
  Default is 0.15 mm per side (pin 16.5 mm in a 16.8 mm bore).
- **Keyhole slot**: three slots at 1.5 / 1.8 / 2.2 mm. Your twine must slide into
  the slot and a knot must **not** pull through it. Set *Keyhole slot width* to match.

## Assembly

1. **End frames (×2, identical).** Feet at the bottom, outer posts up to the eave
   corners, king post up the middle through the crop-height node to the ridge.
   Then the base ties, crop ties and rafters. Every joint is a pin pushed halfway
   into each side. Tubes join end to end with a pin too.
2. **Long rails.** Join the two end frames with 3 anchor rails on the table, 3 crop
   rails at 520 mm and the ridge. Each runs straight through the node's square
   through-hole socket.
3. **Lock bolts (optional).** Every socket and tube end has a 3.4 mm hole at the
   middle of the pin. Use M3 × 25 + nut where a joint feels loose, or glue joints
   you never want to take apart.
4. **Fix it to the table.** Screw or clamp the 4 outer feet through their tabs.
   ⚠️ The vault rig rule: the frame must not move when the gripper closes. A clamped
   frame doesn't.

The rails run in different directions on the two levels. Anchor rails go keyholes
**up**, crop rails keyholes **down**. The tick marks on the rails' sides line up
top and bottom.

## Hanging vines

- Use horticultural twine with a knot (or bead) at each end.
- **Top:** push the knot up through a crop-rail keyhole's round entry and slide it
  into the slot. **Bottom:** do the same in the anchor-rail keyhole directly below,
  with the twine pulled taut. A taut vine doesn't swing away when the jaw closes,
  which single-fruit picking needs.
- Keyholes are every 20 mm. Keep neighbouring vines at least 60 mm apart so the open
  jaw never brushes the next one; the model shows vines at 300 / 360 / 420 mm.
- Clip trusses (tomato truss clips) between **100 and 350 mm** above the table, the
  arm's comfortable band from the aisle.
- Re-hanging a vine at a named keyhole takes seconds, which meets the vault rule of
  re-hanging under 60 s.

## Arm and cameras

- Fix the SO-101 base mid-aisle, 370 mm from either end, centred on y = ±200. Mark
  it on the table, and mark where the feet sit.
- ⚠️ Per the vault recording notes, the front (birdseye) camera mounts to the
  **table**, not to this frame — the frame gets bumped. The printed mast below
  does exactly that.
- The structure works for cutting trusses (the vault plan) as well as plucking
  single fruit.

## Birdseye camera mast

A printed pole for the overview camera, an InnoMaker U20CAM-1080P (`birdseye` in
lerobot). It screws to the table outside the end frame, and you can adjust its
height, pan and tilt. It is the second feature in the same Onshape Part Studio,
*Birdseye Camera Mast*; the source is `birdseye_mast.fs` here.

Every adjustment locks in fixed steps. Per the vault notes, a front camera that
shifts between sessions silently spoils the dataset, so nothing relies on friction:

| Adjustment | Locked by | Step | Read it on |
|---|---|---|---|
| Height | M4 bolt through the outer tube and one of the slider's 16 holes | 10 mm | the slider mark level with the tube's top edge (20 to 170) |
| Pan | serrated faces, 36 teeth, clamped by a knob | 10° | the yoke tick above a pointer line on the slider (long tick every 30°; the dot marks the pointer that faces the arm) |
| Tilt | serrated faces, clamped by a knob | 10° | the cradle tick under the line on the upright's top (long ticks at 0, 30, 60 and 90° down) |

**Lens height** (tilt 0) = 273 mm + mark + 160 mm per riser:

- no riser: 293 to 443 mm
- one riser: 453 to 603 mm
- two risers: 613 to 763 mm

Each range continues the one before it.

### Where it goes

These positions are checked in the model:

- **Base:** at x = −120 mm, so 120 mm outside the end frame's centre line. Its
  centre is 49 mm short of the aisle centre (y = 151 for the +Y aisle), because
  the camera sits 49 mm to the side of the pole and so lines up with the aisle.
- **Orientation:** the "ARM ▶" on the base points along the rows, into the house.
- **Default setting:** mark 150, pan 0, tilt 20° down, which puts the lens at
  423 mm. From there the camera sees all of the following through the end-frame
  opening, with at least 10° to spare, and no greenhouse member blocks the line of
  sight:
  - all four corners of the pick zone (x 300 to 420, z 100 to 350) on both vine rows
  - the arm's base

  The check uses an 80° × 60° field, narrower than the lens's 103° horizontal,
  because lerobot records at 640 × 480 and that mode may crop the sensor.
- **Arm clearance:** as set, the mast is at least 49 mm outside the SO-101's reach
  sphere. The head stays at least 10 mm outside at every height, pan and tilt.
- **Higher views:** with one riser, 523 mm is still clear. At 563 mm the crop tie
  hides 2 of the zone's corners, and at 603 mm it hides 4. Above about 550 mm,
  use the risers only for setups without the greenhouse.

### Parts (print these)

| STL | Qty | Size (mm) | Notes |
|---|---|---|---|
| `gh_cam_fit_coupon.stl` | 1 | 183 × 91 × 34 | **Print first**: see Fit below |
| `gh_cam_base_x1.stl` | 1 | 110 × 110 × 40 | 4 countersunk holes for 4 mm wood screws |
| `gh_cam_outer_tube_x1.stl` | 1 | 240 × 34 × 34 | "TOP" end up |
| `gh_cam_slider_x1.stl` | 1 | 240 × 29 × 29 | height holes and marks, serrated top |
| `gh_cam_yoke_x1.stl` | 1 | 47 × 38 × 30 | pan and tilt serrations, pan scale |
| `gh_cam_cradle_x1.stl` | 1 | 45 × 44 × 38 | camera plate, tilt scale |
| `gh_cam_knob_x4.stl` | 4, plus 2 per riser | Ø 21 × 8 | hex pocket takes an M4 nut or hex bolt head |
| `gh_cam_riser_optional.stl` | 0 to 2 | 160 × 34 × 34 | each adds 160 mm |
| `gh_cam_collar_optional.stl` | 1 per riser | 40 × 40 × 60 | joins a riser to the tube above it |

The basic kit is about 0.3 kg of PLA, roughly 7 to 9 h of printing.

- **Settings:** the same as the house: 0.2 mm layers, 4 walls, 15 % infill.
- **Orientation:** the STLs are already in print orientation and need no supports.
  - The tubes and slider lie flat.
  - The collar stands up.
  - The yoke lies on its side, so both serrated faces print vertical.
- **Rounding:** every edge is rounded except the serration teeth, which need crisp
  flanks.

### Hardware

| Joint | Parts |
|---|---|
| Pan | M4 × 25 hex-head bolt pressed head-first into a knob; M4 nut in the slot at the slider's top |
| Tilt | M4 × 20 hex-head bolt pressed into a knob; M4 nut in the slot on the upright's front face |
| Height lock | M4 × 45 bolt and washer; knob with an M4 nut pressed in |
| Base | M4 × 50 bolt and washer; knob with nut |
| Each collar | 2 × M4 × 50 bolts and washers; 2 knobs with nuts |
| Camera | 4 × M2 × 12 screws, 4 M2 nuts in the traps behind the plate |
| Table | 4 × 4 mm countersunk wood screws, about 20 mm long, or clamp the plate's edge |

### Fit (print the coupon first)

- **Slider stubs:** 1 / 2 / 3 dots = 0.15 / 0.25 / 0.35 mm of clearance per side.
  - Push each one through the tube ring. The right one slides by hand and doesn't rattle.
  - The default is the 2-dot stub. If yours is a different one, set *Slider clearance*
    in Onshape and re-export the slider.
- **Socket rings:** 1 / 2 / 3 dots = 0.05 / 0.15 / 0.25 mm of clearance per side.
  - These test the base socket and the collars. They print standing, as those do.
  - Push the tube ring into each. The right one takes it by hand without wobbling.
  - The default is the 2-dot ring. If yours is a different one, set *Socket clearance*
    and re-export the base and the collar.
- **Board template:** lay the camera on it and check that its four holes line up.
  - The 28 mm spacing comes from InnoMaker's photos. The 32 mm board and 2.2 mm
    holes come from their manual.
  - If the holes don't line up, change *Board hole spacing* before you print the cradle.

### Assembly

1. **Base:** screw it to the table where the model has it, with "ARM ▶" along the
   aisle toward the arm. Tape-mark its outline.
2. **Outer tube:** put it in the base socket "TOP" end up. Fit the M4 × 50 bolt
   and a knob through the base and the tube.
3. **Yoke onto the slider:**
   - Slide an M4 nut into the slider's top slot, from the face with the numbers.
   - Seat the yoke on the serrated top.
   - Bolt it on with the M4 × 25 knob.
4. **Camera onto the cradle:**
   - Fix the board with M2 × 12 screws from the front, nuts in the traps behind.
   - Put the cable connector at the plate's open bottom edge. The board sits on
     6.5 mm standoffs, so the back parts and the connector clear the plate.
5. **Cradle onto the yoke:** slide an M4 nut into the upright's front slot, then
   bolt the cradle to the upright with the M4 × 20 knob.
6. **Height:**
   - Drop the slider into the outer tube with its dotted pointer on the same side as
     the base's "ARM ▶". It also fits turned round, but then every pan reading is
     180° off.
   - Push the M4 × 45 bolt through the tube's top hole and the slider hole for
     the mark you want.
   - Fit a knob on the other end.

To change pan or tilt, loosen its knob a turn, move the head a tooth or more, and
retighten. The serrations re-seat every 10°.

**Cable:** the camera's USB lead is 1 m. From a 420 mm lens height, down the mast
and across to a laptop, that will probably be short, so plan on a USB 2.0
extension. Zip-tie the lead to the mast and leave a loop at the head, so height and
tilt changes don't pull on the board's connector.

**Recording:**
- Write the setting down with each session, for example "mast mark 150, pan 0,
  tilt 20". If any of it changes, it is a new camera pose (vault: re-record).
- If the image comes out upside down, add `rotation: 180` to the birdseye camera
  in the lerobot `--robot.cameras` dict.

