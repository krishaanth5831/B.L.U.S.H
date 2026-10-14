# Laser-cut order: tabletop greenhouse frame

Flat parts for a small slot-together greenhouse frame (about 0.9 x 0.75 x 0.75 m assembled).

## Material

- **6 mm laser-grade birch plywood**, 10 sheets of **600 x 400 mm** (one file per sheet).
- All slots are drawn for a sheet **6.0 mm** thick. Cut on the line, so the laser makes every slot about one
  kerf wider; that small clearance is intended. If your sheet measures outside 5.8-6.2 mm, tell us
  before cutting and we send regenerated files.

## Files

| File | What |
|---|---|
| `00_TEST_cut_first.dxf` | **Cut this first** from the same sheet: slots labelled 5.4-6.4 mm and 3 keyhole widths. A strip of the sheet should slide into the `6.0` slot with a little play (because of the kerf, the snug one is usually labelled 5.8). If it doesn't go into `6.0` at all, stop and tell us. |
| `sheet_01.dxf` ... `sheet_10.dxf` | The parts, nested, 52 pieces in total |
| `*.svg` | The same sheets as SVG, if your software prefers it |
| `PARTS.csv` | Every part, quantity, size and which sheet it's on |
| `sheets_preview.png` | Picture of every sheet |

## Layers and settings

| Layer | Colour | Operation |
|---|---|---|
| `CUT` | red | **Cut through** |
| `ENGRAVE` | blue | **Light score / engrave only.** Part labels, row labels, 20 mm tick marks. Optional; skip it if it costs extra. |

- Units **mm**, 1:1. Please **do not scale**.
- **Cut on the line, no kerf compensation.** The joints are designed for the normal laser kerf (~0.15-0.25 mm).
- Every contour is a closed polyline. Parts are at least 5 mm apart, so no shared edges get cut twice.
- Cut inner holes before outer contours (the default in most laser software).
- Small features: 1.8 mm slots in the keyholes (twine slots), 5 mm round keyhole entries, 4.5 mm bracket holes.
- Total cut length about **60 m**.

Questions about the parts: please contact the person who sent this order.
