"""Nest the greenhouse parts onto laser-bed sheets and write DXF + SVG per sheet.

Called by greenhouse_lasercut.py build. Conventions for the shop:
  layer CUT      red (ACI 1)   cut through
  layer ENGRAVE  blue (ACI 5)  score / light engrave only (labels, tick marks)
  units mm, closed LWPOLYLINEs, no DXF TEXT (labels are vector outlines), parts >= GAP apart
"""
import csv
import math
import pathlib

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polylabel

BORDER = 10.0
GAP = 5.0


# ---------------------------------------------------------------- MaxRects bin packing
class MaxRects:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.free = [(0.0, 0.0, W, H)]

    def insert(self, w, h, allow_rot=True):
        best = None
        for fx, fy, fw, fh in self.free:
            for rw, rh, rot in ((w, h, False), (h, w, True)) if allow_rot else ((w, h, False),):
                if rw <= fw + 1e-9 and rh <= fh + 1e-9:
                    score = (min(fw - rw, fh - rh), max(fw - rw, fh - rh))
                    if best is None or score < best[0]:
                        best = (score, fx, fy, rw, rh, rot)
        if best is None:
            return None
        _, x, y, rw, rh, rot = best
        self._split(x, y, rw, rh)
        return x, y, rot

    def _split(self, x, y, w, h):
        new = []
        for fx, fy, fw, fh in self.free:
            if x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy:
                new.append((fx, fy, fw, fh))
                continue
            if x > fx:
                new.append((fx, fy, x - fx, fh))
            if x + w < fx + fw:
                new.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy:
                new.append((fx, fy, fw, y - fy))
            if y + h < fy + fh:
                new.append((fx, y + h, fw, fy + fh - y - h))
        # prune contained rects
        pr = []
        for i, a in enumerate(new):
            if not any(i != j and a[0] >= b[0] - 1e-9 and a[1] >= b[1] - 1e-9 and a[0] + a[2] <= b[0] + b[2] + 1e-9
                       and a[1] + a[3] <= b[1] + b[3] + 1e-9 and (a != b or j < i) for j, b in enumerate(new)):
                pr.append(a)
        self.free = pr


def nest(items, bed):
    """items: list of (key, poly). Returns sheets: list of list of (key, placed_transform_fn)."""
    W, H = bed[0] - 2 * BORDER, bed[1] - 2 * BORDER
    order = sorted(items, key=lambda it: -max(it[1].bounds[2] - it[1].bounds[0], it[1].bounds[3] - it[1].bounds[1]) *
                   min(it[1].bounds[2] - it[1].bounds[0], it[1].bounds[3] - it[1].bounds[1]))
    sheets = []
    for key, poly in order:
        w = poly.bounds[2] - poly.bounds[0] + GAP
        h = poly.bounds[3] - poly.bounds[1] + GAP
        placed = None
        for s in sheets:
            r = s["packer"].insert(w, h)
            if r:
                placed = (s, r)
                break
        if not placed:
            s = {"packer": MaxRects(W, H), "items": []}
            r = s["packer"].insert(w, h)
            if r is None:
                raise RuntimeError(f"{key} ({w:.0f} x {h:.0f}) does not fit the bed {bed}")
            sheets.append(s)
            placed = (s, r)
        s, (x, y, rot) = placed
        s["items"].append((key, x + BORDER + GAP / 2, y + BORDER + GAP / 2, rot))
    return sheets


def place(geom, src_bounds, x, y, rot):
    g = affinity.translate(geom, -src_bounds[0], -src_bounds[1])
    if rot:
        h = src_bounds[3] - src_bounds[1]
        g = affinity.rotate(g, 90, origin=(0, 0))
        g = affinity.translate(g, h, 0)
    return affinity.translate(g, x, y)


def rings(poly):
    ps = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for p in ps:
        yield p.exterior
        yield from p.interiors


def label_lines(text_lines_fn, poly, name):
    """Engrave the part name where it fits inside the material."""
    for h in (7, 5, 3.5):
        pt = polylabel(poly.buffer(-1.0) if not poly.buffer(-1.0).is_empty else poly, tolerance=0.5)
        lines = text_lines_fn(name, 0, 0, h)
        if not lines:
            return []
        from shapely.ops import unary_union
        bb = unary_union(lines).bounds
        dx, dy = pt.x - (bb[0] + bb[2]) / 2, pt.y - (bb[1] + bb[3]) / 2
        moved = [affinity.translate(l, dx, dy) for l in lines]
        tb = box(bb[0] + dx - 1, bb[1] + dy - 1, bb[2] + dx + 1, bb[3] + dy + 1)
        if poly.contains(tb):
            return moved
    return []


# ---------------------------------------------------------------- writers
def write_dxf(path, cut_geoms, eng_lines):
    import ezdxf
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4        # mm
    doc.header["$MEASUREMENT"] = 1
    doc.layers.add("CUT", color=1)
    doc.layers.add("ENGRAVE", color=5)
    msp = doc.modelspace()
    for g in cut_geoms:
        for r in rings(g):
            pts = list(r.coords)[:-1]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "CUT", "color": 1})
    for l in eng_lines:
        pts = list(l.coords)
        closed = len(pts) > 2 and np.allclose(pts[0], pts[-1])
        msp.add_lwpolyline(pts[:-1] if closed else pts, close=closed, dxfattribs={"layer": "ENGRAVE", "color": 5})
    doc.saveas(path)


def write_svg(path, bed, cut_geoms, eng_lines):
    W, H = bed

    def d_of(coords, close):
        pts = [(x, H - y) for x, y in coords]
        s = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in pts)
        return s + (" Z" if close else "")

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">']
    for g in cut_geoms:
        for r in rings(g):
            out.append(f'<path d="{d_of(list(r.coords)[:-1], True)}" fill="none" stroke="#ff0000" stroke-width="0.1"/>')
    for l in eng_lines:
        pts = list(l.coords)
        closed = len(pts) > 2 and np.allclose(pts[0], pts[-1])
        out.append(f'<path d="{d_of(pts[:-1] if closed else pts, closed)}" fill="none" stroke="#0000ff" stroke-width="0.1"/>')
    out.append("</svg>")
    pathlib.Path(path).write_text("\n".join(out))


def render_sheets(sheet_geoms, bed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(sheet_geoms)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4.4 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, (name, cuts, engs) in zip(axs, sheet_geoms):
        ax.plot([0, bed[0], bed[0], 0, 0], [0, 0, bed[1], bed[1], 0], color="#bbbbbb", linewidth=0.8)
        for g in cuts:
            for r in rings(g):
                xs, ys = r.xy
                ax.plot(xs, ys, color="red", linewidth=0.5)
        for l in engs:
            xs, ys = l.xy
            ax.plot(xs, ys, color="blue", linewidth=0.3)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main entry
def export(parts, bed, outdir, T):
    import greenhouse_lasercut as g
    outdir = pathlib.Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"{outdir} is not empty; move it away first (nothing overwritten)")
    outdir.mkdir(parents=True, exist_ok=True)

    test = [p for p in parts if p.name == "TEST"]
    main = [p for p in parts if p.name != "TEST"]
    groups = [("00_TEST_cut_first", test), ("sheet", main)]
    rows, sheet_geoms, sheet_files = [], [], []
    sheet_no = 0
    where = {}
    for prefix, plist in groups:
        items, lookup = [], {}
        for p in plist:
            for k in range(p.qty):
                key = f"{p.name}#{k + 1}"
                items.append((key, p.poly))
                lookup[key] = p
        sheets = nest(items, bed)
        for s in sheets:
            if prefix == "sheet":
                sheet_no += 1
                name = f"sheet_{sheet_no:02d}"
            else:
                name = prefix
            cuts, engs = [], []
            for key, x, y, rot in s["items"]:
                p = lookup[key]
                b = p.poly.bounds
                pp = place(p.poly, b, x, y, rot)
                cuts.append(pp)
                for e in p.engrave:
                    engs.append(place(e, b, x, y, rot))
                engs += label_lines(g.text_lines, pp, p.name)
                where.setdefault(p.name, set()).add(name)
            # sanity: inside the bed, no overlaps
            for i, a in enumerate(cuts):
                ab = a.bounds
                assert ab[0] >= BORDER - 1e-6 and ab[1] >= BORDER - 1e-6 and ab[2] <= bed[0] - BORDER + 1e-6 \
                    and ab[3] <= bed[1] - BORDER + 1e-6, f"{name}: part outside the bed"
                for b2 in cuts[i + 1:]:
                    assert a.distance(b2) >= GAP - 0.05, f"{name}: parts closer than {GAP} mm"
            write_dxf(outdir / f"{name}.dxf", cuts, engs)
            write_svg(outdir / f"{name}.svg", bed, cuts, engs)
            area = sum(c.area for c in cuts)
            sheet_geoms.append((f"{name}  ({len(cuts)} parts, {100 * area / (bed[0] * bed[1]):.0f}% used)", cuts, engs))
            sheet_files.append(name)

    for p in parts:
        b = p.poly.bounds
        rows.append(dict(part=p.name, qty=p.qty, size_mm=f"{b[2]-b[0]:.1f} x {b[3]-b[1]:.1f}",
                         what=p.note, sheets=" ".join(sorted(where.get(p.name, [])))))
    with open(outdir / "PARTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    render_sheets(sheet_geoms, bed, outdir / "sheets_preview.png")

    # re-open every DXF to prove it parses, and count entities
    import ezdxf
    total_cut = 0.0
    for name in sheet_files:
        doc = ezdxf.readfile(outdir / f"{name}.dxf")
        msp = doc.modelspace()
        for e in msp.query("LWPOLYLINE[layer=='CUT']"):
            pts = np.array([v[:2] for v in e.get_points()])
            seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
            total_cut += seg
    print(f"\nwrote {len(sheet_files)} files ({sheet_no} sheets + test) to {outdir}; total cut length {total_cut/1000:.1f} m")
    write_readme(outdir, bed, T, sheet_no, total_cut, sum(p.qty for p in main))
    return sheet_no, total_cut


def write_readme(outdir, bed, T, n_sheets, cut_len, n_pieces):
    (outdir / "README_FOR_LASER_SHOP.md").write_text(f"""# Laser-cut order: tabletop greenhouse frame

Flat parts for a small slot-together greenhouse frame (about 0.9 x 0.75 x 0.75 m assembled).

## Material

- **6 mm laser-grade birch plywood**, {n_sheets} sheets of **{bed[0]:.0f} x {bed[1]:.0f} mm** (one file per sheet).
- All slots are drawn for a sheet **{T:.1f} mm** thick. Cut on the line, so the laser makes every slot about one
  kerf wider; that small clearance is intended. If your sheet measures outside {T - 0.2:.1f}-{T + 0.2:.1f} mm, tell us
  before cutting and we send regenerated files.

## Files

| File | What |
|---|---|
| `00_TEST_cut_first.dxf` | **Cut this first** from the same sheet: slots labelled {T - 0.6:.1f}-{T + 0.4:.1f} mm and 3 keyhole widths. A strip of the sheet should slide into the `{T:.1f}` slot with a little play (because of the kerf, the snug one is usually labelled {T - 0.2:.1f}). If it doesn't go into `{T:.1f}` at all, stop and tell us. |
| `sheet_01.dxf` ... `sheet_{n_sheets:02d}.dxf` | The parts, nested, {n_pieces} pieces in total |
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
- Total cut length about **{cut_len / 1000:.0f} m**.

Questions about the parts: please contact the person who sent this order.
""")
