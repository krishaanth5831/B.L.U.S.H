FeatureScript 3083;
import(path : "onshape/std/geometry.fs", version : "3083.0");

/*
 * SO-101 Greenhouse - a fully printable, modular test greenhouse for picking
 * cherry tomatoes with an SO-101 arm (Vinea hardware track).
 *
 * Coordinates: +X runs along the rows, +Y across them, +Z up. Z = 0 is the table.
 * Three crop rows hang at y = -P, 0, +P (P = row pitch). The arm stands in an
 * aisle at y = +-P/2, halfway along the house, so every vine it works is P/2 away.
 *
 * Structure: two identical end frames (posts, ties, rafters, king post) joined by
 * long rails - an anchor rail on the table and a crop rail overhead for every row,
 * and a ridge. Every member is a hollow 20 mm square tube; every joint is a
 * printed square pin pushed half into each side, with an optional M3 lock bolt.
 * Nodes are drawn in the end-frame plane, so they print flat with no supports;
 * the through-socket for the long rail points straight up.
 *
 * Vines: horticultural twine hangs between keyholes (entry hole + narrow slot)
 * every 20 mm in the crop rail's underside and the anchor rail's top. Knot the
 * twine, drop the knot in, slide it into the slot: the vine is taut, indexed by
 * the tick marks, and can be re-hung in seconds.
 *
 * Each unique printable part is built once, rounded, then copied into the
 * assembly; the originals are laid out flat as "PRINT - ..." parts for export.
 */

const MM = millimeter;

const B_PITCH_ROW  = { (millimeter) : [200, 400, 800] } as LengthBoundSpec;
const B_HOUSE_LEN  = { (millimeter) : [300, 740, 1500] } as LengthBoundSpec;
const B_CROP_Z     = { (millimeter) : [250, 520, 900] } as LengthBoundSpec;
const B_ROOF       = { (degree) : [5, 22, 45] } as AngleBoundSpec;
const B_TUBE_W     = { (millimeter) : [12, 20, 40] } as LengthBoundSpec;
const B_WALL       = { (millimeter) : [0.8, 1.6, 4] } as LengthBoundSpec;
const B_PIN_CLR    = { (millimeter) : [0, 0.15, 0.6] } as LengthBoundSpec;
const B_SOCKET     = { (millimeter) : [10, 20, 40] } as LengthBoundSpec;
const B_MAX_SEG    = { (millimeter) : [100, 240, 340] } as LengthBoundSpec;
const B_BOLT       = { (millimeter) : [0, 3.4, 6] } as LengthBoundSpec;
const B_KEY_PITCH  = { (millimeter) : [10, 20, 60] } as LengthBoundSpec;
const B_KEY_D      = { (millimeter) : [2, 5, 10] } as LengthBoundSpec;
const B_KEY_SLOT   = { (millimeter) : [0.8, 1.8, 4] } as LengthBoundSpec;
const B_ARM_Z      = { (millimeter) : [50, 117, 300] } as LengthBoundSpec;
const B_ARM_REACH  = { (millimeter) : [200, 400, 700] } as LengthBoundSpec;
const B_EDGE_R     = { (millimeter) : [0, 0.6, 2] } as LengthBoundSpec;

annotation { "Feature Type Name" : "SO-101 Greenhouse",
             "Feature Type Description" : "Fully printable modular greenhouse with three hanging-vine rows for SO-101 cherry tomato picking." }
export const so101Greenhouse = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Group Name" : "Layout", "Collapsed By Default" : false }
        {
            annotation { "Name" : "Row pitch (vine row to vine row)" }
            isLength(definition.rowPitch, B_PITCH_ROW);
            annotation { "Name" : "House length (along the rows)" }
            isLength(definition.houseLen, B_HOUSE_LEN);
            annotation { "Name" : "Crop rail height (centre)" }
            isLength(definition.cropZ, B_CROP_Z);
            annotation { "Name" : "Roof pitch" }
            isAngle(definition.roofPitch, B_ROOF);
        }
        annotation { "Group Name" : "Members and joints", "Collapsed By Default" : false }
        {
            annotation { "Name" : "Tube outer size" }
            isLength(definition.tubeW, B_TUBE_W);
            annotation { "Name" : "Tube wall" }
            isLength(definition.wall, B_WALL);
            annotation { "Name" : "Pin clearance (per side)" }
            isLength(definition.pinClr, B_PIN_CLR);
            annotation { "Name" : "Socket depth (pin goes this far into each side)" }
            isLength(definition.socket, B_SOCKET);
            annotation { "Name" : "Longest printed tube" }
            isLength(definition.maxSeg, B_MAX_SEG);
            annotation { "Name" : "Lock bolt hole (M3 = 3.4, 0 = none)" }
            isLength(definition.boltD, B_BOLT);
        }
        annotation { "Group Name" : "Vines", "Collapsed By Default" : false }
        {
            annotation { "Name" : "Keyhole spacing" }
            isLength(definition.keyPitch, B_KEY_PITCH);
            annotation { "Name" : "Keyhole entry diameter (knot passes)" }
            isLength(definition.keyD, B_KEY_D);
            annotation { "Name" : "Keyhole slot width (twine passes, knot doesn't)" }
            isLength(definition.keySlot, B_KEY_SLOT);
        }
        annotation { "Group Name" : "SO-101 and finish", "Collapsed By Default" : true }
        {
            annotation { "Name" : "Shoulder height above table" }
            isLength(definition.armZ, B_ARM_Z);
            annotation { "Name" : "Reach from shoulder, incl. jaw" }
            isLength(definition.armReach, B_ARM_REACH);
            annotation { "Name" : "Edge rounding radius (0 = off)" }
            isLength(definition.edgeR, B_EDGE_R);
        }
        annotation { "Name" : "Make fit test coupon", "Default" : true }
        definition.makeCoupon is boolean;
        annotation { "Name" : "Show SO-101 reach, twines and tomatoes", "Default" : true }
        definition.showFit is boolean;
    }
    {
        // ------------------------------------------------------------ inputs (mm numbers)
        const P = definition.rowPitch / MM;
        const HL = definition.houseLen / MM;
        const CZ = definition.cropZ / MM;
        const pitch = definition.roofPitch;
        const cp = cos(pitch);
        const sp = sin(pitch);
        const W = definition.tubeW / MM;
        const wall = definition.wall / MM;
        const hw = W / 2;
        const S = W - 2 * wall;                               // socket = tube bore
        const pin = S - 2 * definition.pinClr / MM;           // pin cross-section
        const sd = definition.socket / MM;
        const maxSeg = definition.maxSeg / MM;
        const boltR = definition.boltD / MM / 2;
        const keyPitch = definition.keyPitch / MM;
        const keyR = definition.keyD / MM / 2;
        const keySlot = definition.keySlot / MM;
        const armZ = definition.armZ / MM;
        const reach = definition.armReach / MM;
        const edgeR = definition.edgeR / MM;

        const FZ = hw;                                        // foot node centre: tubes rest on the table
        const RZ = CZ + P * sp / cp;                          // ridge node centre

        // ------------------------------------------------------------ node arm lengths
        // Each arm holds one socket at its tip. Lengths are chosen so the base ties,
        // crop ties and rafters all come out the same tube length.
        const A0 = sd + hw;                                   // plain orthogonal arm
        const A_down = A0 + 10;                               // crop nodes' down arm (room for gussets)
        const A_rr = sd + 15;                                 // ridge rafter arm (clears the king post socket)
        // crop-node rafter arm: its socket must start where it clears the crop tie arm and tube
        const A_ct_min = (hw + 0.7 + hw * cp) / sp - 0;       // crop tie arm needed under the rafter arm
        const A_cr = max((hw + wall + S / 2 * cp) / sp + sd, A_ct_min / cp + sd + 6);
        const rafterLen = P / cp;
        const rafterTube = rafterLen - A_cr - A_rr;
        const A_ct = max(P - A0 - rafterTube, A_ct_min);      // crop tie arm at the outer crop node
        const tieTube = P - A_ct - A0;
        const A_fy = max((P - tieTube) / 2, A0);              // foot arms along Y (base ties)

        // ------------------------------------------------------------ registry + bookkeeping
        const reg = new box({ "n" : 0, "base" : id });
        const report = new box([]);                           // [name, qty, solid volume cm3]
        var rounding = "";

        // ============================================================ UNIQUE PARTS (local frames)
        // ---- nodes: drawn in the end-frame plane (local y, z), thickness along local x
        const nodeDefs = [
            ["F_out", [[0, 1, A0], [-1, 0, A_fy]], [[0, 1]], true],
            ["F_mid", [[0, 1, A0], [1, 0, A_fy], [-1, 0, A_fy]], [[0, 1], [0, 2]], false],
            ["C_out", [[0, -1, A_down], [-1, 0, A_ct], [-cp, sp, A_cr]], [[0, 1]], false],
            ["C_mid", [[0, -1, A_down], [0, 1, A0], [1, 0, A0], [-1, 0, A0]], [[0, 2], [0, 3], [1, 2], [1, 3]], false],
            ["R", [[0, -1, A0], [cp, -sp, A_rr], [-cp, -sp, A_rr]], [], false]
        ];
        var nodeParts = {};
        for (var nd in nodeDefs)
        {
            const kA = "nA_" ~ nd[0];
            const kC = "nC_" ~ nd[0];
            const plN = plane(v3(-hw, 0, 0), vector(1, 0, 0), vector(0, 1, 0));   // sketch (u, v) = (y, z)
            prismPlane(context, nid(reg, kA), plN, rrect(-hw, -hw, hw, hw), W);
            for (var arm in nd[1])
                prismPlane(context, nid(reg, kA), plN, armRect(arm[0], arm[1], 0, arm[2], hw), W);
            for (var g in nd[2])
            {
                const a = nd[1][g[0]];
                const b = nd[1][g[1]];
                const gl = min(a[2], b[2]) - hw - 2;
                const c = [hw * (a[0] + b[0]), hw * (a[1] + b[1])];
                prismPlane(context, nid(reg, kA), plN,
                    [c, [c[0] + gl * a[0], c[1] + gl * a[1]], [c[0] + gl * b[0], c[1] + gl * b[1]]], W);
            }
            if (nd[3])
            {
                // table tab (outward, on the table) with a 4.5 mm screw / clamp hole
                prismPlaneR(context, nid(reg, kA), plN, [[hw - 1, -hw, 0], [hw + 26, -hw, 6], [hw + 26, -hw + 4, 2], [hw - 1, -hw + 4, 0]], 0, W);
                cylAxis(context, nid(reg, kC), [0, hw + 15, -hw - 1], [0, hw + 15, -hw + 5], 2.25);
            }
            // sockets: one per arm, plus the through-socket for the long rail
            const plS = plane(v3(-S / 2, 0, 0), vector(1, 0, 0), vector(0, 1, 0));
            for (var arm in nd[1])
            {
                prismPlane(context, nid(reg, kC), plS, armRect(arm[0], arm[1], arm[2] - sd, arm[2] + 1, S / 2), S);
                if (boltR > 0)
                {
                    const r = arm[2] - sd / 2;
                    cylAxis(context, nid(reg, kC), [-hw - 1, r * arm[0], r * arm[1]], [hw + 1, r * arm[0], r * arm[1]], boltR);
                }
            }
            box3(context, nid(reg, kC), [-hw - 1, -S / 2, -S / 2], [hw + 1, S / 2, S / 2]);
            const q = finishPart(context, reg, kA, kC);
            rounding = rounding ~ nd[0] ~ " " ~ roundConvexEdges(context, reg, q, edgeR) ~ "; ";
            nodeParts[nd[0]] = q;
        }

        // ---- pin: 2 x socket depth long, lock-bolt hole at the middle of each half
        const pinLen = 2 * sd;
        box3(context, nid(reg, "pinA"), [-pinLen / 2, -pin / 2, -pin / 2], [pinLen / 2, pin / 2, pin / 2]);
        if (boltR > 0)
        {
            for (var sx in [-1, 1])
                cylAxis(context, nid(reg, "pinC"), [sx * sd / 2, 0, -pin], [sx * sd / 2, 0, pin], boltR);
        }
        const pinQ = finishPart(context, reg, "pinA", "pinC");
        rounding = rounding ~ "pin " ~ roundConvexEdges(context, reg, pinQ, edgeR) ~ "; ";

        // ---- tubes are made on demand, one per distinct (length, keyholes) combination
        const tubeParts = new box({});
        const tubeCount = new box({});

        // ============================================================ ASSEMBLY GRAPH
        // node instances: [kind, position, mirrored?]
        var nodeInst = {};
        for (var k in ["F_out", "F_mid", "C_out", "C_mid", "R"])
            nodeInst[k] = [];
        // placements for tubes and pins: [coordSystem origin, xAxis, zAxis]
        const pinPlaces = new box([]);
        const tubePlaces = new box({});

        for (var ex in [0, HL])
        {
            for (var sy in [-1, 1])
            {
                nodeInst["F_out"] = append(nodeInst["F_out"], [v3(ex, sy * P, FZ), sy < 0]);
                nodeInst["C_out"] = append(nodeInst["C_out"], [v3(ex, sy * P, CZ), sy < 0]);
            }
            nodeInst["F_mid"] = append(nodeInst["F_mid"], [v3(ex, 0, FZ), false]);
            nodeInst["C_mid"] = append(nodeInst["C_mid"], [v3(ex, 0, CZ), false]);
            nodeInst["R"] = append(nodeInst["R"], [v3(ex, 0, RZ), false]);

            // in-plane members: [from, to, arm at from, arm at to]; lock bolts run along global x
            var mem = [];
            for (var sy in [-1, 1])
            {
                mem = append(mem, [[ex, sy * P, FZ], [ex, 0, FZ], A_fy, A_fy]);          // base tie
                mem = append(mem, [[ex, sy * P, FZ], [ex, sy * P, CZ], A0, A_down]);     // outer post
                mem = append(mem, [[ex, sy * P, CZ], [ex, 0, CZ], A_ct, A0]);            // crop tie
                mem = append(mem, [[ex, sy * P, CZ], [ex, 0, RZ], A_cr, A_rr]);          // rafter
            }
            mem = append(mem, [[ex, 0, FZ], [ex, 0, CZ], A0, A_down]);                   // king post, lower
            mem = append(mem, [[ex, 0, CZ], [ex, 0, RZ], A0, A0]);                       // king post, upper
            for (var m in mem)
                placeMember(context, m[0], m[1], m[2], m[3], [1, 0, 0], false, maxSeg, tubePlaces, tubeCount, pinPlaces);
        }
        // long members between the two end frames (they butt against the node faces)
        for (var y in [-P, 0, P])
        {
            placeMember(context, [0, y, FZ], [HL, y, FZ], hw, hw, [0, 0, -1], true, maxSeg, tubePlaces, tubeCount, pinPlaces);   // anchor rail, keyholes up
            placeMember(context, [0, y, CZ], [HL, y, CZ], hw, hw, [0, 0, 1], true, maxSeg, tubePlaces, tubeCount, pinPlaces);    // crop rail, keyholes down
        }
        placeMember(context, [0, 0, RZ], [HL, 0, RZ], hw, hw, [0, 0, 1], false, maxSeg, tubePlaces, tubeCount, pinPlaces);       // ridge

        // ---- build each distinct tube once
        for (var key in keys(tubePlaces[]))
        {
            const spec = tubePlaces[][key][0];      // [length, keyholes]
            const kA = "tA_" ~ key;
            const kC = "tC_" ~ key;
            const kE = "tE_" ~ key;
            const L = spec[0];
            box3(context, nid(reg, kA), [0, -hw, -hw], [L, hw, hw]);
            box3(context, nid(reg, kC), [-1, -S / 2, -S / 2], [L + 1, S / 2, S / 2]);
            if (boltR > 0)
            {
                for (var bx in [sd / 2, L - sd / 2])
                    cylAxis(context, nid(reg, kC), [bx, 0, -hw - 1], [bx, 0, hw + 1], boltR);
            }
            if (spec[1])
            {
                // keyholes in the local -Z wall, clear of the pins at both ends
                var kx = sd + 10;
                while (kx + keyR + 7 <= L - sd - 3)
                {
                    zcyl(context, nid(reg, kC), kx, 0, -hw - 1, -hw + wall + 0.5, keyR);
                    box3(context, nid(reg, kC), [kx, -keySlot / 2, -hw - 1], [kx + 7, keySlot / 2, -hw + wall + 0.5]);
                    // tick marks on both sides, engraved after rounding
                    for (var syy in [-1, 1])
                        box3(context, nid(reg, kE), [kx - 0.35, syy * (hw + 1), -3], [kx + 0.35, syy * (hw - 0.4), 3]);
                    kx += keyPitch;
                }
            }
            const q = finishPart(context, reg, kA, kC);
            rounding = rounding ~ "tube " ~ key ~ " " ~ roundConvexEdges(context, reg, q, edgeR) ~ "; ";
            cutKind(context, reg, q, kE);
            var tpm = tubeParts[];
            tpm[key] = q;
            tubeParts[] = tpm;
        }

        // ============================================================ COPY INTO THE ASSEMBLY
        const asm = color(0.62, 0.78, 0.55);
        for (var k in keys(nodeInst))
        {
            var tf = [];
            var nm = [];
            var i = 0;
            for (var inst in nodeInst[k])
            {
                const xA = inst[1] ? vector(-1, 0, 0) : vector(1, 0, 0);
                tf = append(tf, toWorld(coordSystem(inst[0], xA, vector(0, 0, 1))));
                nm = append(nm, k ~ "_" ~ i);
                i += 1;
            }
            placeCopies(context, reg, nodeParts[k], tf, nm, "node " ~ k, asm);
            report[] = append(report[], ["node " ~ k, size(tf), evVolume(context, { "entities" : nodeParts[k] }) / (centimeter ^ 3)]);
        }
        for (var key in keys(tubePlaces[]))
        {
            var tf = [];
            var nm = [];
            var i = 0;
            for (var pl in subArray(tubePlaces[][key], 1, size(tubePlaces[][key])))
            {
                tf = append(tf, toWorld(coordSystem(pl[0], pl[1], pl[2])));
                nm = append(nm, "t" ~ key ~ "_" ~ i);
                i += 1;
            }
            const spec = tubePlaces[][key][0];
            const label = (spec[1] ? "rail " : "tube ") ~ fmt1(spec[0]);
            placeCopies(context, reg, tubeParts[][key], tf, nm, label, asm);
            report[] = append(report[], [label, size(tf), evVolume(context, { "entities" : tubeParts[][key] }) / (centimeter ^ 3)]);
        }
        {
            var tf = [];
            var nm = [];
            var i = 0;
            for (var pl in pinPlaces[])
            {
                tf = append(tf, toWorld(coordSystem(pl[0], pl[1], pl[2])));
                nm = append(nm, "pin_" ~ i);
                i += 1;
            }
            placeCopies(context, reg, pinQ, tf, nm, "pin", color(0.95, 0.75, 0.2));
            report[] = append(report[], ["pin", size(tf), evVolume(context, { "entities" : pinQ }) / (centimeter ^ 3)]);
        }

        // ============================================================ PRINT LAYOUT (originals, flat on z = 0)
        // laid out in front of the house (negative Y), one of each, named with the quantity
        const lay = new box([-P - 260, 0, 0]);    // [y row, x cursor, row depth]
        const pla = color(0.2, 0.65, 0.35);
        for (var k in ["F_out", "F_mid", "C_out", "C_mid", "R"])
        {
            const t = layoutNext(lay, 130, 110) * transform(v3(0, 0, hw)) * rotationAround(line(v3(0, 0, 0), vector(0, 1, 0)), -90 * degree);
            opTransform(context, nid(reg, "lay"), { "bodies" : nodeParts[k], "transform" : t });
            nameAs(context, nodeParts[k], "PRINT - node " ~ k ~ " (x" ~ size(nodeInst[k]) ~ ")", pla);
        }
        layoutRow(lay, 140);
        for (var key in keys(tubePlaces[]))
        {
            const spec = tubePlaces[][key][0];
            const n = size(tubePlaces[][key]) - 1;
            const t = layoutNext(lay, spec[0] + 20, 30) * transform(v3(0, 0, hw));
            opTransform(context, nid(reg, "lay"), { "bodies" : tubeParts[][key], "transform" : t });
            nameAs(context, tubeParts[][key], "PRINT - " ~ (spec[1] ? "rail " : "tube ") ~ fmt1(spec[0]) ~ " (x" ~ n ~ ")", pla);
        }
        layoutRow(lay, 40);
        {
            const t = layoutNext(lay, pinLen + 20, 30) * transform(v3(pinLen / 2, 0, pin / 2));
            opTransform(context, nid(reg, "lay"), { "bodies" : pinQ, "transform" : t });
            nameAs(context, pinQ, "PRINT - pin (x" ~ size(pinPlaces[]) ~ ")", pla);
        }

        // ============================================================ COUPON
        if (definition.makeCoupon)
        {
            // three sockets (bore -0.1 / nominal / +0.1) for a printed pin, and three keyhole slot widths
            const off = [-0.1, 0.0, 0.1];
            const slots = [keySlot - 0.3, keySlot, keySlot + 0.4];
            const blk = sd + 6;
            for (var i = 0; i < 3; i += 1)
            {
                const x0 = i * (W + 8);
                box3(context, nid(reg, "kA"), [x0, 0, 0], [x0 + W, W, blk]);
                const s2 = S / 2 + off[i] / 2;
                box3(context, nid(reg, "kC"), [x0 + hw - s2, hw - s2, blk - sd], [x0 + hw + s2, hw + s2, blk + 1]);
                // i + 1 dots on the side
                for (var j = 0; j <= i; j += 1)
                    cylAxis(context, nid(reg, "kE"), [x0 + 4 + j * 3, -1, 4], [x0 + 4 + j * 3, 0.4, 4], 0.7);
            }
            box3(context, nid(reg, "kA"), [0, 0, 0], [3 * (W + 8) - 8, W + 26, wall]);      // one plate under everything
            for (var i = 0; i < 3; i += 1)
            {
                const kx = 6 + i * (W + 8);
                zcyl(context, nid(reg, "kC"), kx, W + 16, -1, wall + 1, keyR);
                box3(context, nid(reg, "kC"), [kx, W + 16 - slots[i] / 2, -1], [kx + 7, W + 16 + slots[i] / 2, wall + 1]);
            }
            const cq = finishPart(context, reg, "kA", "kC");
            rounding = rounding ~ "coupon " ~ roundConvexEdges(context, reg, cq, min(edgeR, 0.4));
            cutKind(context, reg, cq, "kE");
            const t = layoutNext(lay, 3 * (W + 8) + 10, 60);
            opTransform(context, nid(reg, "lay"), { "bodies" : cq, "transform" : t });
            nameAs(context, cq, "PRINT - fit coupon (print first)", color(0.2, 0.55, 0.9));
        }

        // ============================================================ FIT CHECK: SO-101 reach, twines, tomatoes
        if (definition.showFit)
        {
            const ax = HL / 2;
            const ay = P / 2;
            opSphere(context, nid(reg, "fitReach"), { "center" : v3(ax, ay, armZ), "radius" : reach * MM });
            nameAs(context, qKind(reg, "fitReach"), "FIT - SO-101 reach envelope (aisle +Y)", color(0.3, 0.6, 1.0));
            try silent
            {
                setProperty(context, { "entities" : qKind(reg, "fitReach"), "propertyType" : PropertyType.APPEARANCE, "value" : color(0.3, 0.6, 1.0, 0.18) });
            }
            zcyl(context, nid(reg, "fitBase"), ax, ay, 0.05, armZ, 50);
            nameAs(context, qKind(reg, "fitBase"), "FIT - SO-101 base + shoulder", color(0.95, 0.95, 0.95));
            // twines and fruit on the two rows this aisle serves, hung from real keyholes
            // in the middle rail segment (keyholes 2, 5 and 8 of that segment)
            const nRail = ceil((HL - W) / maxSeg - 1e-6);
            const railSeg = (HL - W) / nRail;
            const segStart = hw + floor(nRail / 2) * railSeg;
            var tw = 0;
            for (var y in [0, P])
            {
                for (var kk in [1, 4, 7])
                {
                    const tx = segStart + sd + 10 + kk * keyPitch;
                    const dx = tx - ax;
                    zcyl(context, nid(reg, "fitTwine"), tx, y, W + 0.05, CZ - hw - 0.05, 0.9);
                    for (var h in [180, 260, 330])
                    {
                        const side = (y < ay) ? 1 : -1;
                        opSphere(context, nid(reg, "fitFruit"), { "center" : v3(ax + dx + 6 * ((tw % 3) - 1), y + side * 14, h - 20 * (tw % 2)), "radius" : 11 * MM });
                        tw += 1;
                    }
                }
            }
            nameAs(context, qKind(reg, "fitTwine"), "FIT - twine", color(0.55, 0.45, 0.3));
            nameAs(context, qKind(reg, "fitFruit"), "FIT - cherry tomato", color(0.85, 0.12, 0.1));
        }

        // ------------------------------------------------------------ summary
        var nPieces = 0;
        var vol = 0;
        var nKinds = 0;
        for (var r in report[])
        {
            nPieces += r[1];
            vol += r[1] * r[2];
            nKinds += 1;
        }
        reportFeatureInfo(context, id, nKinds ~ " printable parts, " ~ nPieces ~ " pieces, " ~ fmt1(vol * 1.24 / 1000) ~ " kg PLA if printed solid - footprint "
                    ~ fmt1(HL + W) ~ " x " ~ fmt1(2 * P + W) ~ " mm, ridge " ~ fmt1(RZ + hw) ~ " mm - rounded: " ~ rounding);
    });

// ================================================================ assembly helpers

// Split the straight member between two node points into printable tubes and record
// where every tube and every pin goes. zAx fixes the tube's roll (its lock-bolt axis).
function placeMember(context is Context, a is array, b is array, armA is number, armB is number, zAx is array,
    keyholes is boolean, maxSeg is number, tubePlaces is box, tubeCount is box, pinPlaces is box)
{
    const pa = vector(a[0], a[1], a[2]);
    const pb = vector(b[0], b[1], b[2]);
    const d = norm(pb - pa);
    const u = (pb - pa) / d;
    const z = vector(zAx[0], zAx[1], zAx[2]);
    const total = d - armA - armB;
    const n = ceil(total / maxSeg - 1e-6);
    const seg = total / n;
    const key = (keyholes ? "k" : "p") ~ round(seg * 10);
    var lst = tubePlaces[][key];
    if (lst == undefined)
        lst = [[seg, keyholes]];
    for (var i = 0; i < n; i += 1)
    {
        const s0 = pa + u * (armA + i * seg);
        lst = append(lst, [s0 * MM, u, z]);
    }
    var tp = tubePlaces[];
    tp[key] = lst;
    tubePlaces[] = tp;
    // a pin at both node joints and at every splice between tubes
    for (var i = 0; i <= n; i += 1)
    {
        const c = pa + u * (armA + i * seg);
        pinPlaces[] = append(pinPlaces[], [c * MM, u, z]);
    }
}

function placeCopies(context is Context, reg is box, part is Query, tfs is array, names is array, label is string, col is Color)
{
    const pid = nid(reg, "copies");
    opPattern(context, pid, { "entities" : part, "transforms" : tfs, "instanceNames" : names });
    nameAs(context, qCreatedBy(pid, EntityType.BODY), label, col);
}

function nameAs(context is Context, q is Query, name is string, col is Color)
{
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.NAME, "value" : name });
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.APPEARANCE, "value" : col });
}

// next slot in the print layout: lay = [y, x cursor, row depth]
function layoutNext(lay is box, w is number, depth is number) returns Transform
{
    var l = lay[];
    if (l[1] + w > 1100)
    {
        l = [l[0] - l[2] - 20, 0, 0];
    }
    const t = transform(v3(l[1], l[0] - depth / 2, 0));
    l[1] = l[1] + w;
    l[2] = max(l[2], depth);
    lay[] = l;
    return t;
}

function layoutRow(lay is box, depth is number)
{
    var l = lay[];
    lay[] = [l[0] - max(l[2], depth) - 20, 0, 0];
}

// rectangle along unit direction (dy, dz) from r0 to r1, half-width h, in (y, z)
function armRect(dy is number, dz is number, r0 is number, r1 is number, h is number) returns array
{
    const ny = -dz;
    const nz = dy;
    return [[r0 * dy + h * ny, r0 * dz + h * nz], [r1 * dy + h * ny, r1 * dz + h * nz],
            [r1 * dy - h * ny, r1 * dz - h * nz], [r0 * dy - h * ny, r0 * dz - h * nz]];
}

// ================================================================ geometry helpers (shared with the drone frame)

function v3(x is number, y is number, z is number) returns Vector
{
    return vector(x, y, z) * MM;
}

function nid(reg is box, kind is string) returns Id
{
    var r = reg[];
    r.n = r.n + 1;
    const newId = r.base + ("o" ~ r.n);
    var lst = r[kind];
    if (lst == undefined)
        lst = [];
    r[kind] = append(lst, newId);
    reg[] = r;
    return newId;
}

function qKind(reg is box, kind is string) returns Query
{
    var qs = [];
    const lst = reg[][kind];
    if (lst != undefined)
    {
        for (var i in lst)
            qs = append(qs, qCreatedBy(i, EntityType.BODY));
    }
    return qBodyType(qUnion(qs), BodyType.SOLID);
}

function box3(context is Context, id is Id, p is array, q is array)
{
    fCuboid(context, id, {
                "corner1" : v3(min(p[0], q[0]), min(p[1], q[1]), min(p[2], q[2])),
                "corner2" : v3(max(p[0], q[0]), max(p[1], q[1]), max(p[2], q[2])) });
}

function zcyl(context is Context, id is Id, x is number, y is number, z0 is number, z1 is number, r is number)
{
    fCylinder(context, id, { "bottomCenter" : v3(x, y, z0), "topCenter" : v3(x, y, z1), "radius" : r * MM });
}

function cylAxis(context is Context, id is Id, p is array, q is array, r is number)
{
    fCylinder(context, id, { "bottomCenter" : v3(p[0], p[1], p[2]), "topCenter" : v3(q[0], q[1], q[2]), "radius" : r * MM });
}

function rrect(x0 is number, y0 is number, x1 is number, y1 is number) returns array
{
    return [[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]];
}

function prismPlane(context is Context, id is Id, pl is Plane, pts is array, depth is number)
{
    prismPlaneR(context, id, pl, pts, 0, depth);
}

function prismPlaneR(context is Context, id is Id, pl is Plane, pts is array, r is number, depth is number)
{
    const skId = id + "sk";
    const sk = newSketchOnPlane(context, skId, { "sketchPlane" : pl });
    sketchRounded(sk, pts, r);
    skSolve(sk);
    opExtrude(context, id + "ex", {
                "entities" : qSketchRegion(skId),
                "direction" : pl.normal,
                "endBound" : BoundingType.BLIND,
                "endDepth" : depth * MM });
    opDeleteBodies(context, id + "del", { "entities" : qCreatedBy(skId, EntityType.BODY) });
}

function sketchRounded(sk is Sketch, pts is array, r is number)
{
    const n = size(pts);
    var t1 = [];
    var t2 = [];
    var mid = [];
    var has = [];
    for (var i = 0; i < n; i += 1)
    {
        const ip = (i + n - 1) % n;
        const inx = (i + 1) % n;
        const p = vector(pts[i][0], pts[i][1]);
        const pp = vector(pts[ip][0], pts[ip][1]);
        const pn = vector(pts[inx][0], pts[inx][1]);
        const ri = size(pts[i]) > 2 ? pts[i][2] : r;
        const ua = normalize(pp - p);
        const ub = normalize(pn - p);
        const phi = acos(max(-1, min(1, dot(ua, ub))));
        var t = 0;
        var rr = ri;
        if (ri > 0 && phi < 178 * degree && phi > 2 * degree)
        {
            t = ri / tan(phi / 2);
            const tMax = 0.49 * min(norm(pp - p), norm(pn - p));
            if (t > tMax)
            {
                t = tMax;
                rr = t * tan(phi / 2);
            }
        }
        if (t > 0.005)
        {
            const m = normalize(ua + ub);
            const ctr = p + m * (rr / sin(phi / 2));
            t1 = append(t1, p + ua * t);
            t2 = append(t2, p + ub * t);
            mid = append(mid, ctr - m * rr);
            has = append(has, true);
        }
        else
        {
            t1 = append(t1, p);
            t2 = append(t2, p);
            mid = append(mid, p);
            has = append(has, false);
        }
    }
    for (var i = 0; i < n; i += 1)
    {
        const j = (i + 1) % n;
        if (norm(t1[j] - t2[i]) > 0.005)
            skLineSegment(sk, "l" ~ i, { "start" : t2[i] * MM, "end" : t1[j] * MM });
        if (has[i])
            skArc(sk, "a" ~ i, { "start" : t1[i] * MM, "mid" : mid[i] * MM, "end" : t2[i] * MM });
    }
}

function finishPart(context is Context, reg is box, addKind is string, cutKindName is string) returns Query
{
    const adds = qKind(reg, addKind);
    if (size(evaluateQuery(context, adds)) > 1)
        opBoolean(context, nid(reg, "ops"), { "tools" : adds, "operationType" : BooleanOperationType.UNION });
    cutKind(context, reg, adds, cutKindName);
    return adds;
}

function cutKind(context is Context, reg is box, target is Query, kind is string)
{
    const cuts = qKind(reg, kind);
    if (size(evaluateQuery(context, cuts)) > 0)
        opBoolean(context, nid(reg, "ops"), { "tools" : cuts, "targets" : target, "operationType" : BooleanOperationType.SUBTRACTION });
}

// Fillet every convex edge, chain by chain; chains that refuse get a smaller radius.
function roundConvexEdges(context is Context, reg is box, part is Query, r is number) returns string
{
    if (r <= 0)
        return "off";
    const edges = evaluateQuery(context, qOwnedByBody(part, EntityType.EDGE));
    var isConvex = {};
    for (var e in edges)
    {
        var convex = false;
        try silent
        {
            convex = evEdgeConvexity(context, { "edge" : e }) == EdgeConvexityType.CONVEX;
        }
        if (convex)
            isConvex[transientQueriesToStrings(e)] = true;
    }
    var used = {};
    var chains = [];
    var total = 0;
    for (var e in edges)
    {
        const k = transientQueriesToStrings(e);
        if (isConvex[k] != true || used[k] == true)
            continue;
        var mids = [];
        for (var m in evaluateQuery(context, qUnion([e, qTangentConnectedEdges(e)])))
        {
            const km = transientQueriesToStrings(m);
            if (used[km] == true || isConvex[km] != true)
                continue;
            used[km] = true;
            mids = append(mids, evEdgeTangentLine(context, { "edge" : m, "parameter" : 0.5 }).origin);
        }
        if (size(mids) > 0)
        {
            chains = append(chains, mids);
            total += size(mids);
        }
    }
    if (total == 0)
        return "no edges";
    const failed = new box([]);
    filletSplit(context, reg, part, chains, r, failed);
    var smaller = 0;
    for (var f in [0.6, 0.35])
    {
        var still = [];
        for (var c in failed[])
        {
            if (tryFillet(context, reg, part, c, r * f))
                smaller += size(c);
            else
                still = append(still, c);
        }
        failed[] = still;
    }
    var nf = 0;
    for (var c in failed[])
        nf += size(c);
    if (nf == 0)
        return "all " ~ total ~ (smaller > 0 ? " (" ~ smaller ~ " smaller)" : "");
    var msg = (total - nf) ~ "/" ~ total ~ " sharp at";
    for (var k = 0; k < min(size(failed[]), 4); k += 1)
    {
        const m = failed[][k][0];
        msg = msg ~ " (" ~ fmt1(m[0] / MM) ~ "," ~ fmt1(m[1] / MM) ~ "," ~ fmt1(m[2] / MM) ~ ")";
    }
    return msg;
}

function filletSplit(context is Context, reg is box, part is Query, chains is array, r is number, failed is box)
{
    if (size(chains) == 0)
        return;
    var mids = [];
    for (var c in chains)
        mids = concatenateArrays([mids, c]);
    if (tryFillet(context, reg, part, mids, r))
        return;
    if (size(chains) == 1)
    {
        var alive = 0;
        for (var m in chains[0])
            alive += size(evaluateQuery(context, qContainsPoint(qOwnedByBody(part, EntityType.EDGE), m)));
        if (alive > 0)
            failed[] = append(failed[], chains[0]);
        return;
    }
    const h = floor(size(chains) / 2);
    filletSplit(context, reg, part, subArray(chains, 0, h), r, failed);
    filletSplit(context, reg, part, subArray(chains, h, size(chains)), r, failed);
}

function tryFillet(context is Context, reg is box, part is Query, mids is array, r is number) returns boolean
{
    var qs = [];
    for (var m in mids)
        qs = append(qs, qContainsPoint(qOwnedByBody(part, EntityType.EDGE), m));
    const q = qUnion(qs);
    if (size(evaluateQuery(context, q)) == 0)
        return true;
    var ok = false;
    try silent
    {
        opFillet(context, nid(reg, "ops"), { "entities" : q, "radius" : r * MM });
        ok = true;
    }
    return ok;
}

function fmt1(x is number) returns string
{
    const r = round(abs(x) * 10);
    const sign = (x < 0 && r > 0) ? "-" : "";
    return sign ~ floor(r / 10) ~ "." ~ (r - 10 * floor(r / 10));
}
