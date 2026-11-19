FeatureScript 3083;
import(path : "onshape/std/geometry.fs", version : "3083.0");

/*
 * Vinea pipe-rail test rig, v3 (2026-11-19): built only from the parts in SHOPPING_LIST_GAMMA.md.
 *
 * Every part is named "<code> | <what> | <length> | [<shopping list item>] <Gamma product>", and the wooden
 * parts carry their code embossed on a visible face. Next to the rig, a CUT PLAN lays out every stock length
 * bought (4 battens, the deck plank, 2 conduits, 8 dowels) with its pieces in cutting order, red saw cuts
 * (3 mm kerf), the mark positions measured from the left end, and the offcuts.
 *
 * Fixed by the arm, from the B.L.U.S.H / S.H.E.A.R sims (scene.py): arm base mid-aisle, 200 mm from each row;
 * stems hang from string rails 520 mm above the arm base; trusses 140-320 mm above the arm base; birdseye
 * camera 150 mm behind the row the arm does not face, 340 mm above the arm base.
 * Design choices (v2, unverified for sim2real): posts on the row lines, arms 450 mm apart facing opposite rows,
 * no riser, stems 30 mm off the row line.
 *
 * Joints:
 *   row frames (floor rail + 2 posts + string rail): 2 x [9] 5x80 screws per joint, permanent
 *   top ties + floor tie: [5] 50x50x40 angle screwed to the row frame with [10] 4x30, one [6] M6x60 carriage
 *   bolt through tie + angle, [8] washer, [7] wing nut -> the ties come off and the rig flat-packs
 *   sleepers: screwed to nothing, the tubes sit on them; one [10] 4x30 through each tube into each sleeper
 *   wheels (option A): 3D-printed U-groove wheels + brackets, [6] M6x60 axle, [12] M6 lock nut
 * Axes: x along the rows (0 = inside face of the end posts), y across (row A at 0, row B at the row pitch),
 * z up from the floor. Units mm.
 */

const T = 44;                     // [1] planed spruce 44 x 44 (Gamma B736850)
const PLANK_T = 18;               // [2] planed spruce 18 x 119 (Gamma B270673)
const PLANK_W = 119;
const TUBE_R = 9.5;               // [3] Martens grey conduit 19 mm (Gamma B461108)
const STEM_R = 4;                 // [4] beech round stick 8 mm (Gamma B203982)
const GAUGE = 160;                // rail tube centre to centre (choice: one tube under each deck strip)
const WHEEL_R = 25;               // printed wheel, 50 mm (choice)
const STEM_OFF = 30;              // stem off the row line, aisle side (clears the 44 mm rails)
const FRUIT_R = 11;               // 22 mm printed cherry tomatoes
const KERF = 3;                   // hand/circular saw kerf, assumed
const BAT_LEN = 2700;             // stock lengths as sold
const PLANK_LEN = 2400;
const TUBE_LEN = 2000;
const DOWEL_LEN = 1000;
const FONT = "OpenSans-Regular.ttf";

annotation { "Feature Type Name" : "Pipe-Rail Test Rig" }
export const pipeRailRig = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Inside length" }
        isLength(definition.houseLen, { (millimeter) : [500, 900, 3000] } as LengthBoundSpec);
        annotation { "Name" : "Row pitch (row A to row B)" }
        isLength(definition.rowPitch, { (millimeter) : [250, 400, 1000] } as LengthBoundSpec);
        annotation { "Name" : "Arm spacing along the aisle" }
        isLength(definition.armGap, { (millimeter) : [150, 450, 2000] } as LengthBoundSpec);
        annotation { "Name" : "Arm riser height (on the cart deck)" }
        isLength(definition.riserH, { (millimeter) : [0, 0, 600] } as LengthBoundSpec);
        annotation { "Name" : "String rail height above the arm base" }
        isLength(definition.cropAbove, { (millimeter) : [200, 520, 800] } as LengthBoundSpec);
        annotation { "Name" : "Show cut plan" }
        definition.showCutPlan is boolean;
        annotation { "Name" : "Emboss part codes" }
        definition.labels is boolean;
    }
    {
        try
        {
            rigBody(context, id, definition);
        }
        catch (e)
        {
            setVariable(context, "rigErr", e);
            reportFeatureWarning(context, id, "rig failed, see rigErr");
        }
    });

function rigBody(context is Context, id is Id, definition is map)
{
    {
        const L = definition.houseLen / millimeter;
        const P = definition.rowPitch / millimeter;
        const gap = definition.armGap / millimeter;
        const riser = definition.riserH / millimeter;
        const aisle = P / 2;
        const armX = [L / 2 - gap / 2, L / 2 + gap / 2];
        const armYaw = [90, -90];                              // arm 1 faces row B (+y), arm 2 row A (-y)
        const armRow = [P, 0];

        const reg = new box({ "base" : id, "n" : 0, "cut" : {}, "labels" : definition.labels });
        const wood = color(0.86, 0.72, 0.50);
        const wood2 = color(0.80, 0.64, 0.42);
        const ply = color(0.78, 0.62, 0.40);
        const grey = color(0.62, 0.63, 0.66);
        const steel = color(0.55, 0.56, 0.6);
        const dark = color(0.18, 0.18, 0.2);
        const printCol = color(0.25, 0.45, 0.85);

        const B1 = "[1] Gamma B736850 vuren 44x44 270 cm";
        const B2 = "[2] Gamma B270673 vuren 18x119 240 cm";
        const B3 = "[3] Gamma B461108 Martens elektrabuis grijs 19 mm 2 m";
        const B4 = "[4] Gamma B203982 beuken stok 8 mm 100 cm";

        // --- heights: everything hangs off the arm base ---
        const tubeZ = T + TUBE_R;                              // tubes rest on the 44 mm sleepers
        const wheelZ = tubeZ + TUBE_R - 5 + WHEEL_R;           // wheel groove sits 5 mm down over the tube
        const deckZ = wheelZ + WHEEL_R + 8;                    // deck underside
        const armZ = deckZ + PLANK_T + riser;
        const cropZ = armZ + definition.cropAbove / millimeter;    // underside of the string rails

        // --- row frames: floor rail, end posts, string rail on top (screwed, permanent) ---
        const rows = [[0, "A", -1, "1", "B1-3", "B3-1"], [P, "B", 1, "2", "B2-3", "B3-2"]];
        for (var r in rows)
        {
            const ry = r[0];
            const out = r[2];                                  // outward side of this row frame
            const face = out < 0 ? "-y" : "+y";
            piece(context, reg, "B" ~ r[3] ~ "-2", "floor rail row " ~ r[1], B1, [-T, ry - T / 2, 0], [L + T, ry + T / 2, T], wood, face);
            piece(context, reg, r[4], "post row " ~ r[1] ~ " (closed end)", B1, [-T, ry - T / 2, T], [0, ry + T / 2, cropZ], wood2, face);
            piece(context, reg, r[5], "post row " ~ r[1] ~ " (open end)", B1, [L, ry - T / 2, T], [L + T, ry + T / 2, cropZ], wood2, face);
            piece(context, reg, "B" ~ r[3] ~ "-1", "string rail row " ~ r[1] ~ " (stems hang from it)", B1, [-T, ry - T / 2, cropZ], [L + T, ry + T / 2, cropZ + T], wood, face);
        }

        // --- cross ties, bolted so the rig flat-packs ---
        piece(context, reg, "B3-3", "top cross tie (closed end, bolted)", B1, [-T, -T / 2, cropZ + T], [0, P + T / 2, cropZ + 2 * T], wood2, "+z");
        piece(context, reg, "B3-4", "top cross tie (open end, bolted)", B1, [L, -T / 2, cropZ + T], [L + T, P + T / 2, cropZ + 2 * T], wood2, "+z");
        piece(context, reg, "B3-5", "floor cross tie (closed end only, bolted)", B1, [-T, T / 2, 0], [0, P - T / 2, T], wood2, "+z");
        for (var ry in [0, P])
        {
            topJoint(context, reg, ry, 0, 1, cropZ, steel);            // closed end: bracket inside at x = 0
            topJoint(context, reg, ry, L, -1, cropZ, steel);           // open end: bracket inside at x = L
        }
        floorJoint(context, reg, T / 2, 1, steel);
        floorJoint(context, reg, P - T / 2, -1, steel);

        // --- pipe rail: two conduits on four sleepers, running out past the open end ---
        var sx = 150;
        var k = 1;
        while (sx < L + 250 && k <= 4)
        {
            piece(context, reg, "B4-" ~ k, "rail sleeper", B1, [sx - T / 2, aisle - 120, 0], [sx + T / 2, aisle + 120, T], wood2, "+z");
            sx += 300;
            k += 1;
        }
        const tubeX0 = 20;
        const tubeX1 = L + 300;
        var tn = 1;
        for (var s in [-1, 1])
        {
            cylPiece(context, reg, "T" ~ tn, "rail tube (heating pipe stand-in), 1 screw 4x30 into each sleeper", B3,
                    [tubeX0, aisle + s * GAUGE / 2, tubeZ], [tubeX1, aisle + s * GAUGE / 2, tubeZ], TUBE_R, grey);
            tn += 1;
        }

        // --- cart: two deck strips on two cleats, four printed wheels on M6 axles ---
        const deckX0 = armX[0] - 140;
        const deckX1 = armX[1] + 140;
        piece(context, reg, "D-1", "cart deck strip", B2, [deckX0, aisle - PLANK_W, deckZ], [deckX1, aisle, deckZ + PLANK_T], ply, "+z");
        piece(context, reg, "D-2", "cart deck strip", B2, [deckX0, aisle, deckZ], [deckX1, aisle + PLANK_W, deckZ + PLANK_T], ply, "+z");
        var cn = 3;
        for (var ax in armX)
        {
            piece(context, reg, "D-" ~ cn, "deck cleat under arm " ~ (cn - 2) ~ " (ties the strips, arm bolts through)", B2,
                    [ax - PLANK_W / 2, aisle - PLANK_W, deckZ - PLANK_T], [ax + PLANK_W / 2, aisle + PLANK_W, deckZ], wood2, undefined);
            cn += 1;
        }
        for (var wx in [deckX0 + 50, deckX1 - 50])
        {
            for (var s in [-1, 1])
            {
                const wy = aisle + s * GAUGE / 2;
                wheel(context, reg, wx, wy, wheelZ, deckZ, printCol, steel);
            }
        }
        if (riser > 0)
        {
            for (var ax in armX)
                boxPart(context, reg, [ax - 75, aisle - 75, deckZ + PLANK_T], [ax + 75, aisle + 75, armZ - 0.5], "Riser block (not on the shopping list)", ply);
        }

        // --- stems (dowels) + 3D-printed trusses in front of each arm ---
        const red = color(0.85, 0.12, 0.08);
        const green = color(0.45, 0.68, 0.22);
        const orange = color(0.95, 0.5, 0.1);
        var sn = 1;
        for (var a = 0; a < 2; a += 1)
        {
            const ry = armRow[a];
            const toAisle = ry == 0 ? 1 : -1;
            const sy = ry + toAisle * STEM_OFF;
            var n = 0;
            for (var dx in [-150, -75, 0, 75])
            {
                const px = armX[a] + dx;
                cylPiece(context, reg, "S" ~ sn, "stem, row " ~ (ry == 0 ? "A" : "B") ~ " (screw eye or drilled hole top and bottom)", B4,
                        [px, sy, T], [px, sy, cropZ], STEM_R, color(0.30, 0.55, 0.22));
                sn += 1;
                var t = 0;
                for (var h in [armZ + 160 + 40 * (n % 2), armZ + 280 + 30 * (n % 2)])
                {
                    const kind = (n + t) % 3;           // 0 ready, 1 almost (tip green), 2 unripe
                    const fy = sy + toAisle * 35;
                    boxPart(context, reg, [px - 6, sy - 6, h - 6], [px + 6, sy + 6, h + 6], "PRINT | truss clip (grips the 8 mm stem)", dark);
                    cylPart(context, reg, [px, sy, h], [px, fy, h - 25], 2.5, "PRINT | truss stalk", color(0.35, 0.55, 0.25));
                    for (var f = 0; f < 7; f += 1)
                    {
                        const fdx = (f - 3) * 21;
                        const dz = -35 - 4 * abs(f - 3) - 12 * (f % 2);
                        const tip = (f == 0 || f == 6);
                        const col = kind == 0 ? red : (kind == 1 ? (tip ? green : red) : (f % 3 == 0 ? orange : green));
                        const lab = kind == 0 ? "PRINT | truss fruit, ready (red)" : (kind == 1 ? "PRINT | truss fruit, almost (red, tip green)" : "PRINT | truss fruit, unripe (green)");
                        spherePart(context, reg, [px + fdx, fy + toAisle * 4 * (f % 2), h + dz], FRUIT_R, lab, col);
                    }
                    t += 1;
                }
                n += 1;
            }
        }

        // --- birdseye cameras on free-standing wooden masts, behind the row each arm does NOT face ---
        var fn = 7;
        for (var a = 0; a < 2; a += 1)
        {
            const cy = armRow[a] == P ? -150 : P + 150;
            const camZ = armZ + 340;
            const ax = armX[a];
            piece(context, reg, "B4-" ~ fn, "camera mast foot " ~ (a + 1), B1, [ax - 90, cy - T, 0], [ax + 90, cy, T], wood, undefined);
            piece(context, reg, "B4-" ~ (fn + 1), "camera mast foot " ~ (a + 1), B1, [ax - 90, cy, 0], [ax + 90, cy + T, T], wood, undefined);
            fn += 2;
            piece(context, reg, "B4-" ~ (5 + a), "camera mast pole " ~ (a + 1), B1, [ax - T / 2, cy - T / 2, T], [ax + T / 2, cy + T / 2, camZ - 20], wood2, "-x");
            const fc = armRow[a] == P ? 1 : -1;
            boxPart(context, reg, [ax - 16, cy - fc * 8 + fc * T / 2, camZ - 20], [ax + 16, cy + fc * 17 + fc * T / 2, camZ + 12], "OWN | birdseye camera InnoMaker U20CAM-1080P (screw to the pole top)", dark);
        }

        // --- size reference ---
        spherePart(context, reg, [-300, -350, 110], 110, "Size reference | football (size 5, 220 mm)", color(0.97, 0.97, 0.97));

        // --- cut plan: every stock length, its pieces in cutting order, saw cuts, marks, offcuts ---
        if (definition.showCutPlan)
        {
            const plan = [
                    ["BATTEN 1 OF 4", B1, BAT_LEN, "sq", ["B1-1", "B1-2", "B1-3"]],
                    ["BATTEN 2 OF 4", B1, BAT_LEN, "sq", ["B2-1", "B2-2", "B2-3"]],
                    ["BATTEN 3 OF 4", B1, BAT_LEN, "sq", ["B3-1", "B3-2", "B3-3", "B3-4", "B3-5"]],
                    ["BATTEN 4 OF 4", B1, BAT_LEN, "sq", ["B4-1", "B4-2", "B4-3", "B4-4", "B4-5", "B4-6", "B4-7", "B4-8", "B4-9", "B4-10"]],
                    ["DECK PLANK", B2, PLANK_LEN, "plank", ["D-1", "D-2", "D-3", "D-4"]],
                    ["CONDUIT 1 OF 2", B3, TUBE_LEN, "tube", ["T1"]],
                    ["CONDUIT 2 OF 2", B3, TUBE_LEN, "tube", ["T2"]],
                    ["DOWEL 1 OF 8", B4, DOWEL_LEN, "dowel", ["S1"]], ["DOWEL 2 OF 8", B4, DOWEL_LEN, "dowel", ["S2"]],
                    ["DOWEL 3 OF 8", B4, DOWEL_LEN, "dowel", ["S3"]], ["DOWEL 4 OF 8", B4, DOWEL_LEN, "dowel", ["S4"]],
                    ["DOWEL 5 OF 8", B4, DOWEL_LEN, "dowel", ["S5"]], ["DOWEL 6 OF 8", B4, DOWEL_LEN, "dowel", ["S6"]],
                    ["DOWEL 7 OF 8", B4, DOWEL_LEN, "dowel", ["S7"]], ["DOWEL 8 OF 8", B4, DOWEL_LEN, "dowel", ["S8"]]];
            var y = -650;
            floorText(context, reg, "CUT PLAN - measure every mark from the LEFT end of the stock, cut on the right of the line (3 mm saw kerf)", [0, y + 90, 0], 30, 2600, dark);
            for (var b in plan)
            {
                const sec = b[3];
                if (sec == "plank")
                    y -= 60;                                   // room for batten 4's mark labels above the plank title
                const halfW = sec == "sq" ? T / 2 : (sec == "plank" ? PLANK_W / 2 : (sec == "tube" ? TUBE_R : STEM_R));
                const ht = sec == "sq" ? T : (sec == "plank" ? PLANK_T : 2 * halfW);
                floorText(context, reg, b[0] ~ "  -  " ~ b[1], [0, y + halfW + 22, 0], sec == "dowel" ? 13 : 18, 1500, dark);
                var cur = 0;
                var idx = 0;
                for (var code in b[4])
                {
                    const len = reg[].cut[code];
                    if (len == undefined)
                        continue;
                    const col = idx % 2 == 0 ? color(0.95, 0.80, 0.45) : color(0.70, 0.88, 0.55);
                    const nm = "CUT PLAN | " ~ code ~ " | " ~ len ~ " mm | from " ~ b[0];
                    const txt = code ~ "  " ~ len;
                    if (sec == "tube" || sec == "dowel")
                    {
                        const pid = nextId(reg);
                        fCylinder(context, pid, { "bottomCenter" : v3mm([cur, y, halfW]), "topCenter" : v3mm([cur + len, y, halfW]), "radius" : halfW * millimeter });
                        finish(context, pid, nm, col);
                        floorText(context, reg, txt, [cur + len / 2 - 60, y - halfW - (sec == "dowel" ? 12 : 18), 0], sec == "dowel" ? 9 : 14, len * 0.8, dark);
                    }
                    else
                    {
                        const pid = nextId(reg);
                        fCuboid(context, pid, { "corner1" : v3mm([cur, y - halfW, 0]), "corner2" : v3mm([cur + len, y + halfW, ht]) });
                        emboss(context, reg, pid, txt, [cur + len / 2, y, ht], vector(0, 0, 1), vector(1, 0, 0), min(0.6 * 2 * halfW, 40), len * 0.85);
                        finish(context, pid, nm, col);
                    }
                    cur += len;
                    // saw cut: a red slab across the stock, and the mark position below it
                    if (cur < b[2])
                    {
                        boxPart(context, reg, [cur, y - halfW - 6, -1], [cur + KERF, y + halfW + 6, ht + 8], "CUT PLAN | saw cut at " ~ cur ~ " mm from the left end", color(0.9, 0.1, 0.1));
                        floorText(context, reg, "" ~ cur, [cur - 20, y - halfW - (sec == "dowel" ? 24 : 34), 0], sec == "dowel" ? 9 : 13, 80, color(0.9, 0.1, 0.1));
                        cur += KERF;
                    }
                    idx += 1;
                }
                if (cur < b[2] - 1)
                {
                    const rest = b[2] - cur;
                    const nm = "CUT PLAN | offcut " ~ round(rest) ~ " mm (spare) | " ~ b[0];
                    const pid = nextId(reg);
                    if (sec == "tube" || sec == "dowel")
                        fCylinder(context, pid, { "bottomCenter" : v3mm([cur, y, halfW]), "topCenter" : v3mm([b[2], y, halfW]), "radius" : halfW * millimeter });
                    else
                    {
                        fCuboid(context, pid, { "corner1" : v3mm([cur, y - halfW, 0]), "corner2" : v3mm([b[2], y + halfW, ht]) });
                        if (rest > 90)
                            emboss(context, reg, pid, "OFFCUT", [cur + rest / 2, y, ht], vector(0, 0, 1), vector(1, 0, 0), min(0.5 * 2 * halfW, 30), rest * 0.8);
                    }
                    finish(context, pid, nm, color(0.75, 0.75, 0.75));
                }
                y -= sec == "dowel" ? 75 : (sec == "tube" ? 85 : (sec == "plank" ? 200 : 125));
            }
        }

        setVariable(context, "rigInfo", { "armZ" : armZ, "cropZ" : cropZ, "deckZ" : deckZ, "armX" : armX, "aisle" : aisle, "armYaw" : armYaw });
        setVariable(context, "cutList", reg[].cut);
    }
}

// ---------- joints ----------

// top tie to string rail: angle bracket inside the joint, one leg on top of the string rail (screwed),
// one leg against the tie's inner face; carriage bolt through the tie along x, washer + wing nut inside
function topJoint(context is Context, reg is box, ry is number, xFace is number, inward is number, cropZ is number, steel is Color)
{
    const zTop = cropZ + T;                                    // top of the string rail = bottom of the tie
    boxPart(context, reg, [xFace, ry - 20, zTop], [xFace + inward * 2, ry + 20, zTop + 50], "[5] Gamma B105042 angle bracket 50x50x40 (leg against tie, bolted)", steel);
    boxPart(context, reg, [xFace, ry - 20, zTop], [xFace + inward * 50, ry + 20, zTop + 2], "[5] Gamma B105042 angle bracket 50x50x40 (leg on string rail, 2 x [10] 4x30)", steel);
    bolt(context, reg, [xFace - inward * (T + 4), ry, zTop + T / 2], vector(inward, 0, 0), steel);
}

// floor tie to floor rail: upright angle in the corner, one leg on the floor rail's aisle face (screwed),
// one leg against the tie's inner face (bolted)
function floorJoint(context is Context, reg is box, yFace is number, inward is number, steel is Color)
{
    boxPart(context, reg, [0, yFace, 2], [50, yFace + inward * 2, 42], "[5] Gamma B105042 angle bracket 50x50x40 (leg on floor rail, 2 x [10] 4x30)", steel);
    boxPart(context, reg, [0, yFace, 2], [2, yFace + inward * 50, 42], "[5] Gamma B105042 angle bracket 50x50x40 (leg against floor tie, bolted)", steel);
    bolt(context, reg, [-(T + 4), yFace + inward * 28, 22], vector(1, 0, 0), steel);
}

// [6] M6x60 carriage bolt from `head` along `dir`, with [8] washer and [7] wing nut at the far end
function bolt(context is Context, reg is box, head is array, dir is Vector, steel is Color)
{
    const h = vector(head[0], head[1], head[2]);
    const p = function(d) { const v = h + dir * d; return [v[0], v[1], v[2]]; };
    cylPart(context, reg, p(0), p(4), 6.5, "[6] Gamma B458208 carriage bolt M6x60 (head)", steel);
    cylPart(context, reg, p(4), p(64), 3, "[6] Gamma B458208 carriage bolt M6x60", steel);
    cylPart(context, reg, p(T + 6), p(T + 7.5), 6, "[8] Gamma B458373 washer M6", steel);
    cylPart(context, reg, p(T + 7.5), p(T + 14), 5, "[7] Gamma B458307 wing nut M6", steel);
    boxPart(context, reg, add3(p(T + 9), [0, -2, -12]), add3(p(T + 13), [0, 2, 12]), "[7] Gamma B458307 wing nut M6 (wings)", steel);   // bolts all run along x
}

function add3(a is array, b is array) returns array
{
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

// printed U-groove wheel between two printed bracket plates, [6] M6x60 bolt as the axle, [12] lock nut
function wheel(context is Context, reg is box, wx is number, wy is number, wheelZ is number, deckZ is number, printCol is Color, steel is Color)
{
    cylPart(context, reg, [wx, wy - 8, wheelZ], [wx, wy + 8, wheelZ], WHEEL_R, "PRINT | U-groove wheel 50 mm (19 mm groove)", color(0.15, 0.15, 0.17));
    boxPart(context, reg, [wx - 20, wy - 15, wheelZ - 8], [wx + 20, wy - 9, deckZ], "PRINT | wheel bracket plate (2 x [10] 4x30 into the deck)", printCol);
    boxPart(context, reg, [wx - 20, wy + 9, wheelZ - 8], [wx + 20, wy + 15, deckZ], "PRINT | wheel bracket plate (2 x [10] 4x30 into the deck)", printCol);
    cylPart(context, reg, [wx, wy - 19, wheelZ], [wx, wy - 15, wheelZ], 6.5, "[6] Gamma B458208 carriage bolt M6x60 (axle head)", steel);
    cylPart(context, reg, [wx, wy - 15, wheelZ], [wx, wy + 22, wheelZ], 3, "[6] Gamma B458208 carriage bolt M6x60 (axle)", steel);
    cylPart(context, reg, [wx, wy + 15, wheelZ], [wx, wy + 21, wheelZ], 5.5, "[12] Gamma B458291 lock nut M6", steel);
}

// ---------- parts ----------

// a wooden piece that is on the cut list: name, record its length, emboss its code on `face`
function piece(context is Context, reg is box, code is string, what is string, item is string, p is array, q is array, col is Color, face)
{
    const lo = [min(p[0], q[0]), min(p[1], q[1]), min(p[2], q[2])];
    const hi = [max(p[0], q[0]), max(p[1], q[1]), max(p[2], q[2])];
    const d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const len = round(max(d[0], max(d[1], d[2])));
    var r = reg[];
    r.cut[code] = len;
    reg[] = r;
    const pid = nextId(reg);
    fCuboid(context, pid, { "corner1" : v3mm(lo), "corner2" : v3mm(hi) });
    if (face != undefined && reg[].labels)
    {
        const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
        // long axis of the piece: x, y or z
        const ax = d[0] >= d[1] && d[0] >= d[2] ? 0 : (d[1] >= d[2] ? 1 : 2);
        const axes = [vector(1, 0, 0), vector(0, 1, 0), vector(0, 0, 1)];
        var n = vector(0, 0, 1);
        var at = c;
        var w = 0;
        if (face == "+z") { n = vector(0, 0, 1); at = [c[0], c[1], hi[2]]; w = min(d[0], d[1]); }
        else if (face == "-y") { n = vector(0, -1, 0); at = [c[0], lo[1], c[2]]; w = min(d[0], d[2]); }
        else if (face == "+y") { n = vector(0, 1, 0); at = [c[0], hi[1], c[2]]; w = min(d[0], d[2]); }
        else if (face == "-x") { n = vector(-1, 0, 0); at = [lo[0], c[1], c[2]]; w = min(d[1], d[2]); }
        var xd = axes[ax];
        if (face == "+y" && ax == 0)
            xd = vector(-1, 0, 0);                             // reads left to right from outside row B
        if (face == "-x" && ax == 1)
            xd = vector(0, -1, 0);
        emboss(context, reg, pid, code, at, n, xd, min(0.6 * w, 30), len * 0.5);
    }
    finish(context, pid, code ~ " | " ~ what ~ " | cut " ~ len ~ " mm | " ~ item, col);
}

function cylPiece(context is Context, reg is box, code is string, what is string, item is string, p is array, q is array, r is number, col is Color)
{
    const len = round(norm(vector(q[0] - p[0], q[1] - p[1], q[2] - p[2])));
    var rr = reg[];
    rr.cut[code] = len;
    reg[] = rr;
    cylPart(context, reg, p, q, r, code ~ " | " ~ what ~ " | cut " ~ len ~ " mm | " ~ item, col);
}

// ---------- text ----------

// make `txt` as glyph bodies, scaled to height h (mm) but no wider than maxW, placed centred at `at` (mm)
// on a plane with normal n and text direction xd; returns the glyph bodies' query
function makeText(context is Context, reg is box, txt is string, at is array, n is Vector, xd is Vector, h is number, maxW is number, depth is number) returns Query
{
    const sid = nextId(reg);
    const sk = newSketchOnPlane(context, sid, { "sketchPlane" : plane(vector(0, 0, 0) * meter, vector(0, 0, 1), vector(1, 0, 0)) });
    skText(sk, "t", { "text" : txt, "fontName" : FONT, "corner" : vector(0, 0) * millimeter, "secondCorner" : vector(1, 1) * millimeter });
    skSolve(sk);
    const regs = qSketchRegion(sid, true);
    const bb = evBox3d(context, { "topology" : regs });
    const bw = (bb.maxCorner[0] - bb.minCorner[0]) / millimeter;
    const bh = (bb.maxCorner[1] - bb.minCorner[1]) / millimeter;
    const s = min(h / bh, maxW / bw);
    const eid = nextId(reg);
    opExtrude(context, eid, { "entities" : regs, "direction" : vector(0, 0, 1), "endBound" : BoundingType.BLIND, "endDepth" : depth / s * millimeter });
    const cx = (bb.maxCorner[0] + bb.minCorner[0]) / 2;
    const cy = (bb.maxCorner[1] + bb.minCorner[1]) / 2;
    const toFace = toWorld(coordSystem(v3mm(at), xd, n));
    opTransform(context, nextId(reg), { "bodies" : qCreatedBy(eid, EntityType.BODY),
                "transform" : toFace * scaleUniformly(s) * transform(vector(-cx, -cy, 0 * meter)) });
    opDeleteBodies(context, nextId(reg), { "entities" : qCreatedBy(sid, EntityType.BODY) });
    return qCreatedBy(eid, EntityType.BODY);
}

// raised code letters on a part's face, merged into the part; the letter faces are coloured dark first,
// face colours survive the union (the body colour is set afterwards by finish())
function emboss(context is Context, reg is box, pid is Id, txt is string, at is array, n is Vector, xd is Vector, h is number, maxW is number)
{
    const sunk = [at[0] - n[0] * 0.3, at[1] - n[1] * 0.3, at[2] - n[2] * 0.3];
    const g = makeText(context, reg, txt, sunk, n, xd, h, maxW, 1.0);
    setProperty(context, { "entities" : qOwnedByBody(g, EntityType.FACE), "propertyType" : PropertyType.APPEARANCE, "value" : color(0.1, 0.1, 0.12) });
    opBoolean(context, nextId(reg), { "tools" : qUnion([qCreatedBy(pid, EntityType.BODY), g]), "operationType" : BooleanOperationType.UNION });
}

// a label lying on the floor: dark letters on a thin white tag, one part per label
function floorText(context is Context, reg is box, txt is string, leftAt is array, h is number, maxW is number, col is Color)
{
    const g = makeText(context, reg, txt, [0, 0, 0], vector(0, 0, 1), vector(1, 0, 0), h, maxW, 1.0);
    const bb = evBox3d(context, { "topology" : g });
    const w = (bb.maxCorner[0] - bb.minCorner[0]) / millimeter;
    const dx = leftAt[0] + w / 2;
    opTransform(context, nextId(reg), { "bodies" : g, "transform" : transform(v3mm([dx, leftAt[1], leftAt[2]])) });
    setProperty(context, { "entities" : qOwnedByBody(g, EntityType.FACE), "propertyType" : PropertyType.APPEARANCE, "value" : col });
    const pad = 0.25 * h;
    const tag = nextId(reg);
    fCuboid(context, tag, { "corner1" : v3mm([leftAt[0] - pad, leftAt[1] + bb.minCorner[1] / millimeter - pad, leftAt[2] - 0.2]),
                "corner2" : v3mm([leftAt[0] + w + pad, leftAt[1] + bb.maxCorner[1] / millimeter + pad, leftAt[2] + 0.5]) });
    opBoolean(context, nextId(reg), { "tools" : qUnion([qCreatedBy(tag, EntityType.BODY), g]), "operationType" : BooleanOperationType.UNION });
    finish(context, tag, "CUT PLAN | label: " ~ txt, color(0.98, 0.98, 0.96));
}

// ---------- helpers ----------

function nextId(reg is box) returns Id
{
    var r = reg[];
    r.n = r.n + 1;
    reg[] = r;
    return r.base + ("p" ~ r.n);
}

function v3mm(p is array) returns Vector
{
    return vector(p[0], p[1], p[2]) * millimeter;
}

function finish(context is Context, pid is Id, name is string, col is Color)
{
    const q = qCreatedBy(pid, EntityType.BODY);
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.NAME, "value" : name });
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.APPEARANCE, "value" : col });
}

function boxPart(context is Context, reg is box, p is array, q is array, name is string, col is Color)
{
    const pid = nextId(reg);
    fCuboid(context, pid, {
                "corner1" : v3mm([min(p[0], q[0]), min(p[1], q[1]), min(p[2], q[2])]),
                "corner2" : v3mm([max(p[0], q[0]), max(p[1], q[1]), max(p[2], q[2])]) });
    finish(context, pid, name, col);
}

function cylPart(context is Context, reg is box, p is array, q is array, r is number, name is string, col is Color)
{
    const pid = nextId(reg);
    fCylinder(context, pid, { "bottomCenter" : v3mm(p), "topCenter" : v3mm(q), "radius" : r * millimeter });
    finish(context, pid, name, col);
}

function spherePart(context is Context, reg is box, c is array, r is number, name is string, col is Color)
{
    const pid = nextId(reg);
    opSphere(context, pid, { "center" : v3mm(c), "radius" : r * millimeter });
    finish(context, pid, name, col);
}
