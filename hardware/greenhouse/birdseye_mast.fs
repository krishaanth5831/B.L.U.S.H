FeatureScript 3083;
import(path : "onshape/std/geometry.fs", version : "3083.0");

/*
 * Birdseye Camera Mast - a printed pole with indexed height, pan and tilt for the
 * InnoMaker U20CAM-1080P overview camera (lerobot name "birdseye") of the SO-101
 * test greenhouse.
 *
 * Add it after the SO-101 Greenhouse feature in the same Part Studio. It reads the
 * greenhouse's FIT reach envelope and its members, and checks that the mast stays
 * out of the arm's reach and that the camera sees the pick zone through the end frame.
 *
 * Coordinates match the greenhouse: +X along the rows (end frame at x = 0), +Y across,
 * Z = 0 is the table. The mast stands outside the x = 0 end frame, in line with an aisle.
 *
 * Every adjustment locks positively, because a front camera that drifts between
 * recording sessions silently spoils an imitation-learning dataset:
 *   - height: an M4 bolt through the outer tube and one of the slider's holes (10 mm steps)
 *   - pan and tilt: Hirth-style serrated faces (10 degree steps), clamped by a knob bolt
 * Engraved marks let a setting be written down and restored.
 *
 * Camera: InnoMaker U20CAM-1080P - bare 32 x 32 mm board, 4 x M2 holes (2.2 mm) on a
 * 28 mm square, M12 lens (130 deg diagonal), side-entry cable connector on the back
 * at one edge. The board stands off the plate so the back components and connector
 * clear it.
 */

const MM = millimeter;

const B_MAST_X    = { (millimeter) : [-2000, -120, 2000] } as LengthBoundSpec;
const B_AISLE_Y   = { (millimeter) : [-2000, 200, 2000] } as LengthBoundSpec;
const B_RISERS    = { (unitless) : [0, 0, 4] } as IntegerBoundSpec;
const B_EXT       = { (millimeter) : [0, 150, 400] } as LengthBoundSpec;
const B_PAN       = { (degree) : [-180, 0, 180] } as AngleBoundSpec;
const B_TILT      = { (degree) : [-30, 20, 90] } as AngleBoundSpec;
const B_TUBE_W    = { (millimeter) : [26, 34, 50] } as LengthBoundSpec;
const B_WALL      = { (millimeter) : [1.6, 2.4, 4] } as LengthBoundSpec;
const B_SOCK_CLR  = { (millimeter) : [0, 0.15, 0.6] } as LengthBoundSpec;
const B_SLIDE_CLR = { (millimeter) : [0.05, 0.25, 0.8] } as LengthBoundSpec;
const B_TUBE_LEN  = { (millimeter) : [150, 240, 340] } as LengthBoundSpec;
const B_RISER_LEN = { (millimeter) : [60, 160, 340] } as LengthBoundSpec;
const B_TEETH     = { (unitless) : [12, 36, 72] } as IntegerBoundSpec;
const B_BOARD     = { (millimeter) : [20, 32, 50] } as LengthBoundSpec;
const B_HOLE_P    = { (millimeter) : [15, 28, 46] } as LengthBoundSpec;
const B_M2        = { (millimeter) : [1.5, 2.2, 3.5] } as LengthBoundSpec;
const B_GAP       = { (millimeter) : [3, 6.5, 15] } as LengthBoundSpec;
const B_HFOV      = { (degree) : [20, 80, 170] } as AngleBoundSpec;
const B_VFOV      = { (degree) : [20, 60, 170] } as AngleBoundSpec;
const B_ROW_OFF   = { (millimeter) : [0, 200, 1000] } as LengthBoundSpec;
const B_ZONE_X0   = { (millimeter) : [-2000, 300, 2000] } as LengthBoundSpec;
const B_ZONE_X1   = { (millimeter) : [-2000, 420, 2000] } as LengthBoundSpec;
const B_ZONE_Z0   = { (millimeter) : [0, 100, 2000] } as LengthBoundSpec;
const B_ZONE_Z1   = { (millimeter) : [0, 350, 2000] } as LengthBoundSpec;
const B_EDGE_R    = { (millimeter) : [0, 0.6, 2] } as LengthBoundSpec;

annotation { "Feature Type Name" : "Birdseye Camera Mast",
             "Feature Type Description" : "Printed mast with indexed height, pan and tilt for the InnoMaker U20CAM-1080P birdseye camera, checked against the SO-101 greenhouse." }
export const birdseyeCameraMast = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Group Name" : "Placement and setting", "Collapsed By Default" : false }
        {
            annotation { "Name" : "Mast x (greenhouse end frame at x = 0)" }
            isLength(definition.mastX, B_MAST_X);
            annotation { "Name" : "Aisle centre y (camera sits on it at pan 0)" }
            isLength(definition.aisleY, B_AISLE_Y);
            annotation { "Name" : "Risers" }
            isInteger(definition.risers, B_RISERS);
            annotation { "Name" : "Height mark (slider above outer tube, 10 mm steps)" }
            isLength(definition.ext, B_EXT);
            annotation { "Name" : "Pan (snaps to one tooth)" }
            isAngle(definition.pan, B_PAN);
            annotation { "Name" : "Tilt down (snaps to one tooth)" }
            isAngle(definition.tilt, B_TILT);
        }
        annotation { "Group Name" : "Mast", "Collapsed By Default" : true }
        {
            annotation { "Name" : "Tube outer size" }
            isLength(definition.tubeW, B_TUBE_W);
            annotation { "Name" : "Wall" }
            isLength(definition.wall, B_WALL);
            annotation { "Name" : "Socket clearance (per side, base and collars)" }
            isLength(definition.sockClr, B_SOCK_CLR);
            annotation { "Name" : "Slider clearance (per side)" }
            isLength(definition.slideClr, B_SLIDE_CLR);
            annotation { "Name" : "Outer tube and slider length" }
            isLength(definition.tubeLen, B_TUBE_LEN);
            annotation { "Name" : "Riser length" }
            isLength(definition.riserLen, B_RISER_LEN);
        }
        annotation { "Group Name" : "Head and camera", "Collapsed By Default" : true }
        {
            annotation { "Name" : "Serration teeth (360 / teeth = one step)" }
            isInteger(definition.teeth, B_TEETH);
            annotation { "Name" : "Camera board size" }
            isLength(definition.board, B_BOARD);
            annotation { "Name" : "Board hole spacing" }
            isLength(definition.holeP, B_HOLE_P);
            annotation { "Name" : "Board hole diameter (M2)" }
            isLength(definition.m2D, B_M2);
            annotation { "Name" : "Gap behind the board (back parts + connector)" }
            isLength(definition.gap, B_GAP);
        }
        annotation { "Group Name" : "View check", "Collapsed By Default" : true }
        {
            annotation { "Name" : "Usable horizontal field of view" }
            isAngle(definition.hfov, B_HFOV);
            annotation { "Name" : "Usable vertical field of view" }
            isAngle(definition.vfov, B_VFOV);
            annotation { "Name" : "Vine rows either side of the aisle at" }
            isLength(definition.rowOff, B_ROW_OFF);
            annotation { "Name" : "Pick zone x from" }
            isLength(definition.zoneX0, B_ZONE_X0);
            annotation { "Name" : "Pick zone x to" }
            isLength(definition.zoneX1, B_ZONE_X1);
            annotation { "Name" : "Pick zone z from" }
            isLength(definition.zoneZ0, B_ZONE_Z0);
            annotation { "Name" : "Pick zone z to" }
            isLength(definition.zoneZ1, B_ZONE_Z1);
        }
        annotation { "Name" : "Edge rounding radius (0 = off)" }
        isLength(definition.edgeR, B_EDGE_R);
        annotation { "Name" : "Make fit test coupon", "Default" : true }
        definition.makeCoupon is boolean;
        annotation { "Name" : "Show camera, view and checks", "Default" : true }
        definition.showFit is boolean;
    }
    {
        // ------------------------------------------------------------ inputs (mm numbers)
        const W = definition.tubeW / MM;
        const wall = definition.wall / MM;
        const B = W - 2 * wall;                         // outer tube bore
        const slideClr = definition.slideClr / MM;
        const SW = B - 2 * slideClr;                    // slider outer
        const sBore = SW - 2 * wall;
        const SB = W + 2 * definition.sockClr / MM;     // base and collar socket bore
        const sockWall = 3;
        const SO = SB + 2 * sockWall;
        const OL = definition.tubeLen / MM;
        const SL = OL;
        const RL = definition.riserLen / MM;
        const nR = definition.risers;
        const edgeR = definition.edgeR / MM;

        const bR = 2.2;           // M4 clearance hole radius
        const jb = 12;            // joint bolt, from a tube end
        const lb = 40;            // slider lock bolt, below the outer tube top
        const hp = 10;            // height step
        const PT = 5;             // base plate thickness
        const SD = 35;            // base socket depth
        const BP = 110;           // base plate size
        const CL = 60;            // collar length
        const plug = 25;          // solid top of the slider (pan nut lives here)
        const g = 0.3;            // flat faces stand back this far around a serration

        const N = definition.teeth;
        const pStep = 360 / N;
        const rOut = min(14, SW / 2 - 0.35);
        const kT = PI / (2 * N);  // ridge/valley slope: 45 degree flanks

        // head
        const hy = rOut + 1;              // yoke base half width
        const zb = 8;                     // yoke base top
        const kR = 11;                    // knob radius
        const yu0 = kR + 2;               // upright inner face (knob clearance)
        const yJ = yu0 + 10;              // tilt joint plane
        const bd = definition.board / MM;
        const hpc = definition.holeP / MM;
        const m2r = definition.m2D / MM / 2;
        const gap = definition.gap / MM;
        const a = max(rOut + 5, bd / 2 + 3);          // cradle half height = arc radius
        const zA = ceil(sqrt(2) * a + 1);             // tilt axis above the slider top
        const zTop = zA + a;                          // upright top, level with the cradle arc
        const ta = 8;                                 // cradle arm thickness
        const px0 = rOut + 2;                         // camera plate back
        const px1 = px0 + 3;                          // camera plate front
        const ycB = ta + 4 + hpc / 2;                 // board centre, from the tilt joint plane
        const plateY1 = ycB + bd / 2 + 2;
        const camY = yJ + ycB;                        // camera centre beside the mast axis
        const boardX = px1 + gap;                     // board back face
        const opt = boardX + 1.6 + 14;                // optical centre (inside the lens)

        // heights
        const eMin = ceil((jb + bR + 0.5 + SL - OL) / hp) * hp;    // slider clears the outer tube's bottom bolt
        const eMax = floor((SL - lb - 30) / hp) * hp;               // 30 mm of slider below the lock bolt
        const e = min(eMax, max(eMin, round(definition.ext / MM / hp) * hp));
        const pan = round(definition.pan / degree / pStep) * pStep;
        const tilt = round(definition.tilt / degree / pStep) * pStep;
        const zOb = PT + nR * RL;
        const zOt = zOb + OL;
        const zSt = zOt + e;
        const lens0 = PT + OL + zA;                   // lens height at mark 0, tilt 0, no risers

        const mastX = definition.mastX / MM;
        const mastY = definition.aisleY / MM - camY;

        // greenhouse bodies, read before anything of ours exists. Part names aren't readable
        // during a rebuild, so go by shape: the SO-101 reach envelope is a lone sphere face of
        // radius > 200 mm; the other FIT markers (tomatoes, twines, the arm's base) have at most
        // three faces; every real member (rounded tube, node, pin) has many.
        const others = evaluateQuery(context, qBodyType(qEverything(EntityType.BODY), BodyType.SOLID));
        var members = [];
        var sphere = undefined;
        for (var b in others)
        {
            const faces = evaluateQuery(context, qOwnedByBody(b, EntityType.FACE));
            if (size(faces) > 6)
            {
                members = append(members, b);
                continue;
            }
            if (size(faces) == 1)
            {
                var surf = undefined;
                try silent
                {
                    surf = evSurfaceDefinition(context, { "face" : faces[0] });
                }
                if (surf is Sphere && surf.radius > 200 * MM)
                    sphere = b;
            }
        }

        const reg = new box({ "n" : 0, "base" : id });
        var rounding = "";

        // ============================================================ SERRATION MASTER
        const ros = rosetteMaster(context, reg, N, rOut, kT, 3);

        // ============================================================ SLIDER (top joint plane z = 0)
        box3(context, nid(reg, "sA"), [-SW / 2, -SW / 2, -SL], [SW / 2, SW / 2, -g]);
        box3(context, nid(reg, "sC"), [-sBore / 2, -sBore / 2, -SL - 1], [sBore / 2, sBore / 2, -plug]);
        for (var ee = eMin; ee <= eMax + 0.01; ee += hp)
            cylAxis(context, nid(reg, "sC"), [-SW, 0, -(lb + ee)], [SW, 0, -(lb + ee)], bR);
        box3(context, nid(reg, "sC"), [-3.6, -4.4, -11.75], [3.6, SW / 2 + 1, -8.25]);     // pan nut, slides in from +Y
        finishPart(context, reg, "sA", "sC");
        rounding = rounding ~ "slider " ~ roundConvexEdges(context, reg, qKind(reg, "sA"), edgeR) ~ "; ";
        addSerration(context, reg, ros, identityTransform(), "sA", rOut, kT);
        zcyl(context, nid(reg, "sH"), 0, 0, -14, 2, bR);
        cutKind(context, reg, qKind(reg, "sA"), "sH");
        {
            // height marks on +Y: the mark level with the outer tube's top edge is the setting
            const plY = plane(v3(0, SW / 2, 0), vector(0, 1, 0), vector(-1, 0, 0));
            for (var ee = eMin; ee <= eMax + 0.01; ee += hp)
            {
                const big = round(ee) % 50 == 0;
                box3(context, nid(reg, "sE"), [SW / 2 - (big ? 11 : 6), SW / 2 - 0.5, -ee - 0.3], [SW / 2 + 1, SW / 2 + 1, -ee + 0.3]);
                if (big)
                    engraveText(context, nid(reg, "sE"), plY, "" ~ round(ee), 4, -ee, 3 * 2.8, 4.5, 0.5);
            }
            // pan pointers at the four face centres, just under the joint; a dot under the +X one
            for (var f = 0; f < 4; f += 1)
            {
                const pid = nid(reg, "sE");
                box3(context, pid, [SW / 2 - 0.3, -0.4, -7], [SW / 2 + 1, 0.4, -g + 1]);
                opTransform(context, nid(reg, "mv"), { "bodies" : qCreatedBy(pid, EntityType.BODY),
                            "transform" : rotationAround(line(v3(0, 0, 0), vector(0, 0, 1)), f * 90 * degree) });
            }
            cylAxis(context, nid(reg, "sE"), [SW / 2 - 0.5, 0, -9.5], [SW / 2 + 1, 0, -9.5], 0.9);
            cutKind(context, reg, qKind(reg, "sA"), "sE");
        }
        const sliderQ = qKind(reg, "sA");

        // ============================================================ OUTER TUBE (bottom z = 0)
        box3(context, nid(reg, "oA"), [-W / 2, -W / 2, 0], [W / 2, W / 2, OL]);
        box3(context, nid(reg, "oC"), [-B / 2, -B / 2, -1], [B / 2, B / 2, OL + 1]);
        cylAxis(context, nid(reg, "oC"), [-W, 0, jb], [W, 0, jb], bR);
        cylAxis(context, nid(reg, "oC"), [-W, 0, OL - lb], [W, 0, OL - lb], bR);
        finishPart(context, reg, "oA", "oC");
        rounding = rounding ~ "outer " ~ roundConvexEdges(context, reg, qKind(reg, "oA"), edgeR) ~ "; ";
        engraveText(context, nid(reg, "oE"), plane(v3(0, W / 2, 0), vector(0, 1, 0), vector(-1, 0, 0)), "TOP", 0, OL - 16, 14, 5, 0.5);
        cutKind(context, reg, qKind(reg, "oA"), "oE");
        const outerQ = qKind(reg, "oA");

        // ============================================================ RISER (bottom z = 0)
        box3(context, nid(reg, "rA"), [-W / 2, -W / 2, 0], [W / 2, W / 2, RL]);
        box3(context, nid(reg, "rC"), [-B / 2, -B / 2, -1], [B / 2, B / 2, RL + 1]);
        cylAxis(context, nid(reg, "rC"), [-W, 0, jb], [W, 0, jb], bR);
        cylAxis(context, nid(reg, "rC"), [-W, 0, RL - jb], [W, 0, RL - jb], bR);
        finishPart(context, reg, "rA", "rC");
        rounding = rounding ~ "riser " ~ roundConvexEdges(context, reg, qKind(reg, "rA"), edgeR) ~ "; ";
        const riserQ = qKind(reg, "rA");

        // ============================================================ COLLAR (bottom z = 0, tubes meet at CL / 2)
        box3(context, nid(reg, "cA"), [-SO / 2, -SO / 2, 0], [SO / 2, SO / 2, CL]);
        box3(context, nid(reg, "cC"), [-SB / 2, -SB / 2, -1], [SB / 2, SB / 2, CL + 1]);
        for (var zz in [CL / 2 - jb, CL / 2 + jb])
            cylAxis(context, nid(reg, "cC"), [-SO, 0, zz], [SO, 0, zz], bR);
        finishPart(context, reg, "cA", "cC");
        rounding = rounding ~ "collar " ~ roundConvexEdges(context, reg, qKind(reg, "cA"), edgeR) ~ "; ";
        const collarQ = qKind(reg, "cA");

        // ============================================================ BASE (table z = 0)
        prismPlaneR(context, nid(reg, "bA"), plane(v3(0, 0, 0), vector(0, 0, 1), vector(1, 0, 0)), rrect(-BP / 2, -BP / 2, BP / 2, BP / 2), 12, PT);
        box3(context, nid(reg, "bA"), [-SO / 2, -SO / 2, 0], [SO / 2, SO / 2, PT + SD]);
        for (var q4 = 0; q4 < 4; q4 += 1)
        {
            // gussets on the socket's diagonals (clear of the bolt on X and the screws on the axes)
            const ang = (45 + 90 * q4) * degree;
            const dir = vector(cos(ang), sin(ang), 0);
            const nrm = vector(-sin(ang), cos(ang), 0);
            const c0 = SO / 2 * sqrt(2) - 4;
            prismPlane(context, nid(reg, "bA"), plane(nrm * (-2) * MM, nrm, dir),
                [[c0, -(PT - 1)], [c0, -(PT + SD - 6)], [c0 + 26, -(PT - 1)]], 4);
        }
        box3(context, nid(reg, "bC"), [-SB / 2, -SB / 2, PT], [SB / 2, SB / 2, PT + SD + 1]);
        cylAxis(context, nid(reg, "bC"), [-SO, 0, PT + jb], [SO, 0, PT + jb], bR);
        const scr = BP / 2 - 9;
        for (var s in [[scr, 0], [-scr, 0], [0, scr], [0, -scr]])
        {
            // countersunk holes for 4 mm wood screws
            zcyl(context, nid(reg, "bC"), s[0], s[1], -1, PT + 1, 2.25);
            fCone(context, nid(reg, "bC"), { "bottomCenter" : v3(s[0], s[1], PT - 2.45), "topCenter" : v3(s[0], s[1], PT + 0.01),
                        "bottomRadius" : 2.2 * MM, "topRadius" : 4.65 * MM });
        }
        finishPart(context, reg, "bA", "bC");
        rounding = rounding ~ "base " ~ roundConvexEdges(context, reg, qKind(reg, "bA"), edgeR) ~ "; ";
        {
            // "ARM ->" on the plate: the camera faces this way at pan 0
            const plZ = plane(v3(0, 0, PT), vector(0, 0, 1), vector(1, 0, 0));
            engraveText(context, nid(reg, "bE"), plZ, "ARM", -4, -31, 15, 6, 0.6);
            prismPlane(context, nid(reg, "bE"), plane(v3(0, 0, PT - 0.6), vector(0, 0, 1), vector(1, 0, 0)), [[7, -34.5], [14, -31], [7, -27.5]], 1);
            cutKind(context, reg, qKind(reg, "bA"), "bE");
        }
        const baseQ = qKind(reg, "bA");

        // ============================================================ YOKE (pan joint plane z = 0, camera side +Y)
        box3(context, nid(reg, "yA"), [-hy, -hy, g], [hy, yJ - g, zb]);
        box3(context, nid(reg, "yA"), [-hy, yu0, g], [hy, yJ - g, zTop]);
        const ySlot = yu0 + 4.5;
        box3(context, nid(reg, "yC"), [-4.4, ySlot - 1.75, zA - 3.6], [hy + 1, ySlot + 1.75, zA + 3.6]);   // tilt nut, slides in from +X
        finishPart(context, reg, "yA", "yC");
        rounding = rounding ~ "yoke " ~ roundConvexEdges(context, reg, qKind(reg, "yA"), edgeR) ~ "; ";
        const rotX = function(d) { return rotationAround(line(v3(0, 0, 0), vector(1, 0, 0)), d * degree); };
        const rotZ = function(d) { return rotationAround(line(v3(0, 0, 0), vector(0, 0, 1)), d * degree); };
        addSerration(context, reg, ros, rotX(180) * rotZ(pStep / 2), "yA", rOut, kT);              // pan, teeth down
        addSerration(context, reg, ros, transform(v3(0, yJ, zA)) * rotX(-90), "yA", rOut, kT);   // tilt, teeth +Y
        zcyl(context, nid(reg, "yH"), 0, 0, -2, zb + 1, bR);
        cylAxis(context, nid(reg, "yH"), [0, yu0 - 1, zA], [0, yJ + 2, zA], bR);
        cutKind(context, reg, qKind(reg, "yA"), "yH");
        {
            // pan scale: a tick every tooth around the base, long every 30 degrees
            const yMax = yJ - g;
            for (var j = 0; j < N; j += 1)
            {
                const phi = j * pStep;
                const c = cos(phi * degree);
                const s = sin(phi * degree);
                var t = 1e9;
                var face = 0;
                if (c > 1e-6 && hy / c < t) { t = hy / c; face = 0; }
                if (c < -1e-6 && -hy / c < t) { t = -hy / c; face = 0; }
                if (s > 1e-6 && yMax / s < t) { t = yMax / s; face = 1; }
                if (s < -1e-6 && -hy / s < t) { t = -hy / s; face = 1; }
                const px = t * c;
                const py = t * s;
                if (face == 0 && (py < -hy + 1.5 || py > yMax - 1.5))
                    continue;
                if (face == 1 && abs(px) > hy - 1.5)
                    continue;
                const z0 = (round(phi) % 30 == 0) ? 1.5 : 4;
                if (face == 0)
                    box3(context, nid(reg, "yE"), [px - 0.5, py - 0.3, z0], [px + 0.5, py + 0.3, zb - 0.8]);
                else
                    box3(context, nid(reg, "yE"), [px - 0.3, py - 0.5, z0], [px + 0.3, py + 0.5, zb - 0.8]);
            }
            // tilt pointer on the upright's top, next to the cradle's arc scale
            box3(context, nid(reg, "yE"), [-0.3, yu0 + 2, zTop - 0.5], [0.3, yJ + 1, zTop + 1]);
            cutKind(context, reg, qKind(reg, "yA"), "yE");
        }
        const yokeQ = qKind(reg, "yA");

        // ============================================================ CRADLE (tilt axis = local Y through the origin, joint plane y = 0)
        armProfile(context, nid(reg, "kA"), a, px1, g, ta);
        box3(context, nid(reg, "kA"), [px0, g, -a], [px1, plateY1, a]);
        const holes = [[ycB - hpc / 2, -hpc / 2], [ycB - hpc / 2, hpc / 2], [ycB + hpc / 2, -hpc / 2], [ycB + hpc / 2, hpc / 2]];
        for (var h in holes)
            cylAxis(context, nid(reg, "kA"), [px1 - 0.5, h[0], h[1]], [boardX, h[0], h[1]], 2.4);
        box3(context, nid(reg, "kC"), [px0 - 1, ycB - 9, -a - 1], [px1 + 1, ycB + 9, 10]);              // window, open at the bottom
        for (var h in holes)
        {
            cylAxis(context, nid(reg, "kC"), [px0 - 1, h[0], h[1]], [boardX + 1, h[0], h[1]], m2r);
            prismPlane(context, nid(reg, "kC"), plane(v3(px0 - 0.01, 0, 0), vector(1, 0, 0), vector(0, 1, 0)), hexPts(h[0], h[1], 4.3), 1.8);
        }
        finishPart(context, reg, "kA", "kC");
        rounding = rounding ~ "cradle " ~ roundConvexEdges(context, reg, qKind(reg, "kA"), edgeR) ~ "; ";
        addSerration(context, reg, ros, rotX(90) * rotZ(pStep / 2), "kA", rOut, kT);   // teeth -Y
        cylAxis(context, nid(reg, "kH"), [0, -2, 0], [0, ta + 1, 0], bR);
        cutKind(context, reg, qKind(reg, "kA"), "kH");
        {
            // tilt scale on the arc: the tick under the upright's pointer reads the tilt down
            for (var bt = 0; bt <= 90.01; bt += pStep)
            {
                const phi = (90 + bt) * degree;
                const u = [cos(phi), sin(phi)];
                const tg = [-sin(phi), cos(phi)];
                const L = (round(bt) % 30 == 0) ? 5 : 3;
                var pts = [];
                for (var c in [[a - 0.5, -0.3], [a + 1, -0.3], [a + 1, 0.3], [a - 0.5, 0.3]])
                    pts = append(pts, [c[0] * u[0] + c[1] * tg[0], c[0] * u[1] + c[1] * tg[1]]);
                prismXZ(context, nid(reg, "kE"), pts, g - 0.1, L);
            }
            // tilting up: ticks on the flat top
            for (var bt = pStep; bt <= 30.01; bt += pStep)
            {
                const x = a * tan(bt * degree);
                box3(context, nid(reg, "kE"), [x - 0.3, g - 0.1, a - 0.5], [x + 0.3, g + 3, a + 1]);
            }
            cutKind(context, reg, qKind(reg, "kA"), "kE");
        }
        const cradleQ = qKind(reg, "kA");
        const cb = evBox3d(context, { "topology" : cradleQ, "tight" : true });      // cradle extent, its own frame

        // ============================================================ KNOB (bearing face z = 0; hex pocket on top for an M4 nut or hex bolt head)
        zcyl(context, nid(reg, "nA"), 0, 0, 0, 8, kR);
        for (var j = 0; j < 8; j += 1)
            zcyl(context, nid(reg, "nC"), (kR + 1.2) * cos(j * 45 * degree), (kR + 1.2) * sin(j * 45 * degree), -1, 9, 2.6);
        zcyl(context, nid(reg, "nC"), 0, 0, -1, 9, bR);
        prismPlane(context, nid(reg, "nC"), plane(v3(0, 0, 8 - 3.4), vector(0, 0, 1), vector(1, 0, 0)), hexPts(0, 0, 7.15), 4);
        finishPart(context, reg, "nA", "nC");
        rounding = rounding ~ "knob " ~ roundConvexEdges(context, reg, qKind(reg, "nA"), edgeR);
        const knobQ = qKind(reg, "nA");

        opDeleteBodies(context, nid(reg, "tmp"), { "entities" : ros });

        // ============================================================ ASSEMBLY
        const G = transform(v3(mastX, mastY, 0));
        const Ts = G * transform(v3(0, 0, zSt));
        const Ty = Ts * rotZ(pan);
        const Tc = Ty * transform(v3(0, yJ, zA)) * rotationAround(line(v3(0, 0, 0), vector(0, 1, 0)), tilt * degree);
        const toX = rotationAround(line(v3(0, 0, 0), vector(0, 1, 0)), 90 * degree);    // knob axis +X
        const asmCol = color(0.35, 0.45, 0.75);
        placeCopies(context, reg, baseQ, [G], ["base"], "cam base", asmCol);
        placeCopies(context, reg, outerQ, [G * transform(v3(0, 0, zOb))], ["outer"], "cam outer tube", asmCol);
        placeCopies(context, reg, sliderQ, [Ts], ["slider"], "cam slider", asmCol);
        placeCopies(context, reg, yokeQ, [Ty], ["yoke"], "cam yoke", asmCol);
        placeCopies(context, reg, cradleQ, [Tc], ["cradle"], "cam cradle", asmCol);
        var knobT = [Ty * transform(v3(0, 0, zb)), Tc * transform(v3(0, ta, 0)) * rotX(-90),
                     G * transform(v3(W / 2, 0, zOt - lb)) * toX, G * transform(v3(SO / 2, 0, PT + jb)) * toX];
        if (nR > 0)
        {
            var rT = [];
            var cT = [];
            var rN = [];
            var cN = [];
            for (var i = 0; i < nR; i += 1)
            {
                const zc = PT + (i + 1) * RL;
                rT = append(rT, G * transform(v3(0, 0, PT + i * RL)));
                cT = append(cT, G * transform(v3(0, 0, zc - CL / 2)));
                rN = append(rN, "riser" ~ i);
                cN = append(cN, "collar" ~ i);
                knobT = append(knobT, G * transform(v3(SO / 2, 0, zc - jb)) * toX);
                knobT = append(knobT, G * transform(v3(SO / 2, 0, zc + jb)) * toX);
            }
            placeCopies(context, reg, riserQ, rT, rN, "cam riser", asmCol);
            placeCopies(context, reg, collarQ, cT, cN, "cam collar", asmCol);
        }
        var knobN = [];
        for (var i = 0; i < size(knobT); i += 1)
            knobN = append(knobN, "knob" ~ i);
        placeCopies(context, reg, knobQ, knobT, knobN, "cam knob", color(0.95, 0.75, 0.2));
        const nKnobs = size(knobT);

        // ============================================================ PRINT LAYOUT (left of the greenhouse, on z = 0)
        const pla = color(0.25, 0.4, 0.85);
        const lyingT = function(len, w) { return toWorld(coordSystem(v3(0, 0, w / 2), vector(0, 1, 0), vector(1, 0, 0))); };
        printAt(context, reg, baseQ, transform(v3(-700, 0, 0)), "PRINT - cam base (x1)", pla);
        printAt(context, reg, collarQ, transform(v3(-590, 0, 0)), "PRINT - cam collar (x1 per riser)", pla);
        printAt(context, reg, yokeQ, transform(v3(-520, 0, hy)) * rotationAround(line(v3(0, 0, 0), vector(0, 1, 0)), -90 * degree), "PRINT - cam yoke (x1)", pla);
        printAt(context, reg, cradleQ, transform(v3(-460, -10, a)), "PRINT - cam cradle (x1)", pla);
        printAt(context, reg, knobQ, transform(v3(-380, 0, 0)), "PRINT - cam knob (x4 + 2 per riser)", color(0.95, 0.75, 0.2));
        printAt(context, reg, outerQ, transform(v3(-1000, -90, 0)) * lyingT(OL, W), "PRINT - cam outer tube (x1)", pla);
        printAt(context, reg, sliderQ, transform(v3(-1000 + SL, -140, 0)) * toWorld(coordSystem(v3(0, 0, SW / 2), vector(0, 1, 0), vector(1, 0, 0))), "PRINT - cam slider (x1)", pla);
        printAt(context, reg, riserQ, transform(v3(-1000, -190, 0)) * lyingT(RL, W), "PRINT - cam riser (x1 per riser)", pla);

        // ============================================================ COUPON
        if (definition.makeCoupon)
        {
            const cx = -1000;
            const cy = -300;
            // the board's hole pattern: lay the camera on it before printing the cradle
            box3(context, nid(reg, "qA"), [cx, cy, 0], [cx + bd + 6, cy + bd + 6, 2]);
            for (var h in [[-1, -1], [-1, 1], [1, -1], [1, 1]])
                zcyl(context, nid(reg, "qC"), cx + bd / 2 + 3 + h[0] * hpc / 2, cy + bd / 2 + 3 + h[1] * hpc / 2, -1, 3, m2r);
            finishPart(context, reg, "qA", "qC");
            roundConvexEdges(context, reg, qKind(reg, "qA"), min(edgeR, 0.4));
            nameAs(context, qKind(reg, "qA"), "PRINT - cam coupon: board hole template", color(0.2, 0.55, 0.9));
            // a 20 mm length of outer tube and three slider stubs, all lying the way the real tubes print
            box3(context, nid(reg, "q2A"), [cx + 50, cy, 0], [cx + 70, cy + W, W]);
            box3(context, nid(reg, "q2C"), [cx + 49, cy + wall, wall], [cx + 71, cy + W - wall, W - wall]);
            finishPart(context, reg, "q2A", "q2C");
            roundConvexEdges(context, reg, qKind(reg, "q2A"), min(edgeR, 0.4));
            nameAs(context, qKind(reg, "q2A"), "PRINT - cam coupon: tube ring", color(0.2, 0.55, 0.9));
            const dc = [-0.1, 0, 0.1];
            for (var i = 0; i < 3; i += 1)
            {
                const kA = "q3A" ~ i;
                const kC = "q3C" ~ i;
                const sw = B - 2 * (slideClr + dc[i]);
                const x0 = cx + 90 + i * (sw + 8);
                box3(context, nid(reg, kA), [x0, cy, 0], [x0 + 20, cy + sw, sw]);
                box3(context, nid(reg, kC), [x0 - 1, cy + wall, wall], [x0 + 21, cy + sw - wall, sw - wall]);
                finishPart(context, reg, kA, kC);
                roundConvexEdges(context, reg, qKind(reg, kA), min(edgeR, 0.4));
                for (var j = 0; j <= i; j += 1)
                    zcyl(context, nid(reg, "q3E" ~ i), x0 + 5 + j * 4, cy + sw / 2, sw - 0.5, sw + 1, 0.9);
                cutKind(context, reg, qKind(reg, kA), "q3E" ~ i);
                nameAs(context, qKind(reg, kA), "PRINT - cam coupon: slider stub " ~ (i + 1) ~ " dot" ~ (i > 0 ? "s" : ""), color(0.2, 0.55, 0.9));
            }
            // three short sockets standing up the way the base socket and the collar print (bores
            // printed as XY holes come out small); push the tube ring into each
            const dsk = [-0.1, 0, 0.1];
            for (var i = 0; i < 3; i += 1)
            {
                const kA = "q4A" ~ i;
                const kC = "q4C" ~ i;
                const sb = SB + 2 * dsk[i];
                const x0 = cx + i * (sb + 2 * sockWall + 6);
                const y0 = cy + 50;
                box3(context, nid(reg, kA), [x0, y0, 0], [x0 + sb + 2 * sockWall, y0 + sb + 2 * sockWall, 15]);
                box3(context, nid(reg, kC), [x0 + sockWall, y0 + sockWall, -1], [x0 + sockWall + sb, y0 + sockWall + sb, 16]);
                finishPart(context, reg, kA, kC);
                roundConvexEdges(context, reg, qKind(reg, kA), min(edgeR, 0.4));
                for (var j = 0; j <= i; j += 1)
                    zcyl(context, nid(reg, "q4E" ~ i), x0 + 8 + j * 4, y0 + sockWall / 2, 14.5, 16, 0.8);
                cutKind(context, reg, qKind(reg, kA), "q4E" ~ i);
                nameAs(context, qKind(reg, kA), "PRINT - cam coupon: socket ring " ~ (i + 1) ~ " dot" ~ (i > 0 ? "s" : ""), color(0.2, 0.55, 0.9));
            }
        }

        // ============================================================ FIT: camera, view, reach
        var viewMsg = "";
        var reachMsg = "";
        if (definition.showFit)
        {
            // camera dummy in the cradle frame: board, back parts + connector, lens holder, lens
            box3(context, nid(reg, "fCam"), [boardX, ycB - bd / 2, -bd / 2], [boardX + 1.6, ycB + bd / 2, bd / 2]);
            box3(context, nid(reg, "fCam"), [px1 + 0.3, ycB - 6, -bd / 2 - 1.5], [boardX, ycB + 6, -bd / 2 + 6]);
            box3(context, nid(reg, "fCam"), [boardX + 1.6, ycB - 7.5, -7.5], [boardX + 11.6, ycB + 7.5, 7.5]);
            cylAxis(context, nid(reg, "fCam"), [boardX + 11.6, ycB, 0], [boardX + 24, ycB, 0], 7);
            const camQ = qKind(reg, "fCam");
            const camBox = evBox3d(context, { "topology" : camQ, "tight" : true });
            opTransform(context, nid(reg, "mv"), { "bodies" : camQ, "transform" : Tc });
            nameAs(context, camQ, "FIT - U20CAM-1080P", color(0.1, 0.1, 0.1));

            // optical axes
            const O = Tc * v3(opt, ycB, 0);
            const dv = normalize(Tc * v3(opt + 10, ycB, 0) - O);
            const upv = normalize(Tc * v3(opt, ycB, 10) - O);
            const rt = cross(dv, upv);
            const hh = definition.hfov / 2;
            const vh = definition.vfov / 2;

            // view frustum, 700 mm deep
            var profs = [];
            var skIds = [];
            for (var dist in [4, 700])
            {
                const sid = nid(reg, "fsk");
                skIds = append(skIds, qCreatedBy(sid, EntityType.BODY));
                const pl = plane(O + dv * dist * MM, dv, rt);
                const sk = newSketchOnPlane(context, sid, { "sketchPlane" : pl });
                const hu = dist * tan(hh);
                const hv = dist * tan(vh);
                sketchRounded(sk, [[-hu, -hv], [hu, -hv], [hu, hv], [-hu, hv]], 0);
                skSolve(sk);
                profs = append(profs, qSketchRegion(sid));
            }
            const lid = nid(reg, "fView");
            opLoft(context, lid, { "profileSubqueries" : profs });
            opDeleteBodies(context, nid(reg, "tmp"), { "entities" : qUnion(skIds) });
            nameAs(context, qKind(reg, "fView"), "FIT - camera view (usable field)", color(0.3, 0.6, 1.0));
            try silent
            {
                setProperty(context, { "entities" : qKind(reg, "fView"), "propertyType" : PropertyType.APPEARANCE, "value" : color(0.3, 0.6, 1.0, 0.15) });
            }

            // pick zone on both vine rows of this aisle, plus the arm's base
            const ay = definition.aisleY / MM;
            const ro = definition.rowOff / MM;
            const zx = [definition.zoneX0 / MM, definition.zoneX1 / MM];
            const zz = [definition.zoneZ0 / MM, definition.zoneZ1 / MM];
            var targets = [];
            for (var y in [ay - ro, ay + ro])
            {
                box3(context, nid(reg, "fZone"), [zx[0], y - 4, zz[0]], [zx[1], y + 4, zz[1]]);
                for (var x in zx)
                    for (var z in zz)
                        targets = append(targets, [x, y, z]);
            }
            nameAs(context, qKind(reg, "fZone"), "FIT - pick zone (vine rows)", color(0.9, 0.3, 0.3));
            if (sphere != undefined)
            {
                const sb = evBox3d(context, { "topology" : sphere, "tight" : true });
                const sc = (sb.minCorner + sb.maxCorner) / 2 / MM;
                targets = append(targets, [sc[0], sc[1], 30]);
            }
            var inFrame = 0;
            var blocked = 0;
            var worst = -1000 * degree;
            var outList = "";
            for (var tg in targets)
            {
                const P = v3(tg[0], tg[1], tg[2]);
                const rel = P - O;
                const fwd = dot(rel, dv) / MM;
                const ah = atan2(dot(rel, rt) / MM, fwd);
                const av = atan2(dot(rel, upv) / MM, fwd);
                if (fwd > 0 && abs(ah) <= hh && abs(av) <= vh)
                    inFrame += 1;
                else
                    outList = outList ~ " (" ~ fmt1(tg[0]) ~ "," ~ fmt1(tg[1]) ~ "," ~ fmt1(tg[2]) ~ ")";
                worst = max(worst, max(abs(ah) - hh, abs(av) - vh));
                if (size(members) > 0)
                {
                    // line of sight, stopping 5 mm short of the target
                    const o = O / MM;
                    const L = norm(rel) / MM;
                    const endp = o + (P / MM - o) * ((L - 5) / L);
                    const rid = nid(reg, "fRay");
                    cylAxis(context, rid, [o[0], o[1], o[2]], [endp[0], endp[1], endp[2]], 0.5);
                    const dd = evDistance(context, { "side0" : qCreatedBy(rid, EntityType.BODY), "side1" : qUnion(members) }).distance;
                    if (dd < 0.01 * MM)
                        blocked += 1;
                    opDeleteBodies(context, nid(reg, "tmp"), { "entities" : qCreatedBy(rid, EntityType.BODY) });
                }
            }
            viewMsg = "view: " ~ inFrame ~ "/" ~ size(targets) ~ " pick-zone points in the usable field (worst margin " ~ fmt1(-worst / degree) ~ " deg)"
                    ~ (size(members) > 0 ? ", " ~ blocked ~ " blocked by the greenhouse" : ", no greenhouse found") ~ outList;

            // reach: the assembled mast now, and the head swept over every height, pan and tilt
            if (sphere != undefined)
            {
                const sb = evBox3d(context, { "topology" : sphere, "tight" : true });
                const sc = (sb.minCorner + sb.maxCorner) / 2 / MM;
                const sr = (sb.maxCorner[0] - sb.minCorner[0]) / 2 / MM;
                const dNow = evDistance(context, { "side0" : sphere, "side1" : qKind(reg, "asm") }).distance / MM;
                // head points relative to the tilt axis (cradle box incl. camera), then about the pan axis
                var rMax = 0;
                for (var xx in [min(cb.minCorner[0], camBox.minCorner[0]), max(cb.maxCorner[0], camBox.maxCorner[0])])
                    for (var zq in [cb.minCorner[2], cb.maxCorner[2]])
                        rMax = max(rMax, norm(vector(xx / MM, zq / MM)));
                const yMaxH = yJ + max(cb.maxCorner[1], camBox.maxCorner[1]) / MM + 19;   // + tilt knob
                const rHead = sqrt(rMax * rMax + yMaxH * yMaxH);
                const zLo = PT + OL + eMin + zA - rMax;
                const zHi = PT + nR * RL + OL + eMax + zA + rMax;
                const dh = norm(vector(sc[0] - mastX, sc[1] - mastY)) - rHead;
                const dz = sc[2] < zLo ? zLo - sc[2] : (sc[2] > zHi ? sc[2] - zHi : 0);
                const dSweep = sqrt(max(dh, 0) * max(dh, 0) + dz * dz) - sr;
                reachMsg = "reach: mast " ~ fmt1(dNow) ~ " mm outside the SO-101 envelope as set, head swept over every setting >= " ~ fmt1(dSweep) ~ " mm";
            }
            else
                reachMsg = "reach: no SO-101 envelope found (turn on the greenhouse's fit check)";
        }

        // ------------------------------------------------------------ summary
        const summary = "lens " ~ fmt1(zSt + zA) ~ " mm up at mark " ~ e ~ ", pan " ~ pan ~ ", tilt " ~ tilt ~ " down"
                    ~ " | marks " ~ eMin ~ "-" ~ eMax ~ " give lens " ~ fmt1(lens0 + eMin) ~ "-" ~ fmt1(lens0 + eMax) ~ " mm, + " ~ fmt1(RL) ~ " per riser"
                    ~ " | " ~ viewMsg ~ " | " ~ reachMsg ~ " | knobs " ~ nKnobs ~ " | rounded: " ~ rounding
                    ~ " | " ~ size(members) ~ " greenhouse members seen";
        reportFeatureInfo(context, id, summary);
        setVariable(context, "birdseyeReport", summary);
    });

// ================================================================ serrations

// Hirth-style face serration in a local frame: joint plane z = 0, teeth up (+Z), solid
// down to z = -below. Every flank is a plane through the apex on the axis, so two copies
// (one flipped and turned half a tooth) mesh at every radius. Ridges are flattened by
// 20 % of the tooth height so they bottom out before the valleys do.
function rosetteMaster(context is Context, reg is box, N is number, rOut is number, k is number, below is number) returns Query
{
    const d = (180 / N) * degree;
    const s1 = nid(reg, "rsk");
    toothProfile(context, s1, 1.0, k, d, 0.2);
    const s2 = nid(reg, "rsk");
    toothProfile(context, s2, rOut, k, d, 0.2);
    const lid = nid(reg, "rosM");
    opLoft(context, lid, { "profileSubqueries" : [qSketchRegion(s1), qSketchRegion(s2)] });
    opDeleteBodies(context, nid(reg, "tmp"), { "entities" : qUnion([qCreatedBy(s1, EntityType.BODY), qCreatedBy(s2, EntityType.BODY)]) });
    var tfs = [];
    var nms = [];
    for (var j = 1; j < N; j += 1)
    {
        tfs = append(tfs, rotationAround(line(v3(0, 0, 0), vector(0, 0, 1)), j * 360 / N * degree));
        nms = append(nms, "t" ~ j);
    }
    opPattern(context, nid(reg, "rosM"), { "entities" : qCreatedBy(lid, EntityType.BODY), "transforms" : tfs, "instanceNames" : nms });
    // under the valleys: a cone (valleys fall 2k per mm of radius toward the rim) on a disc
    fCone(context, nid(reg, "rosM"), { "bottomCenter" : v3(0, 0, -k * rOut), "topCenter" : v3(0, 0, -k * 1.0),
                "bottomRadius" : rOut * MM, "topRadius" : 1.0 * MM });
    zcyl(context, nid(reg, "rosM"), 0, 0, -below, -k * rOut, rOut);
    const q = qKind(reg, "rosM");
    opBoolean(context, nid(reg, "ops"), { "tools" : q, "operationType" : BooleanOperationType.UNION });
    return q;
}

// one tooth's cross-section at radius r: valley, flattened ridge, valley - a plane scaled about the apex
function toothProfile(context is Context, sid is Id, r is number, k is number, d is ValueWithUnits, tau is number)
{
    const vA = vector(r * cos(d), -r * sin(d), -k * r);
    const vB = vector(r * cos(d), r * sin(d), -k * r);
    const rg = vector(r, 0, k * r);
    const tA = vA + (rg - vA) * (1 - tau);
    const tB = vB + (rg - vB) * (1 - tau);
    const xd = normalize(vB - vA);
    const nrm = normalize(cross(vB - vA, rg - vA));
    const yd = cross(nrm, xd);
    const o = (vA + vB + tA + tB) / 4;
    var pts = [];
    for (var p in [vA, tA, tB, vB])
        pts = append(pts, [dot(p - o, xd), dot(p - o, yd)]);
    const sk = newSketchOnPlane(context, sid, { "sketchPlane" : plane(o * MM, nrm, xd) });
    sketchRounded(sk, pts, 0);
    skSolve(sk);
}

// Cut a pocket for a serration into the part registered as addKind, then add a copy of the master.
// tf takes the master's frame (joint plane z = 0, teeth +Z) into the part's frame.
function addSerration(context is Context, reg is box, master is Query, tf is Transform, addKind is string, rOut is number, k is number)
{
    const pk = addKind ~ "_pk";
    const cid = nid(reg, pk);
    zcyl(context, cid, 0, 0, -k * rOut - 0.05, 5, rOut);
    opTransform(context, nid(reg, "mv"), { "bodies" : qCreatedBy(cid, EntityType.BODY), "transform" : tf });
    cutKind(context, reg, qKind(reg, addKind), pk);
    opPattern(context, nid(reg, addKind), { "entities" : master, "transforms" : [tf], "instanceNames" : ["ser"] });
    opBoolean(context, nid(reg, "ops"), { "tools" : qKind(reg, addKind), "operationType" : BooleanOperationType.UNION });
}

// ================================================================ part helpers

function armProfile(context is Context, pid is Id, a is number, px1 is number, y0 is number, ta is number)
{
    // cradle arm in the XZ plane (sketch v = -z): square except a quarter arc about the
    // tilt axis at the upper back, which carries the tilt scale
    const skId = pid + "sk";
    const sk = newSketchOnPlane(context, skId, { "sketchPlane" : plane(v3(0, y0, 0), vector(0, 1, 0), vector(1, 0, 0)) });
    const c = a * sqrt(0.5);
    skLineSegment(sk, "l1", { "start" : vector(-a, a) * MM, "end" : vector(px1, a) * MM });
    skLineSegment(sk, "l2", { "start" : vector(px1, a) * MM, "end" : vector(px1, -a) * MM });
    skLineSegment(sk, "l3", { "start" : vector(px1, -a) * MM, "end" : vector(0, -a) * MM });
    skArc(sk, "a1", { "start" : vector(0, -a) * MM, "mid" : vector(-c, -c) * MM, "end" : vector(-a, 0) * MM });
    skLineSegment(sk, "l4", { "start" : vector(-a, 0) * MM, "end" : vector(-a, a) * MM });
    skSolve(sk);
    opExtrude(context, pid + "ex", {
                "entities" : qSketchRegion(skId),
                "direction" : vector(0, 1, 0),
                "endBound" : BoundingType.BLIND,
                "endDepth" : (ta - y0) * MM });
    opDeleteBodies(context, pid + "del", { "entities" : qCreatedBy(skId, EntityType.BODY) });
}

// prism from an (x, z) outline, extruded along +Y from y0
function prismXZ(context is Context, pid is Id, ptsXZ is array, y0 is number, depth is number)
{
    var pts = [];
    for (var p in ptsXZ)
        pts = append(pts, [p[0], -p[1]]);
    prismPlane(context, pid, plane(v3(0, y0, 0), vector(0, 1, 0), vector(1, 0, 0)), pts, depth);
}

function hexPts(cu is number, cv is number, af is number) returns array
{
    const rr = af / sqrt(3);
    var pts = [];
    for (var i = 0; i < 6; i += 1)
        pts = append(pts, [cu + rr * cos((30 + 60 * i) * degree), cv + rr * sin((30 + 60 * i) * degree)]);
    return pts;
}

function printAt(context is Context, reg is box, part is Query, tf is Transform, name is string, col is Color)
{
    opTransform(context, nid(reg, "lay"), { "bodies" : part, "transform" : tf });
    nameAs(context, part, name, col);
}

function engraveText(context is Context, pid is Id, pl is Plane, txt is string, cu is number, cv is number, w is number, h is number, depth is number)
{
    const skId = pid + "sk";
    const lifted = plane(pl.origin + pl.normal * 0.2 * MM, pl.normal, pl.x);
    const sk = newSketchOnPlane(context, skId, { "sketchPlane" : lifted });
    skText(sk, "txt", { "text" : txt, "fontName" : "OpenSans-Bold.ttf",
                "firstCorner" : vector(cu - w / 2, cv - h / 2) * MM, "secondCorner" : vector(cu + w / 2, cv + h / 2) * MM });
    skSolve(sk);
    opExtrude(context, pid + "ex", {
                "entities" : qSketchRegion(skId, true),
                "direction" : -pl.normal,
                "endBound" : BoundingType.BLIND,
                "endDepth" : (depth + 0.2) * MM });
    opDeleteBodies(context, pid + "del", { "entities" : qCreatedBy(skId, EntityType.BODY) });
}

function placeCopies(context is Context, reg is box, part is Query, tfs is array, names is array, label is string, col is Color)
{
    const pid = nid(reg, "asm");
    opPattern(context, pid, { "entities" : part, "transforms" : tfs, "instanceNames" : names });
    nameAs(context, qCreatedBy(pid, EntityType.BODY), label, col);
}

function nameAs(context is Context, q is Query, name is string, col is Color)
{
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.NAME, "value" : name });
    setProperty(context, { "entities" : q, "propertyType" : PropertyType.APPEARANCE, "value" : col });
}

// ================================================================ geometry helpers (shared with the greenhouse)

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
