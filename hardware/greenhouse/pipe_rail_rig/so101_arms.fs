FeatureScript 3083;
import(path : "onshape/std/geometry.fs", version : "3083.0");

/*
 * Places the two imported SO-101 meshes (pose_so101_stl.py) on the rig's riser stacks (2026-11-19).
 * ⚠️ Template: the four __E_*__ element ids and __MV__ (document microversion) are filled in when the
 * source is uploaded to the Onshape document; they differ per document.
 * ⚠️ Why a feature and not the assembly: some mesh pieces import as open surfaces, and Onshape
 * assemblies skip surfaces. Instantiating the Part Studios here keeps every piece.
 * Placement from the rig feature's rigInfo: base mid-aisle (y = row pitch / 2), at rigInfo.armX / armZ, turned by
 * rigInfo.armYaw so shoulder_pan 0 faces that arm's row (the sim's ARM_YAW_DEG 90 = facing row B; arm 2 uses -90).
 */
ArmReachA::import(path : "__E_REACH_A__", version : "__MV__");
ArmReachB::import(path : "__E_REACH_B__", version : "__MV__");
ArmStartA::import(path : "__E_START_A__", version : "__MV__");
ArmStartB::import(path : "__E_START_B__", version : "__MV__");

annotation { "Feature Type Name" : "SO-101 arms on the cart" }
export const so101Arms = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        var info = { "armZ" : 140, "aisle" : 200, "armX" : [225, 675], "armYaw" : [90, -90] };   // rig v2 defaults
        try silent { info = getVariable(context, "rigInfo"); }
        const inst = newInstantiator(id + "arms");
        var qs = [];
        const builds = [[ArmReachA::build, ArmReachB::build], [ArmStartA::build, ArmStartB::build]];
        for (var k = 0; k < 2; k += 1)
        {
            const yaw = rotationAround(line(vector(0, 0, 0) * meter, vector(0, 0, 1)), info.armYaw[k] * degree);
            const T = transform(vector(info.armX[k], info.aisle, info.armZ) * millimeter) * yaw;
            for (var b in builds[k])
                qs = append(qs, addInstance(inst, b, { "transform" : T }));
        }
        instantiate(context, inst);
        const q = qUnion(qs);
        setProperty(context, { "entities" : q, "propertyType" : PropertyType.APPEARANCE, "value" : color(0.93, 0.93, 0.9) });
    });
