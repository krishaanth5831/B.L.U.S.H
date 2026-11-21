#!/usr/bin/env python3
"""Post-mortem 2026-11-21: from the first closing of the jaw to success, stage by stage, for traced runs. No simulation.

    python funnel.py                                  the held-out and training-scene variants of 20-21 Nov
    python funnel.py --group "label" run [run ...]     any traced runs in outputs/diag/runs/ (repeatable)

Why (advisor, 2026-11-21): on held-out scenes the corrections moved the jaw closer at the first closing (24.9 ->
13.1 mm) and the runs within 6 mm from 16 to 26, while success went 19 -> 16; the mean of 16 dropout answers had 40
runs within 6 mm and 12 successes. Closer has not turned into success in any variant, so the question is at which
stage the runs drop out.

Stages, per run (eval row + trace):
    first closing at blade depth   the stalk is between the fingers at blade depth when the jaw first shuts
                                   (where_at_shut.cls on aim_close)
    cut by the first closing       the cut comes before the gripper command has opened and shut a second time, and
                                   the jaw was not already shut for a second (analyse_after_cut's rule)
    held 0.5 s after the cut       the stalk's cut end stays within HELD_MM of the jaw point for 0.5 s
    in crate, success              the eval row
And beside them: cuts made by a later closing (the policy went round again), and how those end.
"""
import argparse
import collections

import analyse_after_cut as ac
import where_at_shut as was

DEFAULT = (("HELD-OUT, policy as trained (base_a + base_b + base_c, first 50 scenes x 3)", ("base_a", "base_b", "base_c")),
           ("HELD-OUT, mean of 16 dropout answers, every plan (fix_mcdropout, first 50)", ("fix_mcdropout",)),
           ("HELD-OUT, mean of 16 dropout answers, from plan 2 (fix_mcdropout2, first 50)", ("fix_mcdropout2",)),
           ("HELD-OUT, pictures through the encoder (codec1, 100)", ("codec1",)),
           ("HELD-OUT, pictures through the encoder + stretch divided out (codec1_unstretch, 100)", ("codec1_unstretch",)),
           ("TRAINING, policy as trained (train_clean, 50)", ("train_clean",)),
           ("TRAINING, recorded start frame (train_recorded, 50)", ("train_recorded",)),
           ("TRAINING, recorded start frame + stretch divided out (train_recorded_unstretch, 50)", ("train_recorded_unstretch",)),
           ("TRAINING, pictures through the encoder + stretch divided out (train_codec1_unstretch, 50)", ("train_codec1_unstretch",)),
           ("TEACHER (teacher + train_teacher, 100)", ("teacher", "train_teacher")))


def fate(r, tr):
    """-> (how the cut was made or None, held 0.5 s after the cut or None)"""
    kc = next((i for i, x in enumerate(tr) if x["cut"]), None)
    if kc is None or not r.get("cut"):
        return None, None
    cmds = [x["cmd_gripper"] for x in tr[:kc + 1]]
    closings = sum(1 for u, v in zip(cmds, cmds[1:]) if u >= ac.OPEN and v < ac.OPEN)
    shut_before = tr[max(0, kc - 30)]["gripper"] < ac.SHUT
    how = "jaw already shut" if shut_before else "first closing" if closings <= 1 else "later closing"
    t0 = tr[kc]["t"]
    seg = [x for x in tr[kc:] if x["t"] - t0 <= 0.5 and x["stalk_to_jaw_mm"] is not None]
    held = bool(seg) and all(x["stalk_to_jaw_mm"] <= ac.HELD_MM for x in seg)
    return how, held


def group(label, names):
    C = collections.Counter()
    for n in names:
        R, T = ac.rows(n), ac.traces(n)
        for s, r in R.items():
            tr = T.get(s)
            if not tr:
                continue
            C["runs"] += 1
            a = r.get("aim_close")
            at_depth = bool(a) and was.cls(a["off_grip_mm"]) == was.CLASSES[1]
            how, held = fate(r, tr)
            C["first closing at blade depth"] += at_depth
            if how == "first closing":
                C["cut by the first closing"] += 1
                C["  ... of them from a closing at blade depth"] += at_depth
                C["  held 0.5 s after the cut"] += held
                C["  in crate"] += bool(r.get("in_crate"))
                C["  success"] += bool(r["success"])
            elif how is not None:
                C["cut later (a later closing, or the stalk dragged into a shut jaw)"] += 1
                C["  later cut held 0.5 s"] += held
                C["  later cut in crate"] += bool(r.get("in_crate"))
                C["  later cut success"] += bool(r["success"])
            C["success (all)"] += bool(r["success"])
    if not C["runs"]:
        print(f"\n{label}: no finished traced run")
        return
    n = C["runs"]
    print(f"\n{label}: {n} runs")
    for k in ("first closing at blade depth", "cut by the first closing", "  ... of them from a closing at blade depth",
              "  held 0.5 s after the cut", "  in crate", "  success", "cut later (a later closing, or the stalk dragged into a shut jaw)",
              "  later cut held 0.5 s", "  later cut in crate", "  later cut success", "success (all)"):
        print(f"   {k:66s} {C[k]:4d}  ({C[k] / n:5.0%} of runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar=("LABEL", "RUN"))
    a = ap.parse_args()
    for label, names in ([(g[0], g[1:]) for g in a.group] if a.group else DEFAULT):
        group(label, names)


if __name__ == "__main__":
    main()
