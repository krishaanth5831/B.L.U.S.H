#!/usr/bin/env python3
"""One table of every checkpoint test of a training run (K7 2026-11-19: "always after each test create a table").
Reads the summary row of each outputs/eval/<run>_<step>_*.jsonl that pick_checkpoint.py --watch wrote (the alone and
hybrid tests of train_and_test.sh) and prints two Markdown tables, one row per step.
    python3 sim/scripts/checkpoint_table.py act_teacher_v5                 the tables now
    python3 sim/scripts/checkpoint_table.py act_teacher_v5 --wait 35000    wait until both tests of step 35000 are in
Plain python3, no env needed. Policy only: the policy drives every frame, no teacher anywhere (+ the holds it should
leave alone). Hybrid: the policy drives and teacher v5 steps in only when it is failing (rescue.Gate); how often it
stepped in, how many takeovers ended in the crate, how many scenes the policy finished by itself, and why it stepped in.
"""
import argparse
import collections
import json
import pathlib
import re
import time

EVAL = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "eval"


def load(run, scenes=25):
    """{(step, infer): {"alone": summary, "hybrid": summary, "why": Counter}} from the newest log of each (step,
    infer, kind) that ran `scenes` scenes (⚠️ the 100-scene runs of the best checkpoints and short diagnostic re-runs
    land in the same folder; without this filter they replaced a step's 25-scene row). infer: "" for the policy as
    trained, else eval_truss.py's label of a changed inference (--unstretch, --mc-dropout), which gets its own row."""
    out = collections.defaultdict(dict)
    for f in sorted(EVAL.glob(f"{run}_[0-9][0-9][0-9][0-9][0-9][0-9]_*.jsonl"), key=lambda p: p.stat().st_mtime):
        step = int(re.match(rf"{re.escape(run)}_(\d{{6}})_", f.name).group(1))
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        summ = [r for r in rows if r.get("summary")]
        if not summ:
            continue                                   # still running
        if summ[-1].get("scenes") != scenes:
            continue
        kind = "hybrid" if summ[-1].get("hybrid") else "alone"
        if kind == "hybrid" and (summ[-1].get("teacher_drives") or summ[-1].get("trigger") != "failing"):
            continue                                   # the teacher-alone check and the old fixed hand-over
        key = (step, summ[-1].get("infer") or "")
        out[key][kind] = summ[-1]
        if kind == "hybrid":
            resc = [r for r in rows if not r.get("summary") and r.get("rescued")]
            out[key]["why"] = collections.Counter((r.get("rescue_why") or "?").split(":")[0] for r in resc)
            out[key]["after_cut"] = sum(r.get("cut_t") is not None and r.get("rescue_t") is not None
                                        and r["rescue_t"] >= r["cut_t"] for r in resc)
    return out


def table(run, scenes=25):
    """Two tables (K7 2026-11-19: "one test which is purely just the policy without the teacher stepping in anywhere
    and one hybrid"): the policy-only test, then the hybrid test, each on the same held-out scenes."""
    t = load(run, scenes)
    # "First closing + held": cut with the first closing of the jaw and the truss still in the jaw 0.5 s later
    # (eval_truss.CleanCut; post-mortem fix 6: the measure that decides success. Tests before 2026-11-26: "-")
    pol = ["**Policy only** (no teacher anywhere)", "",
           "| Step | Success | First closing + held | Clean (neighbours ≤ 15 mm) | Cut | In crate "
           "| Closest to cut point (median) | Holds left alone |", "|---|---|---|---|---|---|---|---|"]
    hyb = ["**Hybrid** (policy drives, teacher steps in only when it is failing)", "",
           "| Step | Success | Teacher stepped in | Takeovers that worked | Policy finished by itself "
           "| Why it stepped in |", "|---|---|---|---|---|---|"]
    for key in sorted(t):
        a, h = t[key].get("alone"), t[key].get("hybrid")
        step = f"{key[0] // 1000}k" + (f" ({key[1]})" if key[1] else "")
        if a is None:
            pol.append(f"| {step} | … | … | … | … | … | … | … |")
        else:
            n = a["scenes"]
            first = f"{a['clean_cut']}/{n}" if "clean_cut" in a else "-"
            # nothing-ripe scenes: those the start guard stopped (the policy was not run) count with the rest
            g, g_ok = a.get("guard_fired", 0), a.get("guard_fired_ok", 0)
            holds = f"{a['holds_ok'] + g_ok}/{a['holds'] + g}" + (f" ({g} by the start guard)" if g else "")
            pol.append(f"| {step} | {a['success']}/{n} | {first} | {a['success_strict']}/{n} | {a['cut']}/{n} "
                       f"| {a['in_crate']}/{n} | {a['closest_mm_median']:.1f} mm | {holds} |")
        if h is None:
            hyb.append(f"| {step} | … | … | … | … | … |")
        else:
            n, after = h["scenes"], t[key]["after_cut"]
            why = ", ".join(f"{k} {v}" for k, v in t[key]["why"].most_common()) or "-"
            hyb.append(f"| {step} | {h['success']}/{n} | {h['rescued']}/{n}"
                       + (f" ({after} after the cut)" if after else "")
                       + f" | {h['rescued_ok']}/{h['rescued']} | {h['on_its_own_ok']}/{h['on_its_own']} | {why} |")
    return "\n".join(pol + ([""] + hyb if any("hybrid" in v for v in t.values()) else []))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--wait", type=int, help="block until this step's test is in (checks every 60 s)")
    ap.add_argument("--hybrid", action="store_true", help="--wait: also wait for the hybrid test (colab_tests.sh "
                                                          "KINDS=\"alone hybrid\"; K7 2026-11-23: policy only)")
    ap.add_argument("--timeout-min", type=float, default=120)
    ap.add_argument("--scenes", type=int, default=25, help="only the tests of this many scenes (the watch's 25; 100 "
                                                            "for the full held-out runs of the best checkpoints)")
    a = ap.parse_args()
    if a.wait:
        t0 = time.time()
        want = {"alone", "hybrid"} if a.hybrid else {"alone"}
        while not want <= set(load(a.run, a.scenes).get((a.wait, ""), {})):
            if time.time() - t0 > a.timeout_min * 60:
                print(f"⚠️ step {a.wait}: test not in after {a.timeout_min:.0f} min")
                break
            time.sleep(60)
    print(table(a.run, a.scenes))
