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


def load(run):
    """{step: {"alone": summary, "hybrid": summary, "why": Counter}} from the newest log of each (step, kind)."""
    out = collections.defaultdict(dict)
    for f in sorted(EVAL.glob(f"{run}_[0-9][0-9][0-9][0-9][0-9][0-9]_*.jsonl"), key=lambda p: p.stat().st_mtime):
        step = int(re.match(rf"{re.escape(run)}_(\d{{6}})_", f.name).group(1))
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        summ = [r for r in rows if r.get("summary")]
        if not summ:
            continue                                   # still running
        kind = "hybrid" if summ[-1].get("hybrid") else "alone"
        out[step][kind] = summ[-1]
        if kind == "hybrid":
            scenes = [r for r in rows if not r.get("summary") and r.get("rescued")]
            out[step]["why"] = collections.Counter((r.get("rescue_why") or "?").split(":")[0] for r in scenes)
            out[step]["after_cut"] = sum(r.get("cut_t") is not None and r.get("rescue_t") is not None
                                         and r["rescue_t"] >= r["cut_t"] for r in scenes)
    return out


def table(run):
    """Two tables (K7 2026-11-19: "one test which is purely just the policy without the teacher stepping in anywhere
    and one hybrid"): the policy-only test, then the hybrid test, each on the same held-out scenes."""
    t = load(run)
    pol = ["**Policy only** (no teacher anywhere)", "",
           "| Step | Success | Clean (neighbours ≤ 15 mm) | Cut | In crate | Closest to cut point (median) "
           "| Holds left alone |", "|---|---|---|---|---|---|---|"]
    hyb = ["**Hybrid** (policy drives, teacher steps in only when it is failing)", "",
           "| Step | Success | Teacher stepped in | Takeovers that worked | Policy finished by itself "
           "| Why it stepped in |", "|---|---|---|---|---|---|"]
    for step in sorted(t):
        a, h = t[step].get("alone"), t[step].get("hybrid")
        if a is None:
            pol.append(f"| {step // 1000}k | … | … | … | … | … | … |")
        else:
            n = a["scenes"]
            pol.append(f"| {step // 1000}k | {a['success']}/{n} | {a['success_strict']}/{n} | {a['cut']}/{n} "
                       f"| {a['in_crate']}/{n} | {a['closest_mm_median']} mm | {a['holds_ok']}/{a['holds']} |")
        if h is None:
            hyb.append(f"| {step // 1000}k | … | … | … | … | … |")
        else:
            n, after = h["scenes"], t[step]["after_cut"]
            why = ", ".join(f"{k} {v}" for k, v in t[step]["why"].most_common()) or "-"
            hyb.append(f"| {step // 1000}k | {h['success']}/{n} | {h['rescued']}/{n}"
                       + (f" ({after} after the cut)" if after else "")
                       + f" | {h['rescued_ok']}/{h['rescued']} | {h['on_its_own_ok']}/{h['on_its_own']} | {why} |")
    return "\n".join(pol + [""] + hyb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--wait", type=int, help="block until both tests of this step are in (checks every 60 s)")
    ap.add_argument("--timeout-min", type=float, default=120)
    a = ap.parse_args()
    if a.wait:
        t0 = time.time()
        while not {"alone", "hybrid"} <= set(load(a.run).get(a.wait, {})):
            if time.time() - t0 > a.timeout_min * 60:
                print(f"⚠️ step {a.wait}: not both tests in after {a.timeout_min:.0f} min")
                break
            time.sleep(60)
    print(table(a.run))
