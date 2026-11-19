#!/usr/bin/env python3
"""One table of every checkpoint test of a training run (K7 2026-11-19: "always after each test create a table").
Reads the summary row of each outputs/eval/<run>_<step>_*.jsonl that pick_checkpoint.py --watch wrote (the alone and
hybrid tests of train_and_test.sh) and prints a Markdown table, one row per step.
    python3 sim/scripts/checkpoint_table.py act_teacher_v5                 the table now
    python3 sim/scripts/checkpoint_table.py act_teacher_v5 --wait 35000    wait until both tests of step 35000 are in
Plain python3, no env needed. Columns: the policy alone (+ the holds it should leave), the policy + teacher v5
stepping in only when it fails (rescue.Gate), how often the teacher stepped in, how many of its takeovers ended in the
crate, how many scenes the policy finished by itself, and what made the teacher step in."""
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
    t = load(run)
    lines = ["| Step | Alone | Holds left | Policy + teacher | Teacher stepped in | Takeovers that worked "
             "| Policy alone (in hybrid) | Why it stepped in |",
             "|---|---|---|---|---|---|---|---|"]
    for step in sorted(t):
        a, h = t[step].get("alone"), t[step].get("hybrid")
        cell = lambda s, k: "…" if s is None else s.get(k)
        alone = "…" if a is None else f"{a['success']}/{a['scenes']}"
        holds = "…" if a is None else f"{a['holds_ok']}/{a['holds']}"
        if h is None:
            hy = st = ok = own = why = "…"
        else:
            hy = f"{h['success']}/{h['scenes']}"
            st = f"{h['rescued']}/{h['scenes']}" + (f" ({t[step]['after_cut']} after the cut)" if t[step]['after_cut'] else "")
            ok = f"{h['rescued_ok']}/{h['rescued']}"
            own = f"{h['on_its_own_ok']}/{h['on_its_own']}"
            why = ", ".join(f"{k} {v}" for k, v in t[step]["why"].most_common())
        lines.append(f"| {step // 1000}k | {alone} | {holds} | {hy} | {st} | {ok} | {own} | {why} |")
    return "\n".join(lines)


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
