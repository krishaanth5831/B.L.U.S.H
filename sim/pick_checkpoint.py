#!/usr/bin/env python3
"""Pick a training run's checkpoint by rollouts, not by "last" or the loss (post-mortem section 6.9, 2026-11-11).

    python pick_checkpoint.py ../outputs/train/act_teacher_onelayout_v1 -- --headless --no-dr --no-yolo \
        --same-seed --seed-start 9100009 --scenes 19 --noise 1.0 --n-action-steps 100

Why: step 1 scored 74 % at 20k steps, 95 % at 40k and 5 % at 50k while the training loss stayed flat (0.11). At 50k
the policy opened the jaw about 10 s into every rollout that cut on the teacher's schedule, wherever the arm was:
it had keyed on time in one memorised scene, and nothing in the loss showed it.

Runs eval_truss.py (everything after "--" is passed through) on every numbered checkpoint of the run, oldest first,
skipping ones already ranked with the same arguments, then ranks them by successes, then strict successes, then
the median closest approach. Writes ../outputs/eval/<run>_ranking.json and prints the table and the best.
Run as eval_truss.py is run (lerobot env, outside the Claude sandbox, under systemd-run MemoryMax).
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def summary(log_path):
    rows = [json.loads(line) for line in pathlib.Path(log_path).read_text().splitlines() if line.strip()]
    return next((r for r in rows if r.get("summary")), None)


def main():
    argv = sys.argv[1:]
    passthru = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=pathlib.Path, help="a training run's output dir (holds checkpoints/)")
    ap.add_argument("--only", nargs="*", default=None, help="these checkpoint names only, e.g. 040000 050000")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    cks = sorted(p for p in (a.run / "checkpoints").iterdir() if p.name.isdigit())
    if a.only:
        cks = [p for p in cks if p.name in a.only]
    out = HERE.parent / "outputs" / "eval" / f"{a.run.name}_ranking.json"
    ranked = json.loads(out.read_text()) if out.exists() else {}
    key = " ".join(passthru)
    for ck in cks:
        tag = f"{ck.name} | {key}"
        if tag in ranked:
            continue
        print(f"▶ {ck.name}", flush=True)
        p = subprocess.run([sys.executable, str(HERE / "eval_truss.py"), str(ck / "pretrained_model"), *passthru],
                           capture_output=True, text=True, cwd=HERE)
        m = re.search(r"log (\S+\.jsonl)", p.stdout)
        if p.returncode or not m:
            print(f"  failed (exit {p.returncode}):\n{p.stdout[-800:]}{p.stderr[-800:]}", flush=True)
            continue
        s = summary(m.group(1))
        ranked[tag] = dict(checkpoint=ck.name, success=s["success"], strict=s["success_strict"],
                           scenes=s["scenes"], cut=s["cut"], in_crate=s["in_crate"],
                           closest_mm=s["closest_mm_median"], log=m.group(1))
        out.write_text(json.dumps(ranked, indent=1))
        print(f"  {s['success']}/{s['scenes']} successes (cut {s['cut']}, crate {s['in_crate']}, "
              f"closest {s['closest_mm_median']:.1f} mm)", flush=True)
    rows = sorted((r for t, r in ranked.items() if t.endswith(f"| {key}")),
                  key=lambda r: (-r["success"], -r["strict"], r["closest_mm"]))
    print(f"\nranking ({key}):")
    for r in rows:
        print(f"  {r['checkpoint']}: {r['success']}/{r['scenes']} (strict {r['strict']}), cut {r['cut']}, "
              f"crate {r['in_crate']}, closest {r['closest_mm']:.1f} mm")
    if rows:
        print(f"best: {a.run / 'checkpoints' / rows[0]['checkpoint'] / 'pretrained_model'}  ({out})")


if __name__ == "__main__":
    main()
