#!/usr/bin/env python3
"""Does eval_truss.CleanCut (the test's own "cut with the first closing and still held 0.5 s later") give the same
answer as the post-mortem's rule (scripts/postmortem_v5/funnel.py fate), frame for frame? No simulator: the traced
runs of the post-mortem (outputs/diag/runs/<run>/trace.jsonl + act_*.jsonl) are replayed through CleanCut.

    python scripts/check_clean_cut.py            # from sim/, env lerobot-mj4
"""
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "postmortem_v5"))
import analyse_after_cut as ac  # noqa: E402
import funnel  # noqa: E402

sys.argv = sys.argv[:1]
import eval_truss as et  # noqa: E402


def main():
    runs = sorted(p.parent.name for p in ac.RUNS.glob("*/trace.jsonl") if list(p.parent.glob("act_*.jsonl")))
    total, differ, C = 0, [], collections.Counter()
    for name in runs:
        R, T = ac.rows(name), ac.traces(name)
        n = bad = 0
        for seed, r in R.items():
            tr = T.get(seed)
            if not tr:
                continue
            how, held = funnel.fate(r, tr)
            cc = et.CleanCut()
            for x in tr:
                cc.step(x["t"], x["cmd_gripper"], x["gripper"], x["cut"], lambda x=x: x["stalk_to_jaw_mm"])
            n += 1
            C[(how, held)] += 1
            if (cc.how, cc.held) != (how, held):
                bad += 1
                differ.append((name, seed, (how, held), (cc.how, cc.held)))
        total += n
        print(f"{name:28s} {n:4d} runs, {bad} differ")
    print(f"\n{total} traced runs in {len(runs)} tests: CleanCut differs from funnel.fate in {len(differ)}")
    for d in differ[:20]:
        print("  ", d)
    print("what the runs were (how the cut was made, held 0.5 s):")
    for k, v in sorted(C.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k[0]):18s} held={k[1]}: {v}")
    return not differ and total > 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
