"""Merge the finalized parts of generate_truss.py --part-size runs into one local dataset.

    python merge_parts.py krish5831/blush_sim_teacher_v3 krish5831/blush_sim_teacher_v3_s   # step 2: 4 shards

Every local dataset whose name starts with one of the prefixes and ends in _pNN is merged, in name order, into the
first argument (~/.cache/huggingface/lerobot/<out>). A part whose parquet files don't open (a run killed before the
part's finalize: info.json exists from the start, so it proves nothing) is listed and left out. Nothing goes to the Hub.
"""
import argparse
import re
import sys

import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.constants import HF_LEROBOT_HOME


def finalized(d):
    files = sorted(d.glob("data/*/*.parquet")) + sorted(d.glob("meta/episodes/*/*.parquet"))
    try:
        return bool(files) and all(pq.ParquetFile(f).metadata.num_rows >= 0 for f in files)
    except Exception:   # no parquet footer
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="merged repo id, e.g. krish5831/blush_sim_teacher_v3")
    ap.add_argument("prefixes", nargs="+", help="repo id prefixes of the parts, e.g. krish5831/blush_sim_teacher_v3_s")
    a = ap.parse_args()
    if (HF_LEROBOT_HOME / a.out).exists():
        sys.exit(f"⚠️ {HF_LEROBOT_HOME / a.out} exists already (nothing was written).")
    parts, broken = [], []
    for pre in a.prefixes:
        user, name = pre.split("/")
        for d in sorted((HF_LEROBOT_HOME / user).glob(name + "*")):
            if not re.search(r"_p\d{2}$", d.name):
                continue
            (parts if finalized(d) else broken).append(f"{user}/{d.name}")
    if broken:
        print("left out (not finalized):", *broken, sep="\n  ")
    if not parts:
        sys.exit("no finalized parts found")
    print(f"merging {len(parts)} parts into {a.out}:", *parts, sep="\n  ", flush=True)
    aggregate_datasets(parts, a.out, roots=[HF_LEROBOT_HOME / p for p in parts], aggr_root=HF_LEROBOT_HOME / a.out)
    print(f"done: {HF_LEROBOT_HOME / a.out}")


if __name__ == "__main__":
    main()
