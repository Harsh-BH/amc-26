"""Independent check of a P47 file vs its base: non-France lines byte-identical, every changed France pair is in the base's
candidate_pairs.tsv, no SX twice, caps 5 S2 / 6 S3 hold on changed rows, row count = base.
    .venv/bin/python probes/P47-fr-shape-match/check.py output/sub46_on43 output/sub47_cls80
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
base, new = Path(sys.argv[1]), Path(sys.argv[2])
fr = set(pd.read_parquet(ROOT / "data/work/rf_test_s1_France.parquet", columns=["id"]).id)
sp = lambda x: set(x.rstrip("\n").split("\t")[1].split(",")) - {""}
plus, minus, seen, n, n_other_diff, dup, over = {}, {}, set(), 0, 0, 0, 0
with open(base / "matching_results.tsv", encoding="utf-8") as fb, open(new / "matching_results.tsv", encoding="utf-8") as fn:
    assert fb.readline() == fn.readline()
    for lb, ln in zip(fb, fn):
        n += 1
        k = ln.split("\t", 1)[0]
        assert k == lb.split("\t", 1)[0]
        ids = sp(ln)
        dup += len(ids & seen)
        seen |= ids
        if lb != ln:
            if k not in fr:
                n_other_diff += 1
            a, b = sp(lb), ids
            if b - a:
                plus[k] = b - a
            if a - b:
                minus[k] = a - b
            over += sum(x.startswith("S2-") for x in b) > 5 or sum(x.startswith("S3-") for x in b) > 6
    assert next(fb, None) is None and next(fn, None) is None
miss = 0
with open(base / "candidate_pairs.tsv", encoding="utf-8") as fc:
    fc.readline()
    for line in fc:
        k = line.split("\t", 1)[0]
        if k in plus:
            miss += len(plus[k] - set(line.rstrip("\n").split("\t")[1].split(",")))
            plus[k] = set()
miss += sum(len(v) for v in plus.values())
print(f"rows {n:,}; non-France lines differing {n_other_diff}; S1 with adds {len(plus):,}, with removals {len(minus):,}, "
      f"removed pairs {sum(len(v) for v in minus.values()):,}; SX listed twice {dup}; changed rows over cap {over}; "
      f"added pairs not in candidate_pairs.tsv {miss}")
assert n_other_diff == 0 and dup == 0 and over == 0 and miss == 0
print("check ok")
