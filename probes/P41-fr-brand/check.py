"""Staged-file check: lines that differ from the base are France S1 only, and every match of a changed line is one of
that S1's candidates.   .venv/bin/python probes/P41-fr-brand/check.py output/sub41 output/sub37_gc70"""
import sys

import pandas as pd
import pyarrow.parquet as pq

new, base = sys.argv[1], sys.argv[2]
fr = set(pq.read_table("data/work/test_source1.parquet", columns=["entity_id"], filters=[("country", "==", "France")])
         .column(0).to_pylist())
chg = {}
with open(f"{new}/matching_results.tsv") as f, open(f"{base}/matching_results.tsv") as g:
    for a, b in zip(f, g):
        if a != b:
            s, m = a.rstrip("\n").split("\t")
            assert s == b.split("\t", 1)[0], "row order"
            chg[s] = set(m.split(","))
assert chg and set(chg) <= fr, "only France lines change"
bad = 0
with open(f"{new}/candidate_pairs.tsv") as f:
    for line in f:
        s, c = line.rstrip("\n").split("\t")
        if s in chg:
            bad += len(chg[s] - set(c.split(",")))
print(f"{new} vs {base}: {len(chg):,} changed lines, all France; matches outside candidates: {bad}")
assert bad == 0
