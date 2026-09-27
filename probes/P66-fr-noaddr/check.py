"""P66 staging check (cheap stand-in for validator --check-ids on the changed lines):
only France lines differ from BASE, only additions, added ids exist in the test S2/S3 files and in the S1's candidate
list, no SX owned twice anywhere in the file.
    .venv/bin/python probes/P66-fr-noaddr/check.py OUT BASE
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def lines(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return dict(line.rstrip("\n").split("\t", 1) for line in f)


def main(out, base):
    out, base = Path(out), Path(base)
    a, b = lines(out / "matching_results.tsv"), lines(base / "matching_results.tsv")
    assert a.keys() == b.keys() and len(a) == 1_732_544
    ch = {s for s in a if a[s] != b[s]}
    fr = set(pq.read_table(W / "test_source1.parquet", columns=["entity_id"], filters=[("country", "=", "France")]).column(0).to_pylist())
    assert ch <= fr, "only France lines change"
    add = {}
    for s in ch:
        x, y = set(filter(None, a[s].split(","))), set(filter(None, b[s].split(",")))
        assert y <= x, "additions only"
        add[s] = x - y
    del b
    ids = sorted({i for v in add.values() for i in v})
    got = set()
    for k in (2, 3):
        got |= set(pq.read_table(W / f"test_source{k}.parquet", columns=["entity_id"],
                                 filters=[("entity_id", "in", pa.array(ids))]).column(0).to_pylist())
    assert got == set(ids), "added ids exist in test S2/S3"
    miss = 0
    with open(out / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t", 1)
            if s in add:
                miss += len(add[s] - set(c.split(",")))
    assert miss == 0, f"{miss} added matches outside the candidates"
    e = pd.Series([i for v in a.values() for i in v.split(",") if i])
    assert not e.duplicated().any(), "an SX owned by two S1"
    print(f"OK {out.name}: {len(ch)} France lines changed, +{sum(map(len, add.values()))} pairs, all in candidates, ids exist, "
          f"{len(e):,} matches unique")


if __name__ == "__main__":
    main(*sys.argv[1:])
