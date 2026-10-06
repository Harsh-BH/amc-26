"""P79 staging check of output/<new> against output/<base> (aggregates only -> probes/P79-p73-takeover/check_<new>.json):
only France lines change; added ids exist in test S2/S3 and in the S1's candidate list; no SX owned twice; France links
per S1 (truth 3.46); P38 France Qwen r of the adds and removes (audit, not a gate: P73 found the Qwen gate drops genuine
France pairs); N54's decoy-word removals not re-added. P64's legal-form veto is applied by build67 to every add.
    .venv/bin/python probes/P79-p73-takeover/check79.py sub77 sub79
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, O = ROOT / "data/work", ROOT / "output"


def lines(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return dict(line.rstrip("\n").split("\t", 1) for line in f)


def qstats(q, pairs):
    r = pd.Series([q.get(p) for p in pairs], dtype="float64").dropna()
    return {"pairs": len(pairs), "qwen_scored": len(r), "qwen_median": round(float(r.median()), 3) if len(r) else None,
            "qwen_ge_.5": round(float((r >= .5).mean()), 3) if len(r) else None}


def main(base, new):
    a, b = lines(O / new / "matching_results.tsv"), lines(O / base / "matching_results.tsv")
    assert a.keys() == b.keys() and len(a) == 1_732_544
    fr = set(pq.read_table(W / "test_source1.parquet", columns=["entity_id"], filters=[("country", "=", "France")]).column(0).to_pylist())
    ch = {s for s in a if a[s] != b[s]}
    assert ch <= fr, "only France lines change"
    sp = lambda v: set(filter(None, v.split(",")))
    add = [(s, x) for s in ch for x in sp(a[s]) - sp(b[s])]
    rm = [(s, x) for s in ch for x in sp(b[s]) - sp(a[s])]
    links = lambda d: sum(len(sp(d[s])) for s in fr) / len(fr)
    info = {"base": base, "new": new, "s1_changed": len(ch), "adds": len(add), "removes": len(rm),
            "s1_made_empty": sum(bool(b[s]) and not a[s] for s in ch), "s1_filled": sum(not b[s] and bool(a[s]) for s in ch),
            "fr_links_per_s1": {base: round(links(b), 4), new: round(links(a), 4), "truth": 3.46}}
    del b
    ids = sorted({x for _, x in add})
    got = set()
    for k in (2, 3):
        got |= set(pq.read_table(W / f"test_source{k}.parquet", columns=["entity_id"],
                                 filters=[("entity_id", "in", pa.array(ids))]).column(0).to_pylist())
    assert got == set(ids), "added ids exist in test S2/S3"
    want = {}
    for s, x in add:
        want.setdefault(s, set()).add(x)
    miss = 0
    with open(O / new / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t", 1)
            if s in want:
                miss += len(want[s] - set(c.split(",")))
    assert miss == 0, f"{miss} added matches outside the candidates"
    e = pd.Series([i for v in a.values() for i in v.split(",") if i])
    assert not e.duplicated().any(), "an SX owned by two S1"
    info["matches_total"] = len(e)
    del a, e
    t = pq.read_table(W / "p78/final_changes.parquet").to_pydict()
    n54 = {(s, x) for st, s, x in zip(t["step"], t["s1"], t["sx"]) if st == "n54"}
    info["n54_readded"] = sum(p in n54 for p in add)
    assert info["n54_readded"] == 0, "an N54 decoy-word removal came back"
    q = pd.read_parquet(W / "p38/r/r_qwen_pool_test_France.parquet")
    q = dict(zip(zip(q.s1, q.sx), q.r))
    info["qwen"] = {"adds": qstats(q, add), "removes": qstats(q, rm)}
    (HERE / f"check_{new}.json").write_text(json.dumps(info, indent=1))
    print("OK", json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
