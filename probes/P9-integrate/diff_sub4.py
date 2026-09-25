"""Line-by-line comparison of a run.py test output with a staged submission (#4, #5), both files, plus the
differing S1 with the ids only one side has.

    .venv/bin/python probes/P9-integrate/diff_sub4.py output/p9 /home/harsh1/github-repos/amc-26/output/sub4
"""
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = ["code/business_entity_resolution/src", "."]
from worlds import WORK

mine, ref = Path(sys.argv[1]), Path(sys.argv[2])
country = pd.read_parquet(WORK / "test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
res, diffs = {}, []
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    n = same = 0
    bad = {}
    with open(mine / name, encoding="utf-8") as a, open(ref / name, encoding="utf-8") as b:
        assert a.readline() == b.readline(), "header"
        for la, lb in zip(a, b):
            n += 1
            if la == lb:
                same += 1
                continue
            sa, sb = la.rstrip("\n").split("\t"), lb.rstrip("\n").split("\t")
            assert sa[0] == sb[0], f"row order differs at line {n}: {sa[0]} vs {sb[0]}"
            bad[sa[0]] = (set(filter(None, sa[1].split(","))), set(filter(None, sb[1].split(","))))
        assert a.readline() == b.readline() == "", "same row count"
    by_c = pd.Series([country[s] for s in bad]).value_counts().to_dict() if bad else {}
    res[name] = {"rows": n, "identical": same, "share": same / n, "differ": n - same, "differ_by_country": by_c}
    print(name, res[name], flush=True)
    if name == "matching_results.tsv":
        diffs = [(s, sorted(x - y), sorted(y - x)) for s, (x, y) in bad.items()]
for s, only_mine, only_ref in diffs[:20]:
    print(f"  {s} {country[s]}: only in mine {only_mine}, only in {ref.name} {only_ref}")
if len(sys.argv) > 3:
    from ledger import log
    log("exp", probe="P9-integrate", subproblem="integration", name=f"test-model-vs-{ref.name}", world="test",
        metrics={f"{k.split('_')[0]}_{m}": v for k, r in res.items() for m, v in r.items() if m != "differ_by_country"}
        | {f"match_differ_{c}": v for c, v in res["matching_results.tsv"]["differ_by_country"].items()},
        config={"mine": str(mine), "ref": str(ref)}, hypothesis=f"run.py --split test --stage model reproduces {ref.name}",
        verdict="keep" if res["matching_results.tsv"]["share"] >= 0.9999 else "drop", runtime_s=0, peak_gb=0)
