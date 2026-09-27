"""P50: per-country diff of output/<NEW> vs output/<REF> (any row order): lines changed, +/- pairs, S1 made empty
that REF had non-empty, S1 filled that REF had empty.   diffc.py NEW REF [REF2 ...]"""
import sys, collections, json
import pandas as pd
cty = pd.read_parquet("data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country.to_dict()


def rows(folder):
    with open(f"output/{folder}/matching_results.tsv", encoding="utf-8") as f:
        next(f)
        return {k: v for k, v in (line.rstrip("\n").split("\t", 1) for line in f)}


new = rows(sys.argv[1])
for ref in sys.argv[2:]:
    old = rows(ref)
    r = collections.defaultdict(lambda: collections.Counter())
    for k, v in new.items():
        w = old[k]
        if v == w:
            continue
        a, b = set(v.split(",")) - {""}, set(w.split(",")) - {""}
        if a == b:
            continue
        c = r[cty[k]]
        c["lines"] += 1
        c["add"] += len(a - b)
        c["rm"] += len(b - a)
        c["emptied"] += (not a) and bool(b)
        c["filled"] += bool(a) and not b
    print(f"{sys.argv[1]} vs {ref}:", json.dumps({k: dict(v) for k, v in sorted(r.items())}), flush=True)
    del old
