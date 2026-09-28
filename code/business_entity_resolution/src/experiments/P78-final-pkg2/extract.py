"""P78: the France add-on change sets of #76 / #77 as rows, from the staged folders (all share sub66_final's S1 order).
Adds = a staged file's pairs that sub66_final's line lacks, in that file's order (what combine.py's '+' reads);
n54 = sub75a's pairs that build74.py dropped (sub75a − sub75a_n54).
    .venv/bin/python probes/P78-final-pkg2/extract.py   -> data/work/p78/final_changes.parquet"""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

O = Path("output")
ADDS = [("67q", "sub67q"), ("70a", "sub70a"), ("t975ce", "sub66_t975ce"), ("n54", None), ("71a", "sub71a"), ("41", "sub41_on66")]


def lines(d):
    with open(O / d / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for ln in f:
            s1, _, v = ln.rstrip("\n").partition("\t")
            yield s1, [t for t in v.split(",") if t]


rows, info = [], {}
for step, d in ADDS:
    n = gone = 0
    pairs = zip(lines("sub75a"), lines("sub75a_n54")) if d is None else zip(lines("sub66_final"), lines(d))
    for (k, a), (k2, b) in pairs:
        assert k == k2
        if d is None:
            assert set(b) <= set(a), k
            new = [x for x in a if x not in set(b)]
        else:
            fa = set(a)
            new = [x for x in b if x not in fa]
            gone += len(fa - set(b))
        rows += [(step, k, x) for x in new]
        n += len(new)
    info[step] = dict(rows=n, base_pairs_missing_in_file=gone)
    print(step, info[step], flush=True)
t = pa.table({"step": [r[0] for r in rows], "s1": [r[1] for r in rows], "sx": [r[2] for r in rows]})
pq.write_table(t, "data/work/p78/final_changes.parquet", compression="zstd")
Path("probes/P78-final-pkg2/extract_info.json").write_text(json.dumps(info, indent=1))
