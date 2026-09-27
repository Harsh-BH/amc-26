"""P54: BASE vs OUT sanity. Same S1 set and order; pairs only added; per-country adds; every SX at <= 1 S1; matches a subset
of candidates; lines over the 5 / 6 cap before and after.

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/check54.py BASE OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def lines(p):
    with open(p) as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            yield k, (v.split(",") if v else [])


def eid(x):
    return int(x[1]) * 10**10 + int(x[3:])


def main(base, out):
    B, O = ROOT / "output" / base, ROOT / "output" / out
    cty = {}
    for c in ("France", "India", "US"):
        cty.update(dict.fromkeys(pd.read_parquet(ROOT / f"data/work/rf_test_s1_{c}.parquet", columns=["id"]).id, c))
    add = {c: 0 for c in ("France", "India", "US")}
    from array import array
    rm, s1chg, ids, over_b, over_o, notcand = 0, 0, array("q"), 0, 0, 0
    for (kb, vb), (ko, vo), (kc, vc) in zip(lines(B / "matching_results.tsv"), lines(O / "matching_results.tsv"), lines(O / "candidate_pairs.tsv")):
        assert kb == ko == kc, (kb, ko, kc)
        sb, so = set(vb), set(vo)
        assert len(so) == len(vo), ko
        rm += len(sb - so)
        add[cty[ko]] += len(so - sb)
        s1chg += sb != so
        notcand += len(so - set(vc))
        ids.extend(eid(x) for x in vo)
        cap = lambda v: sum(x.startswith("S2") for x in v) > 5 or sum(x.startswith("S3") for x in v) > 6
        over_b += cap(vb)
        over_o += cap(vo)
    a = np.array(ids, np.int64)
    info = {"base": base, "out": out, "added": add, "removed": rm, "s1_changed": s1chg, "sx_multi_s1": int(len(a) - len(np.unique(a))),
            "matched_not_in_cand": notcand, "lines_over_cap_base": over_b, "lines_over_cap_out": over_o}
    print(json.dumps(info))
    assert info["sx_multi_s1"] == 0 and notcand == 0 and rm == 0
    return info


if __name__ == "__main__":
    main(*sys.argv[1:3])
