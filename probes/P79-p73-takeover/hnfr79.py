"""P79: label-free France side of the Qwen question: hn-equal share (P52 france_hn_rel == 1) of output/<new>'s adds and
removes over output/<base>, split by P38's pooled France Qwen r (>= .5 / < .5 / unscored). Aggregates -> hnfr79_<new>.json
    .venv/bin/python probes/P79-p73-takeover/hnfr79.py sub77 sub79
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P42-fr-joint-tune")]
import p42  # noqa: E402


def main(base, new):
    A, B = p42.read_fr(ROOT / "output" / base), p42.read_fr(ROOT / "output" / new)
    sp = lambda v: set(filter(None, v.split(",")))
    ch = A.index[A.to_numpy() != B.reindex(A.index).to_numpy()]
    add = [(k, x) for k in ch for x in sp(B[k]) - sp(A[k])]
    rm = [(k, x) for k in ch for x in sp(A[k]) - sp(B[k])]
    del A, B
    z = p42.fr_frame()
    i1, ix = p42.fr_ids()
    hn = pd.Series(np.load(ROOT / "data/work/p52/france_hn_rel.npy"),
                   index=pd.MultiIndex.from_arrays([i1[z["s1"]], ix[z["sx"]]]))
    del z
    q = pd.read_parquet(ROOT / "data/work/p38/r/r_qwen_pool_test_France.parquet")
    q = pd.Series(q.r.to_numpy(), index=pd.MultiIndex.from_arrays([q.s1, q.sx]))
    out = {}
    for name, pairs in (("adds", add), ("removes", rm)):
        mi = pd.MultiIndex.from_tuples(pairs)
        h, r = hn.reindex(mi).to_numpy(), q.reindex(mi).to_numpy()
        grp = {"all": np.ones(len(pairs), bool), "q>=.5": r >= .5, "q<.5": r < .5, "unscored": np.isnan(r)}
        out[name] = {k: {"pairs": int(m.sum()), "hn_eq": round(float((h[m] == 1).mean()), 3) if m.any() else None,
                         "in_frame": round(float((~np.isnan(h[m].astype(float))).mean()), 3) if m.any() else None}
                     for k, m in grp.items()}
    (HERE / f"hnfr79_{new}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
