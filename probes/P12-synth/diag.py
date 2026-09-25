"""Synthetic vs real pair-feature means (true / false pairs), folds >= 2 only: is the generator in the right range?
    .venv/bin/python probes/P12-synth/diag.py India US
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

COLS = ["q1", "qx", "is_true", "fold", "core_eq", "core_jacc", "name_cos", "hn_both", "hn_eq", "hn_diff", "city_eq",
        "state_eq", "addr_cos", "addr_empty_x", "web_x", "sx_margin", "sx_n", "s1_n", "block_rank_sx", "chain_1",
        "rule_acc", "blocked"]


def real(c, n=3):
    fs = sorted(glob.glob(str(W.WORK / f"p4/X/train_{c}_sx_*.parquet")))[:n]
    x = pd.concat([pd.read_parquet(p, columns=COLS) for p in fs])
    return x[x.fold >= 2]


def synth(tgt, src):
    keep = np.sort(pd.Index(W.pool(tgt).entity_id).get_indexer(pd.read_parquet(W.d(tgt, src) / "sources.parquet").entity_id))
    x = pd.concat([pd.read_parquet(p, columns=COLS) for p in sorted((W.OUT / "X").glob(f"{tgt}_from_{src}_*.parquet"))])
    tr = pd.read_parquet(W.d(tgt, src) / "truth.parquet", columns=["s1"])
    rec = x.is_true.sum() / len(tr)
    return x[np.isin(x.q1.to_numpy(), keep)], rec


def table(tgt, src):
    s, rec = synth(tgt, src)
    R = {f"syn{tgt}<-{src}": s, tgt: real(tgt)} | ({src: real(src)} if src in ("US", "India") else {})
    rows = {}
    for k, x in R.items():
        t, f = x[x.is_true], x[~x.is_true]
        am = x.sx_margin.to_numpy() > 0
        rows[k] = {"true_share": x.is_true.mean(), "pairs_per_sx": len(x) / x.qx.nunique(),
                   "pairs_per_s1": len(x) / x.q1.nunique(), "true_is_sx_top": (t.sx_margin > 0).mean()} \
            | {f"T_{c}": t[c].mean() for c in COLS[4:]} \
            | {f"F_{c}": f[c].mean() for c in ("core_eq", "name_cos", "hn_eq", "hn_diff", "city_eq", "sx_margin", "rule_acc")} \
            | {"F_top_share": (am & ~x.is_true.to_numpy()).sum() / max(am.sum(), 1)}
    df = pd.DataFrame(rows)
    print(f"synthetic pair recall (true pairs in candidates): {rec:.4f}")
    print(df.round(3).to_string())
    return df, rec


if __name__ == "__main__":
    table(*sys.argv[1:3])
