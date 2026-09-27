"""P20: the labelled analog of patterns.py on the US/India holdout (P10 hold rows: fold-0 S1 + every competing train pair,
m2_big p). Same name kinds x hn relations, on each SX's argmax pair: truth rate, mean p, acceptance at t. Aggregates only.
    .venv/bin/python probes/P20-france-patterns/analog.py US     # -> data/work/p20/hold_US.parquet + table
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
import features as F
import model as M
from patterns import hn_rel, name_rel

W = ROOT / "data/work"


def build(c: str) -> pd.DataFrame:
    f = W / "p20" / f"hold_{c}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    h = pd.read_parquet(W / f"p10/hold_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "p"])
    h = h[M.argmax_sx(h.qx.to_numpy(), h.p.to_numpy()) & (h.fold.to_numpy() == 0)].reset_index(drop=True)
    s1, sx = F.ids("train", c)

    def of(ids_q, need, side, cols):
        rf = pq.read_table(W / f"rf_train_{side}_{c}.parquet", columns=["id", *cols])
        pos = pc.index_in(ids_q.take(pa.array(need)), value_set=rf.column("id")).to_numpy(zero_copy_only=False)
        return {k: pd.Series(rf.column(k).to_numpy(zero_copy_only=False)[pos], index=need) for k in cols}

    n1, nx = np.unique(h.q1.to_numpy()), np.unique(h.qx.to_numpy())
    a = of(s1, n1, "s1", ["core", "hn"])
    b = of(sx, nx, "sx", ["core", "hn", "is_web"])
    c1, h1 = a["core"].reindex(h.q1).fillna("").to_numpy(), a["hn"].reindex(h.q1).fillna("").to_numpy()
    cx, hx = b["core"].reindex(h.qx).fillna("").to_numpy(), b["hn"].reindex(h.qx).fillna("").to_numpy()
    wx = b["is_web"].reindex(h.qx).fillna(False).to_numpy()
    nr = [name_rel(x, y, w) for x, y, w in zip(c1, cx, wx)]
    h["kind"], h["word"] = [k for k, _ in nr], [w for _, w in nr]
    h["hrel"] = [hn_rel(x, y) for x, y in zip(h1, hx)]
    h.to_parquet(f, index=False)
    return h


def cells(h: pd.DataFrame, t: float) -> pd.DataFrame:
    g = h.assign(acc=h.p >= t, band=(h.p >= 0.3) & (h.p < 0.95))
    return g.groupby(["kind", "hrel"]).agg(n=("p", "size"), true=("is_true", "mean"), p_mean=("p", "mean"),
                                            acc=("acc", "mean"), band=("band", "mean"),
                                            prec_acc=("is_true", lambda s: s[g.loc[s.index, "acc"]].mean()))


if __name__ == "__main__":
    c = sys.argv[1]
    pd.set_option("display.width", 250), pd.set_option("display.max_rows", 200)
    h = build(c)
    print(c, len(h), "argmax fold-0 pairs; true share", round(h.is_true.mean(), 4))
    print(cells(h, 0.69).round(3).to_string())
