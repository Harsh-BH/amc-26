"""P20: why does France score a pattern cell low? Mean m2_big per-feature contributions (LightGBM pred_contrib, logit
scale) on seeded samples of one (name kind, hn relation) cell: France test (#8c's X) vs the US/India holdout (train X,
fold-0 argmax pairs). Aggregates only.
    .venv/bin/python probes/P20-france-patterns/shap_cells.py sub1 eq [n=20000]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
import features as F
import model as M

W = ROOT / "data/work"


def rows(files, keys: np.ndarray, cols) -> pd.DataFrame:
    """Rows of the X files whose (q1, qx) key is in keys; streamed in 100k batches, only the needed columns."""
    out = []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(100_000, columns=cols):
            k = F._key(b.column("q1").to_numpy().astype(np.int64), b.column("qx").to_numpy().astype(np.int64))
            m = np.isin(k, keys)
            if m.any():
                out.append(b.filter(pa.array(m)).to_pandas())
    return pd.concat(out, ignore_index=True)


def main(kind, hrel, n=20000):
    m = M.load()
    cols = ["q1", "qx"] + m.feature_name()
    rng = np.random.default_rng(0)
    res = {}
    fr = pd.read_parquet(W / "p20/pairs_France.parquet", columns=["s1", "sx", "p8c", "kind", "hrel"])
    fr = fr[fr.p8c.notna()]
    fr = fr[M.argmax_sx(pd.factorize(fr.sx)[0], fr.p8c.to_numpy()) & (fr.kind == kind) & (fr.hrel == hrel)]
    fr = fr.iloc[rng.choice(len(fr), min(n, len(fr)), replace=False)]
    id1, idx = F.ids("test", "France")
    q1 = pd.Index(id1.to_pandas()).get_indexer(fr.s1)
    qx = pd.Index(idx.to_pandas()).get_indexer(fr.sx)
    del fr
    X = rows(sorted((W / "p4/X").glob("test_France_sx_*.parquet")), F._key(q1.astype(np.int64), qx.astype(np.int64)), cols)
    res["France"] = X
    for c in ("US", "India"):
        h = pd.read_parquet(W / f"p20/hold_{c}.parquet", columns=["q1", "qx", "kind", "hrel"])
        h = h[(h.kind == kind) & (h.hrel == hrel)]
        h = h.iloc[rng.choice(len(h), min(n, len(h)), replace=False)]
        res[c] = rows(sorted((W / "p4/X").glob(f"train_{c}_sx_*.parquet")),
                      F._key(h.q1.to_numpy().astype(np.int64), h.qx.to_numpy().astype(np.int64)), cols)
        del h
    t = {}
    for c, X in res.items():
        ct = m.predict(X[m.feature_name()], pred_contrib=True)
        t[c] = pd.Series(ct.mean(0), index=m.feature_name() + ["bias"])
        t[c + "_val"] = X[m.feature_name()].mean().reindex(t[c].index)
        print(c, len(X), "mean logit", round(float(ct.sum(1).mean()), 3), flush=True)
    t = pd.DataFrame(t)
    t["FR-US"] = t.France - t.US
    print(t.reindex(t["FR-US"].abs().sort_values(ascending=False).index).head(18).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))
