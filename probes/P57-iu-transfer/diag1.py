"""P57 diag 1: calibration of #8c p / m2 p on competitor rows by fold, with the real train labels (gt_pairs)."""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work"
fo = pd.read_parquet(f"{W}/s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
gt = pq.read_table(f"{W}/gt_pairs.parquet")
names = gt.schema.names
key = lambda a, b: pc.binary_join_element_wise(a.cast(pa.large_string()), b.cast(pa.large_string()), pa.scalar("|", pa.large_string()))
gk = key(gt.column(names[0]), gt.column(names[1]))
del gt
def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
out = {}
for c in ("India", "US"):
    t = pq.read_table(f"{W}/p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p", "am"])
    yt = pc.is_in(key(t.column("s1"), t.column("sx")), value_set=gk).to_numpy(zero_copy_only=False)
    d = t.drop(["sx"]).to_pandas(); del t
    d["yt"] = yt
    assert (d.yt[d.holdout] == d.y[d.holdout]).all(), "gt != y on holdout rows"
    d["fold"] = np.where(d.holdout, 0, fo.reindex(d.s1).to_numpy())
    rows = []
    for f, x in d.groupby("fold"):
        b = (x.p_m2 > .02) & (x.p_m2 < .98)
        acc = x.am & (x.p >= .71)
        rows.append(dict(fold=f, rows=len(x), y=x.yt.mean(), ll_p=ll(x.p, x.yt), ll_m2=ll(x.p_m2, x.yt),
                         ll_m2_band=ll(x.p_m2[b], x.yt[b]), ll_r_band=ll(x.r[b & x.r.notna()], x.yt[b & x.r.notna()]),
                         band=b.mean(), acc_prec=x.yt[acc].mean(), am_prec=x.yt[x.am].mean(),
                         err_p=float(((x.p >= .5) != x.yt).mean())))
    r = pd.DataFrame(rows).set_index("fold")
    print(c); print(r.round(4).to_string())
    out[c] = r
pd.concat(out).to_csv(f"{W}/p57/diag1.csv")
