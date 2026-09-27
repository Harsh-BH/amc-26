"""P57 diag 0: fold composition + calibration of the holdout world's rows, and coverage of per-pair model files."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = "data/work"
fo = pd.read_parquet(f"{W}/s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
for c in ("India", "US"):
    d = pd.read_parquet(f"{W}/p23/pairs_{c}.parquet")
    d["fold"] = fo.reindex(d.s1).to_numpy()
    print(c, "rows", len(d), "holdout rows", int(d.holdout.sum()), "holdout folds", d[d.holdout].fold.value_counts().to_dict())
    print(" pred == am & p>=.71:", bool((d.pred == (d.am & (d.p >= 0.71))).all()), "pred sum", int(d.pred.sum()))
    o = d[~d.holdout]
    g = o.groupby("fold")
    ll = lambda x: float(-(x.y * np.log(np.clip(x.p, 1e-6, 1)) + (~x.y) * np.log(np.clip(1 - x.p, 1e-6, 1))).mean())
    t = pd.DataFrame({"rows": g.size(), "y": g.y.mean(), "p": g.p.mean(), "p_ge_t": g.p.apply(lambda s: (s >= .71).mean()),
                      "band": g.p_m2.apply(lambda s: ((s > .02) & (s < .98)).mean()),
                      "ll_p": g.apply(ll), "ll_m2": g.apply(lambda x: ll(x.assign(p=x.p_m2)))})
    h = d[d.holdout]
    t.loc["holdout"] = [len(h), h.y.mean(), h.p.mean(), (h.p >= .71).mean(), ((h.p_m2 > .02) & (h.p_m2 < .98)).mean(), ll(h), ll(h.assign(p=h.p_m2))]
    print(t.round(4).to_string())
    A = pd.read_parquet(f"{W}/p44ms/A_{c}.parquet", columns=["s1", "sx"])
    for name, path in (("m2hn", f"{W}/p44ms/pred_m2hn/holdout_{c}.parquet"), ("hn21", f"{W}/p44ms/pred_hn21/holdout_{c}.parquet"),
                       ("zoo", f"{W}/p45/oof/holdout_{c}.parquet")):
        f = pq.ParquetFile(path)
        print(" ", name, f.metadata.num_rows, f.schema_arrow.names[:6])
    del d, o
