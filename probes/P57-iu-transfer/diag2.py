"""P57 diag 2: is P45's fsub (trained on folds 6-9) a fair out-of-sample stand-in for m2_big (folds 2-5) on competitor rows?
Log-loss by fold with the real labels: m2 is in-sample on 2-5, fsub on 6-9."""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work"
fo = pd.read_parquet(f"{W}/s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
gt = pq.read_table(f"{W}/gt_pairs.parquet")
L = lambda a: a.cast(pa.large_string())
key = lambda a, b: pc.binary_join_element_wise(L(a), L(b), pa.scalar("|", pa.large_string()))
gk = key(gt.column("s1"), gt.column("sx")); del gt
ll = lambda p, y: float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
for c in ("India", "US"):
    t = pq.read_table(f"{W}/p45/oof/holdout_{c}.parquet", columns=["s1", "sx", "holdout", "p_m2", "p_fsub", "p8c"])
    y = pc.is_in(key(t.column("s1"), t.column("sx")), value_set=gk).to_numpy(zero_copy_only=False)
    d = t.drop(["sx"]).to_pandas(); del t
    d["y"] = y
    d["fold"] = np.where(d.holdout, 0, fo.reindex(d.s1).to_numpy())
    r = d.groupby("fold").apply(lambda x: pd.Series({"ll_m2": ll(x.p_m2, x.y), "ll_fsub": ll(x.p_fsub, x.y), "ll_p8c": ll(x.p8c, x.y),
                                                      "corr_logit": np.corrcoef(np.log(np.clip(x.p_m2, 1e-6, 1-1e-6) / (1 - np.clip(x.p_m2, 1e-6, 1-1e-6))),
                                                                                np.log(np.clip(x.p_fsub, 1e-6, 1-1e-6) / (1 - np.clip(x.p_fsub, 1e-6, 1-1e-6))))[0, 1]}))
    print(c); print(r.round(4).to_string())
