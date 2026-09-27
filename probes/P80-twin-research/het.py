"""Is address-dropping iid per copy, or per-entity (overdispersed)? Train truth, aggregates only."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = "data/work/"
NA = {"", "null", "<null>"}
g = pq.read_table(W + "gt_pairs.parquet").to_pandas()
sx = pd.concat([pq.read_table(W + f"train_source{i}.parquet", columns=["entity_id", "business_name", "business_address"]).to_pandas()
                for i in (2, 3)], ignore_index=True)
sx["na"] = sx.business_address.str.strip().str.lower().isin(NA)
sx = sx.set_index("entity_id")
g = g.join(sx, on="sx")
# drop content-identical duplicates inside a cluster (copies of one generated record)
g = g.drop_duplicates(["s1", "business_name", "business_address"])
c = g.groupby("s1").agg(k=("na", "size"), m=("na", "sum"))
q = g.na.mean()
print(f"copies {len(g):,}  address-less rate q={q:.4f}  clusters {len(c):,}")
for k in range(2, 9):
    s = c[c.k == k]
    if len(s) < 1000: continue
    qk = s.m.mean() / k
    obs = s.m.var(); exp = k * qk * (1 - qk)
    p2_obs = (s.m >= 2).mean(); p2_exp = 1 - (1 - qk) ** k - k * qk * (1 - qk) ** (k - 1)
    print(f"k={k} n={len(s):,} q_k={qk:.4f} var_ratio={obs/exp:.3f}  P(m>=2) obs={p2_obs:.4f} binom={p2_exp:.4f} ratio={p2_obs/p2_exp:.2f}")
# conditional: P(copy address-less | another (distinct) copy in cluster address-less)
g = g.join(c, on="s1")
other = g.m - g.na
print("P(na | >=1 other na) =", round(g.na[other >= 1].mean(), 4), " P(na | 0 other na) =", round(g.na[other == 0].mean(), 4))
