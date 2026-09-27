import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import A, LB_W, OUT, W, state
for c in LB_W:
    s = state(c); pr = s["pairs"]
    e = pd.read_parquet(OUT / f"err_{c}.parquet"); fn = e[e.type.isin(["lowp", "stolen", "block_addr", "block_noaddr", "pruner"])]
    tp = pr[pr.acc & pr.y][["s1", "sx"]]
    q = pd.concat([fn[["s1", "sx"]].assign(k="fn"), tp[tp.s1.isin(set(fn.s1))].assign(k="tp")])
    nm = pd.concat([A.grab(W / f"train_source{i}.parquet", q.sx.unique(), ["entity_id", "business_name", "business_address"]) for i in (2, 3)])
    q["key"] = (nm.business_name.reindex(q.sx).str.lower().str.strip() + "|" + nm.business_address.reindex(q.sx).str.lower().str.strip()).to_numpy()
    tpk = set(zip(q.s1[q.k == "tp"], q.key[q.k == "tp"]))
    f = q[q.k == "fn"].copy(); f["dup_tp"] = [(a, b) in tpk for a, b in zip(f.s1, f.key)]
    f = f.merge(fn[["s1", "sx", "type"]], on=["s1", "sx"])
    print(c, "FN pairs with a content-identical TP twin at the same S1:", int(f.dup_tp.sum()), "of", len(f), f.groupby("type").dup_tp.sum().to_dict())
