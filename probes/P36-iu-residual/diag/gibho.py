import sys
sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import A, W, OUT, state, LB_W
import gib as G
M = G.learn("train")
gib = lambda n: bool(G.is_gib([n], M)[0])
tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
for c in LB_W:
    s = state(c)
    t = tr[tr.s1.isin(set(s["ids"]))]
    nm = pd.concat([A.grab(W / f"train_source{i}.parquet", t.sx.unique(), ["entity_id", "business_name"]) for i in (2, 3)])
    t = t.assign(gib=nm.business_name.reindex(t.sx).map(gib).to_numpy())
    e = pd.read_parquet(OUT / f"err_{c}.parquet")
    st = e.set_index(["s1", "sx"]).type
    t["status"] = pd.Series(st.reindex(pd.MultiIndex.from_frame(t[["s1", "sx"]])).to_numpy()).fillna("tp").to_numpy()
    print(c, "holdout true pairs", len(t), "gib", int(t.gib.sum()), f"{t.gib.mean():.4f}")
    print(t[t.gib].status.value_counts().to_dict(), "| non-gib:", t[~t.gib].status.value_counts(normalize=True).round(4).to_dict())
    # gib SX in the predicted FP set?
    fp = e[e.type.str.startswith("fp")]
    fnm = pd.concat([A.grab(W / f"train_source{i}.parquet", fp.sx.unique(), ["entity_id", "business_name"]) for i in (2, 3)])
    print("  FP pairs with gib SX:", int(fnm.business_name.reindex(fp.sx).map(gib).sum()), "of", len(fp))
