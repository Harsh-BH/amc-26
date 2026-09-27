import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import A, LB_W, W, state
import gib as G
M = G.learn("train")
for c in LB_W:
    s = state(c); pr = s["pairs"]
    nm = pd.concat([A.grab(W / f"train_source{i}.parquet", pr.sx.unique(), ["entity_id", "business_name"]) for i in (2, 3)])
    gx = pd.Series(G.is_gib(nm.business_name.tolist(), M), index=nm.index)
    a = pr[pr.am].copy(); a["gib"] = gx.reindex(a.sx).to_numpy()
    b = pd.cut(a.pa, [0, .1, .2, .3, .4, .5, .6, .71, .8, .9, 1])
    t = a.groupby([a.gib, b], observed=True).y.agg(["size", "mean"]).round(3)
    print(c, "\n", t.unstack(0).to_string())
