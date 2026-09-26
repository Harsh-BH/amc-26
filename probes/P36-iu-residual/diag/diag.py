import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd, lightgbm as lgb
from base import LB_W, OUT, W, state
from recover import FEATS
import ho_gib as H
b = lgb.Booster(model_file=str(OUT / "grecover.txt"))
for c in LB_W:
    s = state(c); g = H.frame(c, s)
    dump = set(pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["sx"]).sx)
    g["in_dump"] = g.sx.isin(dump)
    p = b.predict(g[FEATS])
    ad = H.decide(g, p, 0.5)
    ah = ad[ad.half >= 0]
    print(c, "adds to holdout S1 at t .5:", len(ah), "prec", round(ah.y.mean(), 3))
    print(ah.groupby(["in_dump", "rank"]).y.agg(["size", "mean"]).head(12).to_string())
    # where is the SX's true S1 for the wrong adds?
    w = ah[~ah.y]
    tru = g[g.y].set_index("sx")[["s1", "rank", "cos", "half"]]
    w = w.join(tru, on="sx", rsuffix="_true")
    print(" wrong adds: sx true somewhere", round(w.sx_true_any.mean(), 3), "| true S1 in list", round(w.s1_true.notna().mean(), 3),
          "| true S1 is a train S1", round((w.half_true == -1).mean(), 3), "| true rank dist", w.rank_true.value_counts().sort_index().head(5).to_dict())
    print(" wrong adds cos vs true cos (median)", w.cos.median(), w.cos_true.median(), " n_close med", w.n_close.median())
    # all queried SX, by dump status: best-candidate precision
    top = g.sort_values(["sx", "cos"], ascending=[True, False]).drop_duplicates("sx")
    print(" top-1 precision all queries by in_dump:", top.groupby("in_dump").y.agg(["size", "mean"]).round(3).to_dict())
