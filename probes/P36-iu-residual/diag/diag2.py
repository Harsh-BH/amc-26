import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import LB_W, OUT, W, state
import ho_gib as H
pd.set_option("display.width", 200)
for c in LB_W:
    s = state(c); g = H.frame(c, s)
    dump = set(pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["sx"]).sx)
    g["in_dump"] = g.sx.isin(dump)
    top = g.sort_values(["sx", "cos"], ascending=[True, False]).drop_duplicates("sx")
    top["cls"] = np.where(top.y, "true_top1", np.where(top.sx_true_any, "true_elsewhere", "distractor"))
    for dd in (True, False):
        t = top[top.in_dump == dd]
        print(c, "in_dump", dd, t.cls.value_counts(normalize=True).round(3).to_dict())
        print(t.groupby("cls")[["cos", "gap", "rel", "n_close", "hn_eq", "hn_x", "tset", "part", "nx_tok"]].mean().round(3).to_string())
    t = top[top.in_dump]
    for q in ["hn_eq==1 and n_close==1", "hn_eq==1 and n_close==1 and gap>0.2", "cos>0.95 and n_close==1", "tset>=95 and n_close==1 and hn_eq==1"]:
        u = t.query(q); print(f"  {q}: n {len(u)} prec {u.y.mean():.3f}")
