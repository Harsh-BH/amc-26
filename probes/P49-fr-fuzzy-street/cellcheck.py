import sys; sys.path[:0]=["."]
import pandas as pd, numpy as np
import p49
fr = pd.read_parquet("../../data/work/p31/pairs_France.parquet", columns=["s1","sx","p","acc","argmax"])
m = p49.rule_p49(fr, "test", "France", 0.9868)
for k in ("arm_eq","arm_miss"):
    x = fr[m[k]]; print("France", k, len(x), "acc share", round(x.acc.mean(),4), "p<0.9868 share", round((x.p<0.9868).mean(),4), "rejected & am", int((~x.acc & x.argmax.astype(bool)).sum()))
for c in ("US","India"):
    d = pd.read_parquet(f"../../data/work/p23/pairs_{c}.parquet"); d = d[d.holdout].reset_index(drop=True)
    t = d[d.pred].p.min()
    m = p49.rule_p49(d[["s1","sx","p"]], "holdout", c, t)
    x = d[m["arm_eq"]]; print(c, "arm_eq n", len(x), "prec", round(x.y.mean(),4), "rejected share", round((~x.pred).mean(),4), "rej prec", round(x[~x.pred].y.mean(),3), "neg per S1", (~x.y).sum()/d.s1.nunique())
