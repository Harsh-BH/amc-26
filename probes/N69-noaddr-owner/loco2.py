import sys, numpy as np, pandas as pd
sys.path[:0] = ["probes/P64-legal-form-feat", "code/business_entity_resolution/src"]
import lf
bins = [0, .1, .3, .6, .9, .98, 1.01]
for s_, t_ in (("US", "India"), ("India", "US")):
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","p","y","acc","noaddr","c1","cx","name_1","name_x"])
    nc = pd.read_parquet(f"data/work/rf_train_s1_{t_}.parquet", columns=["core"]).core.value_counts()
    d = d[d.noaddr & (d.c1 == d.cx) & (d.c1 != "") & (nc.reindex(d.c1).fillna(0).to_numpy() == 1)]
    d["lfc"] = lf.pair_class(lf.sig(d.name_1), lf.sig(d.name_x))
    print(f"== {s_}>{t_}: addr-less unique eq-core argmax {len(d):,}, accepted share {d.acc.mean():.3f}, true {d.y.mean():.3f}")
    u = d[~d.acc]
    print(u.groupby(pd.cut(u.p, bins), observed=True).y.agg(["size", "mean"]).round(3).T.to_string())
    print("   leftover by lfc:", u.groupby("lfc").y.agg(["size", "mean"]).round(3).to_dict("index"))
f = pd.read_parquet("data/work/n69/fr_noaddr.parquet")
f = f[(f.name_rel == "eq") & (f.ncore == 1)]
print(f"== France: addr-less unique eq-core argmax {len(f):,}, accepted share {(f.st=='acc').mean():.3f}")
u = f[f.st == "free"]; print(u.groupby(pd.cut(u.p, bins), observed=True).size().to_string())
print("   leftover by lfc:", u.lfc.value_counts().to_dict())
