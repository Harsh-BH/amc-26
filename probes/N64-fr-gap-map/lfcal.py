# N64: truth rate of unaccepted argmax pairs by (name_rel, hn relation, legal-form class) in the France-like LOCO chain,
# vs the count of FREE France argmax pairs in the same cell (sub76). Labels only score.
import sys; sys.path[:0] = ["probes/P64-legal-form-feat", "code/business_entity_resolution/src"]
import numpy as np, pandas as pd, lf
pd.set_option("display.width", 250)
rows = []
for s_, t_ in (("US", "India"), ("India", "US")):
    d = pd.read_parquet(f"data/work/n63/chain_{s_}_{t_}.parquet", columns=["s1","sx","y","acc","rel","hn_eq","h1","hx","p","name_1","name_x","n54","sib"])
    d = d[~d.acc].reset_index(drop=True)
    d["lfc"] = lf.pair_class(lf.sig(d.name_1), lf.sig(d.name_x))
    d["hr"] = np.where(d.hn_eq, "eq", np.where((d.h1 == "") | (d.hx == ""), "miss", "diff"))
    d["pb"] = np.where(d.p >= .7, ">=.7", np.where(d.p >= .3, ".3-.7", "<.3"))
    g = d.groupby(["rel", "hr", "lfc", "pb"]).y.agg(["size", "mean"]).rename(columns={"size": f"n_{t_}", "mean": f"y_{t_}"})
    rows.append(g)
G = pd.concat(rows, axis=1)
f = pd.read_parquet("data/work/n64/frfree.parquet", columns=["s1","sx","name_1","name_x","hn_1","hn_x","name_rel","p","r","k1"])
f["lfc"] = lf.pair_class(lf.sig(f.name_1), lf.sig(f.name_x))
f["hr"] = np.where((f.hn_1 != "") & (f.hn_1 == f.hn_x), "eq", np.where((f.hn_1 == "") | (f.hn_x == ""), "miss", "diff"))
f["pb"] = np.where(f.p >= .7, ">=.7", np.where(f.p >= .3, ".3-.7", "<.3"))
f.to_parquet("data/work/n64/frfree_lf.parquet", index=False)
fc = f.groupby(["name_rel", "hr", "lfc", "pb"]).agg(n_FR=("sx", "size"), r_FR=("r", "mean"))
fc.index.names = ["rel", "hr", "lfc", "pb"]
G = G.join(fc, how="outer")
G = G[(G.n_FR >= 200)]
print(G.sort_values("n_FR", ascending=False).round(3).head(60).to_string())
