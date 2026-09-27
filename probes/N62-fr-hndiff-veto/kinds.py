"""N62: kind mix of hn differences — train truth (US, India) vs France accepted pairs (p31 frame, sub31-era chain)."""
import sys; import numpy as np, pandas as pd
sys.path.insert(0, "probes/N62-fr-hndiff-veto")
from n62 import dtype
W = "data/work"
def mix(h1, hx):
    h1, hx = pd.Series(h1).fillna("").astype(str).to_numpy(), pd.Series(hx).fillna("").astype(str).to_numpy()
    both = (h1 != "") & (hx != ""); eq = both & (h1 == hx); diff = both & ~eq
    k = pd.Series([dtype(a, b) for a, b in zip(h1[diff], hx[diff])]).value_counts(normalize=True).round(3).to_dict()
    return dict(n=int(both.sum()), eq=round(eq.sum() / max(both.sum(), 1), 3), **k)
gt = pd.read_parquet(f"{W}/gt_pairs.parquet"); print(gt.columns.tolist(), len(gt))
for c in ("US", "India"):
    s1 = pd.read_parquet(f"{W}/rf_train_s1_{c}.parquet", columns=["id", "hn"]).set_index("id").hn
    sx = pd.read_parquet(f"{W}/rf_train_sx_{c}.parquet", columns=["id", "hn"]).set_index("id").hn
    g = gt[gt.s1.isin(s1.index)].sample(400000, random_state=0)
    print(c, "TRUE pairs:", mix(s1.reindex(g.s1).to_numpy(), sx.reindex(g.sx).to_numpy()))
f = pd.read_parquet(f"{W}/p31/pairs_France.parquet", columns=["hn_1", "hn_x", "acc", "p", "argmax"])
a = f[f.acc.astype(bool)]
print("France ACCEPTED (p31 chain):", mix(a.hn_1.to_numpy(), a.hn_x.to_numpy()))
for lo, hi in ((0, .9), (.9, .98), (.98, .995), (.995, 1.01)):
    b = a[(a.p >= lo) & (a.p < hi)]; print(f"   p [{lo},{hi}):", mix(b.hn_1.to_numpy(), b.hn_x.to_numpy()))
