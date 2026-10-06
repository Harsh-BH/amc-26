"""Position signature of one-word swaps (label-free): share of pairs whose swapped-in word is the SX name's LAST token while
the S1 keeps a later token after the dropped word (the filler is appended after the legal form: 'Soma Loisirs SAS' -> 'Soma
SAS Groupe'), vs in-place replacement ('X Club SAS' -> 'X Ecole SAS').
    .venv/bin/python probes/P71-fr-noiseword/pos71.py France|US|India"""
import sys

import pandas as pd

sys.path.insert(0, "probes/P71-fr-noiseword")
import p71 as P  # noqa: E402
import normalize as N  # noqa: E402

c = sys.argv[1]
split = "test" if c == "France" else "train"
pool = pd.read_parquet(P.WORK / (f"pool_{c}.parquet" if c == "France" else f"subd0_{c}.parquet"))
pool = pool[pool.kind == "sub"]
pool["k"] = pool.kind + ":" + pool.ins
top = pool.k.value_counts()
pool = pool[pool.k.isin(top.index[top >= 30])]
n1 = N.name_norm(P.raw(pool.s1, (1,), "business_name", split).reindex(pool.s1)).to_numpy()
nx = N.name_norm(P.raw(pool.sx, (2, 3), "business_name", split).reindex(pool.sx)).to_numpy()
app, inplace, after = [], [], []
for a, b, w, dr in zip(n1, nx, pool.ins, pool["drop"]):
    ta, tb = a.split(), b.split()
    last = bool(tb) and tb[-1] == w
    dpos = ta.index(dr) if dr in ta else -1
    app.append(last and 0 <= dpos < len(ta) - 1)                 # appended at the end; the S1 had tokens after the dropped word
    inplace.append(w in tb and dpos >= 0 and tb.index(w) == dpos)
    after.append(0 <= dpos < len(ta) - 1)
pool["appended"], pool["inplace"], pool["after"] = app, inplace, after
g = pool.groupby("k").agg(n=("s1", "size"), appended=("appended", "mean"), inplace=("inplace", "mean"), n_after=("after", "sum"))
g["cond_app"] = pool[pool.after].groupby("k").appended.mean()     # appended share when the S1 has a token after the dropped word
dec = g[(g.n >= 300) & (g.inplace >= 0.8)]
a_d = float((dec.cond_app * dec.n_after).sum() / dec.n_after.sum())
a_t = float(g.cond_app.max())
g["decoy_frac_max"] = ((1 - g.cond_app) / (1 - a_d)).clip(0, 1)            # true noise always appended (a_T = 1)
g["decoy_frac_at"] = ((a_t - g.cond_app) / (a_t - a_d)).clip(0, 1)         # a_T = the most-appended word's share
print(f"decoy words (inplace >= .8, n >= 300): {len(dec)}, pooled cond_app a_D = {a_d:.3f}; max cond_app a_T = {a_t:.3f}")
if "y" in pool:
    g["y"] = pool.groupby("k").y.mean()
    g["y_appended"] = pool[pool.appended].groupby("k").y.mean()
    g["y_inplace"] = pool[pool.inplace].groupby("k").y.mean()
print(g.sort_values("n", ascending=False).head(30).round(3).to_string())
