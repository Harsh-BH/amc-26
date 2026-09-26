# Website-named SX at the S1's identical address (hn + street key + city): holdout precision; France unaccepted count.
import sys; import numpy as np, pandas as pd
sys.path[:0] = ["probes/N55-fr-noiseword-fill"]
import n55
H = n55._helpers(); W = n55.W
def mask(d, split, c):
    r1 = H["rf"](W / f"rf_{split}_s1_{c}.parquet", pd.unique(d.s1), ["hn", "places"]); rx = H["rf"](W / f"rf_{split}_sx_{c}.parquet", pd.unique(d.sx), ["hn", "places", "is_web"])
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy().astype(bool)
    idx = np.flatnonzero(wx & (h1 != "") & (h1 == hx))
    p1 = r1.places.reindex(d.s1.to_numpy()[idx]).fillna("").to_numpy(); px = rx.places.reindex(d.sx.to_numpy()[idx]).fillna("").to_numpy()
    idx = idx[np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)], bool)]
    s1i, sxi = d.s1.to_numpy()[idx], d.sx.to_numpy()[idx]
    a1, ax = H["addr"](split, pd.unique(s1i), True).reindex(s1i), H["addr"](split, pd.unique(sxi), False).reindex(sxi)
    key = H["french_street_key"] if c == "France" else H["street_key"]
    k1, kx = key(a1).to_numpy(), key(ax).to_numpy()
    m = np.zeros(len(d), bool); m[idx[(k1 == kx) & (k1 != "")]] = True
    return m
for c in ("US", "India"):
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred", "p"]); d = d[d.holdout & d.am].reset_index(drop=True)
    n1 = d.s1.nunique(); x = d[mask(d, "train", c)]
    print(c, "n", len(x), "prec", round(x.y.mean(), 4), "neg/S1", round((~x.y).sum() / n1, 6), "rejected", int((~x.pred).sum()), "rej prec", round(x[~x.pred].y.mean(), 3))
fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax", "hn_cls"])
x = fr[mask(fr, "test", "France") & fr.argmax.astype(bool) & (fr.hn_cls != 1)]
taken = set(fr.sx[fr.acc]); nonemp = set(fr.s1[fr.acc])
r = x[~x.acc & ~x.sx.isin(taken)]; r = r[~((r.p >= 0.8) & r.s1.isin(nonemp))]
print("France cell", len(x), "acc", int(x.acc.sum()), "new", len(r), "fills empty", int((~r.s1.isin(nonemp)).sum()), "p bands", pd.cut(r.p, [0, .02, .1, .3, .5, .8, 1]).value_counts().sort_index().tolist())
r[["s1", "sx", "p"]].to_parquet(W / "n55/web_new.parquet", index=False)
