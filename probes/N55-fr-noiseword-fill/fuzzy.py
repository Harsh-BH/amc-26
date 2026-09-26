# Extension check: noise-word swap, hn equal, city equal, street key DIFFERENT but fuzzy (ratio >= 85) — holdout analogue.
import sys; from pathlib import Path
import numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio
sys.path[:0] = ["probes/N55-fr-noiseword-fill", "probes/P46-noise-word-tier"]
import n55, p46
H = n55._helpers(); W = n55.W
def mask(d, split, c, words):
    r1 = H["rf"](W / f"rf_{split}_s1_{c}.parquet", pd.unique(d.s1), ["core", "hn", "places"]); rx = H["rf"](W / f"rf_{split}_sx_{c}.parquet", pd.unique(d.sx), ["core", "hn", "places", "is_web"])
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    g = n55.gained(c1, cx)
    pre = (h1 != "") & (h1 == hx) & np.array([len(x) == 1 and x[0] in words for x in g]) & np.array([bool(set(a.split()) & set(b.split())) for a, b in zip(c1, cx)])
    idx = np.flatnonzero(pre)
    p1 = r1.places.reindex(d.s1.to_numpy()[idx]).fillna("").to_numpy(); px = rx.places.reindex(d.sx.to_numpy()[idx]).fillna("").to_numpy()
    idx = idx[np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)], bool)]
    s1i, sxi = d.s1.to_numpy()[idx], d.sx.to_numpy()[idx]
    a1, ax = H["addr"](split, pd.unique(s1i), True).reindex(s1i), H["addr"](split, pd.unique(sxi), False).reindex(sxi)
    key = H["french_street_key"] if c == "France" else H["street_key"]
    k1, kx = key(a1).to_numpy(), key(ax).to_numpy()
    fz = np.array([a != "" and b != "" and a != b and ratio(a, b) >= 85 for a, b in zip(k1, kx)])
    diff = np.array([a != b for a, b in zip(k1, kx)])
    m_f = np.zeros(len(d), bool); m_f[idx[fz]] = True
    m_d = np.zeros(len(d), bool); m_d[idx[diff & ~fz]] = True
    return m_f, m_d
for c in ("US", "India"):
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred"]); d = d[d.holdout & d.am].reset_index(drop=True)
    lt = p46.lens_table(d[["s1", "sx"]], "train", c); noise = set(lt[(lt.n >= 20) & (lt.s <= 0.05) & (lt.e >= 0.8)].index)
    mf, md = mask(d, "train", c, noise)
    for nm, m in (("fuzzy>=85", mf), ("street differs <85", md)):
        x = d[m]; print(c, nm, len(x), "prec", round(x.y.mean(), 4) if len(x) else None, "rej", int((~x.pred).sum()), "neg", int((~x.y).sum()))
fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax", "hn_cls"])
mf, md = mask(fr, "test", "France", {"fils", "cie", "associes", "groupe"})
taken = set(fr.sx[fr.acc]); nonemp = set(fr.s1[fr.acc])
for nm, m in (("fuzzy>=85", mf), ("street differs <85", md)):
    x = fr[m & fr.argmax.astype(bool) & (fr.hn_cls != 1)]; r = x[~x.acc & ~x.sx.isin(taken)]; r = r[~((r.p >= 0.8) & r.s1.isin(nonemp))]
    print("France", nm, "cell", len(x), "new", len(r))
    if nm == "fuzzy>=85": r[["s1", "sx", "p"]].to_parquet(W / "n55/fuzzy_new.parquet", index=False)
