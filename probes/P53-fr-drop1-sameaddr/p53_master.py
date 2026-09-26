# P53 (master): pure one-word drop at an IDENTICAL address (hn + street key + city). Holdout negatives per S1 vs France adds.
import sys; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
R = Path("."); W = R / "data/work"
sys.path[:0] = [str(R / "code/business_entity_resolution/src"), str(R / "probes/P28-addr-anchor"), str(R / "probes/P20-france-patterns"), str(R / "probes/P49-fr-fuzzy-street")]
from anchor import street_key
from fr_street import french_street_key
from patterns import name_rel
LEGAL = {"sarl","sas","sasu","sa","eurl","sci","snc","ei","llc","inc","ltd","pvt","private","limited","co","corp","llp","pc","plc","company","l.l.c."}
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}

def rf(path, ids, cols):
    t = pq.read_table(path, columns=["id", *cols]); t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
    return t.to_pandas().set_index("id").reindex(ids)

def addr(split, ids, s1):
    fs = [W / f"{split}_source1.parquet"] if s1 else [W / f"{split}_source{i}.parquet" for i in (2, 3)]
    out = []
    for f in fs:
        t = pq.read_table(f, columns=["entity_id", "business_address"]); t = t.filter(pc.is_in(t.column("entity_id"), value_set=pa.array(ids, pa.large_string())))
        out.append(t.to_pandas())
    return pd.concat(out).set_index("entity_id").business_address.reindex(ids).fillna("")

def cell(d, split, c):
    """d: s1, sx (+ anything). Returns (drop1 & identical address & not web) mask and the dropped word."""
    u1, ux = pd.unique(d.s1), pd.unique(d.sx)
    r1 = rf(W / f"rf_{split}_s1_{c}.parquet", u1, ["core", "hn", "places"]); rx = rf(W / f"rf_{split}_sx_{c}.parquet", ux, ["core", "hn", "places", "is_web"])
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy().astype(bool)
    pre = (h1 != "") & (h1 == hx) & ~wx
    idx = np.flatnonzero(pre)
    kind = np.array([name_rel(a, b, False)[0] for a, b in zip(c1[idx], cx[idx])])
    idx = idx[kind == "drop1"]
    p1 = r1.places.reindex(d.s1.to_numpy()[idx]).fillna("").to_numpy(); px = rx.places.reindex(d.sx.to_numpy()[idx]).fillna("").to_numpy()
    ceq = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)])
    idx = idx[ceq]
    s1i, sxi = d.s1.to_numpy()[idx], d.sx.to_numpy()[idx]
    a1, ax = addr(split, pd.unique(s1i), True).reindex(s1i), addr(split, pd.unique(sxi), False).reindex(sxi)
    key = french_street_key if c == "France" else street_key
    k1, kx = key(a1).to_numpy(), key(ax).to_numpy()
    idx = idx[(k1 == kx) & (k1 != "")]
    m = np.zeros(len(d), bool); m[idx] = True
    dropped = np.array([next((t for t in a.split() if t not in set(b.split())), "") for a, b in zip(c1, cx)], object)
    return m, dropped

out = {}
for c in ("US", "India"):
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet"); d = d[d.holdout].reset_index(drop=True)
    n1 = d.s1.nunique()
    m, dw = cell(d, "train", c)
    x = d[m].assign(legal=np.isin(dw[m], list(LEGAL)))
    for lg in (False, True):
        y = x[x.legal == lg]
        out[(c, lg)] = dict(n=len(y), prec=y.y.mean(), neg_per_s1=(~y.y).sum() / n1, rej=(~y.pred).sum(), rej_prec=y[~y.pred].y.mean(), per_s1=len(y) / n1)
print(pd.DataFrame(out).T.round(5))

fr = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax", "hn_cls", "src"])
taken = set(fr.sx[fr.acc]); nonemp = fr.groupby("s1").acc.transform("any")
has = fr[fr.acc].assign(s=lambda z: z.src).groupby("s1").s.agg(lambda z: set(z))
m, dw = cell(fr, "test", "France")
x = fr[m].assign(legal=np.isin(dw[m], list(LEGAL)), nonemp=nonemp[m])
print("France cell", len(x), "per S1", round(len(x) / 259452, 4), "acc", int(x.acc.sum()))
r = x[~x.acc & x.argmax.astype(bool) & ~x.sx.isin(taken) & (x.hn_cls != 1)]
hnx = (r.p >= 0.8) & r.nonemp
src_missing = [ (s not in has.get(a, set())) and len(has.get(a, set())) > 0 for a, s in zip(r.s1, r.src)]
gc = (r.p >= 0.7) & np.array(src_missing)
new = r[~hnx & ~gc]
for lg in (False, True):
    y = new[new.legal == lg]
    nb = {c: out[(c, lg)]["neg_per_s1"] * 259452 for c in ("US", "India")}
    print(f"legal={lg}: new {len(y)} (fill empty {int((~y.nonemp).sum())}) bound US {1 - nb['US'] / max(len(y), 1):.3f} India {1 - nb['India'] / max(len(y), 1):.3f}; p bands",
          pd.cut(y.p, [0, .1, .3, .5, .7, .8, 1]).value_counts().sort_index().tolist())
new[["s1", "sx", "p", "legal"]].to_parquet(W / "p53_new_pairs.parquet", index=False)
