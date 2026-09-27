"""(1) US/India holdout (labels): legal-form swap with equal core -> true rate + p profile.
(2) France test: same class -> p profile + hn relation. Aggregates only."""
import sys, re, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds, pyarrow.parquet as pq
ROOT = "/home/harsh1/github-repos/amc-26/"
sys.path[:0] = [ROOT + "code/business_entity_resolution/src"]
import features as F, normalize as N
from anyascii import anyascii
NA = re.compile(r"[^a-z0-9]+")
LEG = N.LEGAL - {"co", "company", "etablissement", "etablissements", "ets"}
LEG |= {"selas", "selafa", "selca", "scm", "earl", "gaec", "scea", "sccv", "sem", "spl", "scic", "sep", "sasp", "selurl"}
def legset(names):
    out = []
    for x in names.fillna("").astype(object):
        x = (x if x.isascii() else anyascii(x)).lower().replace(".", "")
        t = NA.sub(" ", x).split()
        out.append(frozenset(t) & LEG)
    return out
SIB = {3, 4, 5, 7, 9, 11, 13, 21}
def hnrel(h1, hx):
    r = []
    for a, b in zip(h1, hx):
        if not a or not b: r.append("missing")
        elif a == b: r.append("eq")
        elif b.isdigit() and a.isdigit() and int(b) - int(a) in SIB: r.append("sib")
        else: r.append("other")
    return np.array(r)
res = {}
for c in sys.argv[2:]:
    h = pd.read_parquet(ROOT + f"data/work/p15/_hn_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "p", "am", "h1", "hx"])
    h = h[h.am & (h.fold == 0)].drop(columns=["am"]).reset_index(drop=True)
    s1, sx = F.ids("train", c)
    h["id1"] = s1.take(pa.array(h.q1.to_numpy())).to_numpy(zero_copy_only=False)
    h["idx"] = sx.take(pa.array(h.qx.to_numpy())).to_numpy(zero_copy_only=False)
    def names(files, ids):
        parts = []
        for f in files:
            t = ds.dataset(ROOT + "data/work/" + f).to_table(filter=pc.field("entity_id").isin(pa.array(np.unique(ids))),
                                                             columns=["entity_id", "business_name"])
            parts.append(t.to_pandas())
        return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id").business_name
    n1 = names(["train_source1.parquet"], h.id1); nx = names(["train_source2.parquet", "train_source3.parquet"], h.idx)
    c1 = pd.read_parquet(ROOT + f"data/work/rf_train_s1_{c}.parquet", columns=["id", "core"]).set_index("id").core
    cx = pd.read_parquet(ROOT + f"data/work/rf_train_sx_{c}.parquet", columns=["id", "core"]).set_index("id").core
    L1, Lx = legset(h.id1.map(n1)), legset(h.idx.map(nx))
    ce = (h.id1.map(c1).fillna("").to_numpy() == h.idx.map(cx).fillna("").to_numpy())
    del c1, cx, n1, nx
    swap = np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(L1, Lx)])
    same = np.array([bool(a) and a == b for a, b in zip(L1, Lx)])
    rel = hnrel(h.h1.fillna("").astype(str), h.hx.fillna("").astype(str))
    r = {"n_argmax": len(h)}
    for nm, m in {"swap&core_eq": swap & ce, "samelegal&core_eq": same & ce}.items():
        r[nm] = {"n": int(m.sum()), "true_rate": float(h.is_true[m].mean()),
                 "p_q": [float(x) for x in np.quantile(h.p[m], [.1, .25, .5, .75, .9])],
                 "acc_at_.71": float((h.p[m] >= .71).mean()),
                 "true_rate_by_hn": {k: [int((m & (rel == k)).sum()), float(h.is_true[m & (rel == k)].mean())] for k in ("eq", "sib", "other", "missing")}}
    res[c] = r
    print(c, json.dumps(r), flush=True)
    del h
# France test
W = ROOT + "data/work/"
p = pd.read_parquet(W + "p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
def load(f):
    return ds.dataset(W + f).to_table(filter=pc.field("country") == "France", columns=["entity_id", "business_name"]).to_pandas().set_index("entity_id").business_name
n1 = load("test_source1.parquet"); nx = pd.concat([load("test_source2.parquet"), load("test_source3.parquet")])
r1 = pd.read_parquet(W + "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
rx = pd.read_parquet(W + "rf_test_sx_France.parquet", columns=["id", "core", "hn"]).set_index("id")
L1, Lx = legset(p.s1.map(n1)), legset(p.sx.map(nx))
ce = p.s1.map(r1.core).fillna("").to_numpy() == p.sx.map(rx.core).fillna("").to_numpy()
rel = hnrel(p.s1.map(r1.hn).fillna("").astype(str), p.sx.map(rx.hn).fillna("").astype(str))
swap = np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(L1, Lx)])
same = np.array([bool(a) and a == b for a, b in zip(L1, Lx)])
pp = p.p.to_numpy()
r = {"n_argmax": len(p)}
for nm, m in {"swap&core_eq": swap & ce, "samelegal&core_eq": same & ce}.items():
    r[nm] = {"n": int(m.sum()), "p_q": [float(x) for x in np.quantile(pp[m], [.1, .25, .5, .75, .9])],
             "acc_at_.985": float((pp[m] >= .985).mean()), "acc_at_.71": float((pp[m] >= .71).mean()),
             "hn_mix": {k: int((m & (rel == k)).sum()) for k in ("eq", "sib", "other", "missing")},
             "acc_.985_by_hn": {k: float((pp[m & (rel == k)] >= .985).mean()) for k in ("eq", "sib", "other", "missing")}}
# swap pairs: which forms
from collections import Counter
cnt = Counter((tuple(sorted(a)), tuple(sorted(b))) for a, b, m in zip(L1, Lx, swap & ce) if m)
r["top_swaps"] = [(f"{'+'.join(a)}->{'+'.join(b)}", n) for (a, b), n in cnt.most_common(15)]
res["France"] = r
print("France", json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
