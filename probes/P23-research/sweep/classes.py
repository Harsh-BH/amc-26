"""Generator-invariant pair classes: (core_eq, legal relation, hn relation). US/India holdout: true rate + model
acceptance; France test: acceptance at t .985 on P19 p. Aggregates only."""
import sys, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds
sys.path[:0] = [__file__.rsplit("/", 1)[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("ls", sys.path[0] + "/legal_swap.py")
src = open(spec.origin).read().split("res = {}")[0]          # reuse helpers only
exec(src)
def lrel(L1, Lx):
    o = []
    for a, b in zip(L1, Lx):
        if not a and not b: o.append("none")
        elif a == b: o.append("same")
        elif a and b and not (a & b): o.append("swap")
        elif not a or not b: o.append("dropadd")
        else: o.append("overlap")
    return np.array(o)
def table(ce, lr, hr, p, t, y=None):
    df = pd.DataFrame({"ce": ce, "lr": lr, "hr": hr, "acc": p >= t})
    if y is not None: df["y"] = y
    g = df.groupby(["ce", "lr", "hr"])
    out = g.acc.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "acc"})
    if y is not None: out["true"] = g.y.mean()
    return out
res = {}
for c in sys.argv[2:]:
    h = pd.read_parquet(ROOT + f"data/work/p15/_hn_{c}.parquet", columns=["q1", "qx", "fold", "is_true", "p", "am", "h1", "hx"])
    h = h[h.am & (h.fold == 0)].drop(columns=["am"]).reset_index(drop=True)
    s1, sx = F.ids("train", c)
    h["id1"] = s1.take(pa.array(h.q1.to_numpy())).to_numpy(zero_copy_only=False)
    h["idx"] = sx.take(pa.array(h.qx.to_numpy())).to_numpy(zero_copy_only=False)
    def names(files, ids):
        parts = [ds.dataset(ROOT + "data/work/" + f).to_table(filter=pc.field("entity_id").isin(pa.array(np.unique(ids))),
                 columns=["entity_id", "business_name"]).to_pandas() for f in files]
        return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id").business_name
    n1 = names(["train_source1.parquet"], h.id1); nx = names(["train_source2.parquet", "train_source3.parquet"], h.idx)
    c1 = pd.read_parquet(ROOT + f"data/work/rf_train_s1_{c}.parquet", columns=["id", "core"]).set_index("id").core
    cx = pd.read_parquet(ROOT + f"data/work/rf_train_sx_{c}.parquet", columns=["id", "core"]).set_index("id").core
    ce = (h.id1.map(c1).fillna("").to_numpy() == h.idx.map(cx).fillna("").to_numpy())
    lr = lrel(legset(h.id1.map(n1)), legset(h.idx.map(nx)))
    hr = hnrel(h.h1.fillna("").astype(str), h.hx.fillna("").astype(str))
    res[c] = table(ce, lr, hr, h.p.to_numpy(), 0.71, h.is_true.to_numpy())
    print(c, "\n", res[c].to_string(), flush=True)
    del h, n1, nx, c1, cx
W = ROOT + "data/work/"
p = pd.read_parquet(W + "p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
def load(f):
    return ds.dataset(W + f).to_table(filter=pc.field("country") == "France", columns=["entity_id", "business_name"]).to_pandas().set_index("entity_id").business_name
n1 = load("test_source1.parquet"); nx = pd.concat([load("test_source2.parquet"), load("test_source3.parquet")])
r1 = pd.read_parquet(W + "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
rx = pd.read_parquet(W + "rf_test_sx_France.parquet", columns=["id", "core", "hn"]).set_index("id")
ce = p.s1.map(r1.core).fillna("").to_numpy() == p.sx.map(rx.core).fillna("").to_numpy()
lr = lrel(legset(p.s1.map(n1)), legset(p.sx.map(nx)))
hr = hnrel(p.s1.map(r1.hn).fillna("").astype(str), p.sx.map(rx.hn).fillna("").astype(str))
fr = table(ce, lr, hr, p.p.to_numpy(), 0.985)
m = res["US"].join(res["India"], lsuffix="_US", rsuffix="_IN", how="outer").join(fr.rename(columns={"n": "n_FR", "acc": "acc_FR"}), how="outer")
m["FR_missed_if_US_true"] = (m.n_FR * (m.true_US - m.acc_FR)).round()
print(m.to_string(), flush=True)
m.reset_index().to_csv(sys.argv[1], index=False)
