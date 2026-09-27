"""US/India holdout analog: truth rate and acceptance of argmax pairs by hn class (#8c final p, t 0.71)."""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
W = "/home/harsh1/github-repos/amc-26/data/work"
c = sys.argv[1]
d = pd.read_parquet(f"{W}/p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "p_m2", "am", "pred"])
d = d[d.am & d.holdout]
def rf(side, ids):
    tb = pq.read_table(f"{W}/rf_train_{side}_{c}.parquet", columns=["id", "hn", "core"])
    tb = tb.filter(pc.is_in(tb["id"], value_set=pa.array(ids.unique())))
    return tb.to_pandas().drop_duplicates("id").set_index("id")
d = d.join(rf("s1", d.s1), on="s1").join(rf("sx", d.sx), on="sx", rsuffix="_x")
h1 = d.hn.fillna("").astype(str).str.extract(r"^(\d+)")[0].fillna("").to_numpy()
hx = d.hn_x.fillna("").astype(str).str.extract(r"^(\d+)")[0].fillna("").to_numpy()
def drop1(x, y):
    return len(y) == len(x) - 1 and len(y) > 0 and any(x[:i] + x[i+1:] == y for i in range(len(x)))
cls = np.full(len(d), "hn_missing", object)
for i in np.flatnonzero((h1 != "") & (hx != "")):
    x, y = h1[i], hx[i]
    if x == y: cls[i] = "eq"
    elif drop1(x, y) or drop1(y, x): cls[i] = "digit_dropadd"
    else:
        dd = int(y) - int(x)
        cls[i] = "d12" if abs(dd) in (1, 2) else "sib3" if dd in (3, 4, 5, 7, 9, 11, 13, 21) else "far"
d["cls"] = cls
d["core_eq"] = d.core.fillna("") == d.core_x.fillna("")
g = d.groupby("cls").agg(n=("y", "size"), true_rate=("y", "mean"), acc=("pred", "mean"), p_mean=("p", "mean"),
                         pm2_mean=("p_m2", "mean"), acc_m2_095=("p_m2", lambda s: (s >= 0.95).mean()))
g["share"] = g.n / len(d)
print(c, "holdout-list argmax pairs", len(d))
print(g.round(3).to_string())
# recall-side: share of TRUE argmax pairs in each class that the M2-only rule at t .95 would reject (France regime)
tr = d[d.y]
print("true pairs: class share", tr.cls.value_counts(normalize=True).round(3).to_dict())
print("true pairs rejected by p_m2<0.95 by class", tr.groupby("cls").p_m2.apply(lambda s: round((s < 0.95).mean(), 3)).to_dict())
