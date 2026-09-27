"""Label-free: hn relation class of argmax pairs per country, counts per 1k S1 and accepted share at the shipped t."""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
W = "/home/harsh1/github-repos/amc-26/data/work"
c, path, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = pq.read_table(f"{W}/{path}", columns=["s1", "sx", "p"]).to_pandas()
mx = d.groupby("sx").p.transform("max")
d = d[d.p >= mx].drop_duplicates("sx")
n_s1 = pq.ParquetFile(f"{W}/rf_test_s1_{c}.parquet").metadata.num_rows
def rf(side, ids):
    tb = pq.read_table(f"{W}/rf_test_{side}_{c}.parquet", columns=["id", "hn", "core"])
    tb = tb.filter(pc.is_in(tb["id"], value_set=pa.array(ids.unique())))
    return tb.to_pandas().drop_duplicates("id").set_index("id")
a, b = rf("s1", d.s1), rf("sx", d.sx)
d = d.join(a, on="s1").join(b, on="sx", rsuffix="_x")
h1 = d.hn.fillna("").astype(str).str.extract(r"^(\d+)")[0].fillna("")
hx = d.hn_x.fillna("").astype(str).str.extract(r"^(\d+)")[0].fillna("")
def drop1(x, y):   # y = x with one digit deleted
    return len(y) == len(x) - 1 and len(y) > 0 and any(x[:i] + x[i+1:] == y for i in range(len(x)))
cls = np.full(len(d), "hn_missing", object)
both = (h1 != "") & (hx != "")
H1, HX = h1.to_numpy(), hx.to_numpy()
for i in np.flatnonzero(both.to_numpy()):
    x, y = H1[i], HX[i]
    if x == y: cls[i] = "eq"
    elif drop1(x, y) or drop1(y, x): cls[i] = "digit_dropadd"
    else:
        dd = int(y) - int(x)
        cls[i] = "d12" if abs(dd) in (1, 2) else "sib3" if dd in (3, 4, 5, 7, 9, 11, 13, 21) else "far"
d["cls"] = cls
d["core_eq"] = d.core.fillna("") == d.core_x.fillna("")
g = d.groupby("cls").agg(n=("p", "size"), p_mean=("p", "mean"), acc=("p", lambda s: (s >= t).mean()),
                         core_eq=("core_eq", "mean"))
g["per_1kS1"] = g.n / n_s1 * 1000
g["acc_per_1kS1"] = g.n * g.acc / n_s1 * 1000
print(c, "argmax pairs", len(d), "per S1", round(len(d) / n_s1, 3), "t", t)
print(g.round(3).to_string())
for k in ("digit_dropadd", "d12", "eq"):
    s = d[d.cls == k].p
    print(k, "p deciles", np.round(s.quantile([.1, .25, .5, .75, .9]).values, 3).tolist())
