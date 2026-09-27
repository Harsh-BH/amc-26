"""Test side, label-free: address-less SX whose name key is shared by >=2 test S1 of the country — how many does sub77 assign?"""
import re, unicodedata, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work/"
def key(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
t = pq.read_table(W + "test_source1.parquet", columns=["entity_id", "business_name", "country"])
s1 = pd.DataFrame({"s1": t["entity_id"].to_pylist(), "country": t["country"].to_pylist(), "k": [key(n) for n in t["business_name"].to_pylist()]}); del t
K = s1.groupby(["country", "k"]).size().rename("K").reset_index()
s1 = s1.merge(K, on=["country", "k"])
xs = []
for i in (2, 3):
    t = pq.read_table(W + f"test_source{i}.parquet", columns=["entity_id", "business_address", "country"])
    m = pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(t["business_address"])), value_set=pa.array(["", "null", "<null>"]))
    idx = pa.array(np.flatnonzero(m.to_numpy(zero_copy_only=False)))
    nm = pq.read_table(W + f"test_source{i}.parquet", columns=["business_name"])["business_name"].take(idx).to_pylist()
    xs.append(pd.DataFrame({"sx": t["entity_id"].take(idx).to_pylist(), "country": t["country"].take(idx).to_pylist(), "k": [key(n) for n in nm]})); del t
x = pd.concat(xs, ignore_index=True).merge(K, on=["country", "k"], how="left").fillna({"K": 0})
want = set(x.sx); asg = {}
with open("output/sub76_brand/matching_results.tsv") as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip("\n").partition("\t")
        for s in b.split(",") if b else ():
            if s in want: asg[s] = a
x["s1"] = x.sx.map(asg); kk = dict(zip(s1.s1, s1.k))
x["to_twin"] = [isinstance(a, str) and kk.get(a) == k for a, k in zip(x.s1, x.k)]
x["cls"] = np.select([x.K >= 2, x.K == 1], ["twin key (K>=2)", "unique key"], "no S1 with that key")
r = x.groupby(["country", "cls"]).agg(n=("sx", "size"), assigned=("s1", lambda v: v.notna().mean()), to_same_key=("to_twin", "mean"))
print(r.round(3).to_string())
u = x[(x.K >= 2) & x.s1.isna()]
print("UNASSIGNED twin-key address-less SX:", u.groupby("country").size().to_dict(), " by K:", u.groupby(["country", u.K.clip(upper=5)]).size().unstack().to_dict("index"))
