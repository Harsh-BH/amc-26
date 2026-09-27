"""Per-record modification budget? Compare name corruption rates of copies whose address was dropped vs kept (train truth, 2M-pair sample)."""
import re, unicodedata, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work/"
def key(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
LF = re.compile(r"\b(llc|inc|corp|co|ltd|pvt|private|limited|pc|llp|lp|plc|sarl|sas|sasu|sa|eurl|sci|ei)\b")
g = pq.read_table(W + "gt_pairs.parquet")
g = g.take(pa.array(np.sort(np.random.default_rng(0).choice(len(g), 2_000_000, replace=False))))
t1 = pq.read_table(W + "train_source1.parquet", columns=["entity_id", "business_name", "business_address"])
i1 = pc.index_in(g["s1"], value_set=t1["entity_id"])
n1, a1 = t1["business_name"].take(i1).to_pylist(), t1["business_address"].take(i1).to_pylist(); del t1
nx, ax = [None] * len(g), [None] * len(g)
sxl = g["sx"].to_pylist()
for i in (2, 3):
    t = pq.read_table(W + f"train_source{i}.parquet", columns=["entity_id", "business_name", "business_address"])
    pos = np.flatnonzero(np.char.startswith(np.array(sxl, dtype="U3"), f"S{i}-"))
    idx = pc.index_in(pa.array([sxl[p] for p in pos]), value_set=t["entity_id"])
    for p, n, a in zip(pos, t["business_name"].take(idx).to_pylist(), t["business_address"].take(idx).to_pylist()):
        nx[p], ax[p] = n, a
    del t
d = pd.DataFrame({"na": [a.strip().lower() in ("", "null", "<null>") for a in ax]})
k1 = [key(n) for n in n1]; kx = [key(n) for n in nx]
d["raw_eq"] = [a == b for a, b in zip(n1, nx)]
d["key_eq"] = [a == b for a, b in zip(k1, kx)]
d["lf_drop"] = [bool(LF.search(a)) and not LF.search(b) for a, b in zip(k1, kx)]
d["web"] = [("www" in b or ".com" in b or ".in" in b or ".fr" in b) for b in (n.lower() for n in nx)]
d["case"] = [a != b and a.lower() == b.lower() for a, b in zip(n1, nx)]
d["addr_eq"] = [a == b for a, b in zip(a1, ax)]
r = d.groupby("na")[["raw_eq", "key_eq", "lf_drop", "web", "case"]].mean().T
r.columns = ["address kept", "address dropped"]; r["ratio"] = r["address dropped"] / r["address kept"]
print(f"n {len(d):,}, address dropped {d.na.mean():.4f}"); print(r.round(4).to_string())
print("P(addr exactly kept | name raw-equal) vs (| name changed):", round(d.addr_eq[d.raw_eq].mean(), 4), round(d.addr_eq[~d.raw_eq].mean(), 4))
