"""Compact frame of address-less twin-key SX (train) -> scratchpad/amb.parquet + s1tw.parquet. Lean: filter in arrow first."""
import re, unicodedata, sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work/"; OUT = sys.argv[1]
def key(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
t = pq.read_table(W + "train_source1.parquet", columns=["entity_id", "business_name", "country"])
s1 = pd.DataFrame({"s1": t["entity_id"].to_pylist(), "country": t["country"].to_pylist(), "k": [key(n) for n in t["business_name"].to_pylist()]})
del t
s1["row"] = np.arange(len(s1), dtype=np.int32); N1 = len(s1)
s1["K"] = s1.groupby(["country", "k"]).s1.transform("size").astype(np.int32)
gt = pq.read_table(W + "train_ground_truth.parquet").to_pandas()
ntot = pd.Series([0 if m == "" else m.count(",") + 1 for m in gt.matched_entity_ids], index=gt.source1_entity_id.to_numpy())
del gt
s1["ntot"] = s1.s1.map(ntot).astype(np.int16)
s1[["row", "ntot"]].assign(N1=N1).to_parquet(OUT + "/s1all.parquet")          # for the global row-vs-size check
tw = s1[s1.K >= 2].reset_index(drop=True); del s1
xs = []
for i in (2, 3):
    t = pq.read_table(W + f"train_source{i}.parquet", columns=["entity_id", "business_address", "country"])
    a = pc.utf8_lower(pc.utf8_trim_whitespace(t["business_address"]))
    m = pc.is_in(a, value_set=pa.array(["", "null", "<null>"]))
    idx = np.flatnonzero(m.to_numpy(zero_copy_only=False))
    nm = pq.read_table(W + f"train_source{i}.parquet", columns=["business_name"])["business_name"].take(pa.array(idx)).to_pylist()
    xs.append(pd.DataFrame({"sx": t["entity_id"].take(pa.array(idx)).to_pylist(), "country": t["country"].take(pa.array(idx)).to_pylist(),
                            "k": [key(n) for n in nm], "src": i, "row": idx.astype(np.int32), "Nsrc": len(t)}))
    del t, a, m
x = pd.concat(xs, ignore_index=True); del xs
g = pq.read_table(W + "gt_pairs.parquet")
g = g.filter(pc.is_in(g["sx"], value_set=pa.array(x.sx.tolist()))).to_pandas()   # owners of all address-less SX
x["owner"] = x.sx.map(dict(zip(g.sx, g.s1)))
nna = g.s1.value_counts()                                                        # address-less copies per S1
tw["nna"] = tw.s1.map(nna).fillna(0).astype(np.int16); tw["naddr"] = tw.ntot - tw.nna
x = x.merge(tw.groupby(["country", "k"]).s1.apply(list).rename("tw").reset_index(), on=["country", "k"])
x["own_in"] = [o in t for o, t in zip(x.owner, x.tw)]
x.to_parquet(OUT + "/amb.parquet"); tw.to_parquet(OUT + "/s1tw.parquet")
print(len(x), "ambiguous address-less SX;", len(tw), "twin S1;", "own_in", round(x.own_in.mean(), 3))
