"""Test France, label-free: unassigned address-less SX that KEEP a legal form, whose full name matches exactly one France S1 (E),
# Usage (repo root): frcount.py [BASE_DIR, default output/sub76_brand] -> data/work/p80/r1_pool.parquet; then build_lf.py data/work/p80/r1_pool.parquet p80_lfkept
E's core shared by 2-3 France S1 (P66's xfc needed core-unique). How many, and is E already a candidate of x?"""
import re, sys, unicodedata, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from pathlib import Path
W = "data/work/"; SUB = (sys.argv[1].rstrip("/") + "/") if len(sys.argv) > 1 else "output/sub76_brand/"
def key(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
LF = re.compile(r"\b(?:sarl|sas|sasu|sa|eurl|sci|ei|snc|scop|selarl|llc|inc|ltd|pvt|limited|private|corp|pc|llp|plc)\b")
r1 = pq.read_table(W + "rf_test_s1_France.parquet", columns=["id", "core"]).to_pandas()
t = pq.read_table(W + "test_source1.parquet", columns=["entity_id", "business_name", "country"]); t = t.filter(pc.equal(t["country"], "France"))
s1 = pd.DataFrame({"s1": t["entity_id"].to_pylist(), "k": [key(n) for n in t["business_name"].to_pylist()]}).merge(r1.rename(columns={"id": "s1"}), on="s1"); del t
kc, cc = s1.k.value_counts(), s1.core.value_counts(); e_of = dict(zip(s1.k, s1.s1)); core_of = dict(zip(s1.s1, s1.core))
xs = []
for i in (2, 3):
    t = pq.read_table(W + f"test_source{i}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
    m = pc.and_(pc.equal(t["country"], "France"), pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(t["business_address"])), value_set=pa.array(["", "null", "<null>"])))
    idx = pa.array(np.flatnonzero(m.to_numpy(zero_copy_only=False)))
    xs.append(pd.DataFrame({"sx": t["entity_id"].take(idx).to_pylist(), "k": [key(n) for n in t["business_name"].take(idx).to_pylist()], "src": i})); del t
x = pd.concat(xs, ignore_index=True)
x = x[x.k.map(kc).fillna(0) == 1].copy(); x["E"] = x.k.map(e_of); x["ncore"] = x.E.map(core_of).map(cc); x["lf"] = x.k.str.contains(LF)
want = set(x.sx); asg = {}
with open(SUB + "matching_results.tsv") as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip("\n").partition("\t")
        for s in b.split(",") if b else ():
            if s in want: asg[s] = a
x["asg"] = x.sx.map(asg)
Es = set(x.E); cand = {}
with open(SUB + "candidate_pairs.tsv") as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip("\n").partition("\t")
        if a in Es: cand[a] = set(b.split(",")) if b else set()
x["is_cand"] = [s in cand.get(e, ()) for s, e in zip(x.sx, x.E)]
print(x.groupby([x.ncore.clip(upper=4), "lf"]).agg(n=("sx", "size"), assigned=("asg", lambda v: v.notna().mean()),
      unassigned=("asg", lambda v: int(v.isna().sum()))).to_string())
pool = x[x.lf & x.ncore.between(2, 3) & x.asg.isna()]
Path("data/work/p80").mkdir(parents=True, exist_ok=True); pool.to_parquet("data/work/p80/r1_pool.parquet")
print(f"POOL (keeps legal form, core shared by 2-3, unassigned): {len(pool):,}; E already a candidate: {pool.is_cand.mean():.3f}")
