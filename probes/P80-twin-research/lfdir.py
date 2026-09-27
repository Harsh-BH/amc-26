"""Address-less x whose exact name matches ONE S1 (E) while other S1s share E's core (legal-form variants, e.g. 'X' vs 'X PC').
Does direction matter? P(owner = E) when x carries a legal form vs when x is bare. Train truth."""
import re, unicodedata, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work/"
def key(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
LF = re.compile(r"\b(llc|inc|corp|corporation|co|company|ltd|limited|pvt|private|pc|llp|lp|plc|pllc|sarl|sas|sasu|sa|eurl|sci|ei)\b")
out = []
for c in ("US", "India"):
    r1 = pq.read_table(W + f"rf_train_s1_{c}.parquet", columns=["id", "core"]).to_pandas()
    t = pq.read_table(W + "train_source1.parquet", columns=["entity_id", "business_name", "country"])
    t = t.filter(pc.equal(t["country"], c))
    s1 = pd.DataFrame({"s1": t["entity_id"].to_pylist(), "k": [key(n) for n in t["business_name"].to_pylist()]}); del t
    s1 = s1.merge(r1.rename(columns={"id": "s1"}), on="s1"); del r1
    kc = s1.k.value_counts(); cc = s1.core.value_counts()
    e_of = dict(zip(s1.k, s1.s1)); core_of = dict(zip(s1.s1, s1.core))
    for i in (2, 3):
        t = pq.read_table(W + f"train_source{i}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
        m = pc.and_(pc.equal(t["country"], c), pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(t["business_address"])), value_set=pa.array(["", "null", "<null>"])))
        idx = pa.array(np.flatnonzero(m.to_numpy(zero_copy_only=False)))
        out.append(pd.DataFrame({"sx": t["entity_id"].take(idx).to_pylist(), "k": [key(n) for n in t["business_name"].take(idx).to_pylist()], "country": c}))
        del t
    x = out[-1] if False else pd.concat([o for o in out if o.country.iloc[0] == c])
    x = x[x.k.map(kc).fillna(0) == 1].copy()                      # exact name matches exactly one S1 (E)
    x["E"] = x.k.map(e_of); x["ncore"] = x.E.map(core_of).map(cc)
    x = x[x.ncore >= 2]                                            # ... but E's core is shared (legal-form variants exist)
    x["x_has_lf"] = x.k.str.contains(LF)
    out = [o for o in out if o.country.iloc[0] != c] + [x.assign(country=c)]
d = pd.concat([o for o in out if "E" in o.columns])
g = pq.read_table(W + "gt_pairs.parquet"); g = g.filter(pc.is_in(g["sx"], value_set=pa.array(d.sx.tolist()))).to_pandas()
d["owner"] = d.sx.map(dict(zip(g.sx, g.s1))); d["y"] = d.owner == d.E
print(d.groupby(["country", "x_has_lf", d.ncore.clip(upper=4)]).y.agg(["mean", "size"]).round(3).to_string())
print("overall by direction:", d.groupby(["country", "x_has_lf"]).y.agg(["mean", "size"]).round(3).to_dict("index"))
