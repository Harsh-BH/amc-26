"""P50 A: French house-number suffixes / ranges. How N.house_number treats them, how many France records and chain
pairs carry them, and where they sit (accepted in sub44r or not, chain p band, hn_eq / sib flags).
    .venv/bin/python probes/P50-idea-batch/a_hn.py"""
import re, sys, json, collections
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path[:0] = ["code/business_entity_resolution/src"]
import normalize as N
import features as F

# 1. parser behaviour on the patterns
for x in ["12 bis Rue X, Lille", "12bis Rue X", "12 ter Rue X", "12-14 Rue X", "12 à 14 Rue X", "12/14 Rue X",
          "16 d Rue de Maugis", "12A Rue X", "N°16 Allee Du Corail", "(30) R. PEIRRE BAUMANN", "12 et 14 rue X"]:
    print(f"{x!r:32} -> {N.house_number(pd.Series([x])).iloc[0]!r}")

SUF = re.compile(r"^(\d+)\s*(bis|ter|quater|[a-d])\b")             # 12 bis / 12bis / 12 b
RNG = re.compile(r"^(\d+)\s*(?:-|/|a|au|et|&)\s*(\d+)\b")          # 12-14 / 12/14 / 12 a 14 (after fold)


def seg(x: str) -> str:
    """The segment N.house_number reads its number from (label-stripped), '' if none."""
    for s in x.split(","):
        if N._UNIT.match(s.strip()):
            continue
        s = N._HN_LABEL.sub("", s.strip().lstrip("#").strip())
        if re.match(r"[a-z]?[-/]?\d", s):
            return s
    return ""


def kinds(addr: pd.Series):
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    s = f.map(seg)
    suf = s.str.extract(SUF.pattern)
    rng = s.str.extract(RNG.pattern)
    return suf[1].fillna("").to_numpy(object), rng[1].fillna("").to_numpy(object), s


id1, idx = F.ids("test", "France")
res = {}
addr = {}
for name, ids, srcs in (("s1", id1, (1,)), ("sx", idx, (2, 3))):
    a = pd.concat([pq.read_table(f"data/work/test_source{i}.parquet", columns=["entity_id", "business_address"],
                                 filters=[("country", "==", "France")]).to_pandas() for i in srcs])
    a = a.set_index("entity_id").business_address.reindex(ids.to_pylist()).fillna("")
    suf, rng_end, s = kinds(a)
    addr[name] = (suf, rng_end)
    c = collections.Counter(suf)
    res[name] = {"n": len(a), "suffix": {k: v for k, v in c.items() if k}, "range": int((rng_end != "").sum())}
    print(name, res[name], flush=True)
    ex = s[(suf != "") | (rng_end != "")].head(0)
    del a, s

# 2. chain pairs touched, and whether the current decision accepts them
z = np.load("data/work/p42/france.npz")
q1, qx, p = z["s1"], z["sx"], z["p10"]
s1ids = pd.read_parquet("data/work/rf_test_s1_France.parquet", columns=["id"]).id.to_numpy()
assert (s1ids == np.asarray(id1.to_pylist(), dtype=object)).all(), "rf s1 order != F.ids order"
acc = set()
fr = set(s1ids)
with open("output/sub44r/matching_results.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s, v = line.rstrip("\n").split("\t", 1)
        if v and s in fr:
            acc.update((s, x) for x in v.split(","))
sxs = np.asarray(idx.to_pylist(), dtype=object)
a_ = np.fromiter(((s1ids[i], sxs[j]) in acc for i, j in zip(q1, qx)), bool, len(q1))
print("chain pairs", len(q1), "accepted in sub44r", int(a_.sum()), "of", len(acc), flush=True)
h1 = pd.read_parquet("data/work/rf_test_s1_France.parquet", columns=["hn"]).hn.to_numpy(object)
hx = pd.read_parquet("data/work/p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(sxs).fillna("").to_numpy(object)
H1, HX = h1[q1], hx[qx]
s1suf, s1rng = addr["s1"][0][q1], addr["s1"][1][q1]
sxsuf, sxrng = addr["sx"][0][qx], addr["sx"][1][qx]
band = pd.cut(p, [-1, 0.5, 0.8, 0.9, 0.97, 0.99, 2], labels=["<.5", ".5-.8", ".8-.9", ".9-.97", ".97-.99", ">=.99"])
rows = {}
def tab(name, m):
    d = pd.DataFrame({"band": band[m], "acc": a_[m], "hn_eq": z["hn_eq"][m], "sib": z["sib"][m]})
    rows[name] = {"pairs": int(m.sum()), "accepted": int(a_[m].sum()), "hn_eq": int(z["hn_eq"][m].sum()),
                  "sib": int(z["sib"][m].sum()),
                  "by_band": d.groupby("band", observed=False).agg(n=("acc", "size"), acc=("acc", "sum")).to_dict("index")}
    print(name, rows[name], flush=True)
tab("any_suffix", (s1suf != "") | (sxsuf != ""))
tab("suffix_one_side_hn_eq", ((s1suf != "") ^ (sxsuf != "")) & z["hn_eq"])
tab("suffix_both_differ", (s1suf != "") & (sxsuf != "") & (s1suf != sxsuf) & z["hn_eq"])
tab("any_range", (s1rng != "") | (sxrng != ""))
# a range on one side whose OTHER end equals the other side's hn: the parser calls it a mismatch today
end_eq = ((s1rng != "") & (s1rng == HX) & (H1 != HX)) | ((sxrng != "") & (sxrng == H1) & (H1 != HX))
tab("range_other_end_eq", end_eq)
json.dump({"records": res, "pairs": rows}, open("data/work/p50/a_hn.json", "w"), indent=1, default=str)
