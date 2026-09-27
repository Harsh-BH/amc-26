"""Lead 19:20. (1) Broad type-word veto (public repo's v12 definition) vs our N54 lens set, on BASE's France lines:
list = France S1 core tokens with df >= 100, len >= 4, minus proven noise words. A pair is flagged if the SX adds a list word w
with similarity < .5 to every dropped word while dropping >= 1 word (swap), or adds w and drops none (add). Split by word with
lens s / e; the N54 set (s > .25, n >= 20) is already removed from BASE. (2) Noise-suffix pairs that were never candidates:
free SX whose core = an S1 core +/- one noise word or with one word swapped for it, same hn + city, exactly one such S1,
pair missing from BASE's candidates, no legal form added / conflicting.
    tw_audit.py sub81"""
import csv, json, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
from rapidfuzz.distance import Indel
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P68-iu-retrieval"), str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import fr_key, lf  # noqa: E402
import france as FR  # noqa: E402
W = ROOT / "data/work"
FR.C = "France"
NOISE = {"groupe", "developpement", "fils", "associes", "services", "service", "france"}
base = sys.argv[1]
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
lens = FR.lens_table(d.s1.to_numpy(object), d.sx.to_numpy(object))
n54 = set(lens.index[(lens.n >= 20) & (lens.s > 0.25)])
df = collections.Counter(t for c in r1.core.fillna("") for t in set(c.split()))
LIST = {t for t, n in df.items() if n >= 100 and len(t) >= 4} - NOISE
m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
m = m[m.source1_entity_id.isin(r1.index)]
e = m[m.matched_entity_ids != ""].assign(sx=lambda x: x.matched_entity_ids.str.split(",")).explode("sx").rename(columns={"source1_entity_id": "s1"})[["s1", "sx"]]
A, B = r1.core.reindex(e.s1).fillna("").to_numpy(), rx.core.reindex(e.sx).fillna("").to_numpy()
same = ((r1.hn.reindex(e.s1).fillna("").to_numpy() == rx.hn.reindex(e.sx).fillna("").to_numpy()) & (r1.hn.reindex(e.s1).fillna("").to_numpy() != "")
        & (r1.city.reindex(e.s1).fillna("").to_numpy() == rx.city.reindex(e.sx).fillna("").to_numpy()))
rows = []
for i, (a, b) in enumerate(zip(A, B)):
    ta, tb = set(a.split()), set(b.split())
    add, drop = tb - ta, ta - tb
    for w in add & LIST:
        if drop and all(Indel.normalized_similarity(w, x) < .5 for x in drop):
            rows.append((i, w, "swap"))
        elif not drop:
            rows.append((i, w, "add"))
f = pd.DataFrame(rows, columns=["i", "w", "kind"]).drop_duplicates("i")
f["same_addr"], f["in_n54"] = same[f.i], f.w.isin(n54)
f = f.join(lens[["n", "s", "e"]], on="w")
g = f[~f.in_n54]
print(f"(1) {base} France pairs {len(e):,}; list words {len(LIST)}; flagged {len(f):,} (in N54 set {int(f.in_n54.sum())}); surviving N54 {len(g):,}; same-address {int(g.same_addr.sum()):,}")
t = g.groupby("w").agg(pairs=("i", "size"), same_addr=("same_addr", "sum"), swap=("kind", lambda k: int((k == "swap").sum())), lens_n=("n", "first"),
                       lens_s=("s", "first"), lens_e=("e", "first")).sort_values("pairs", ascending=False)
print(t.head(25).round(3).to_string())
dec = g[g.s > 0.15]
print(f"decoy-like (lens s > .15, n >= 20) surviving: {len(dec):,} pairs ({int(dec.same_addr.sum()):,} same address) over {dec.w.nunique()} words; "
      f"s unknown (n < 20 / never swapped): {int(g.s.isna().sum()):,}")
out = e.iloc[g.i.to_numpy()].assign(w=g.w.to_numpy(), kind=g.kind.to_numpy(), same_addr=g.same_addr.to_numpy(), s=g.s.to_numpy(), e=g.e.to_numpy())
out.to_parquet(W / f"p70/tw_{base}.parquet", index=False)
# (2) never-candidate noise-suffix pairs
used = {x for v in m.matched_entity_ids if v for x in v.split(",")}
cand = {}
with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        s, v = line.rstrip("\n").split("\t", 1)
        if s in r1.index and v:
            cand[s] = set(v.split(","))
S1 = r1[(r1.core.fillna("") != "") & (r1.hn.fillna("") != "")].reset_index()
key = lambda toks: " ".join(sorted(toks))
idx = collections.defaultdict(list)              # (tokens-key, hn, city) -> S1 ids; also 'minus one word' keys for swaps
idx1 = collections.defaultdict(list)
for sid, c, h, ct in zip(S1.id, S1.core, S1.hn, S1.city):
    tk = c.split()
    idx[(key(tk), h, ct)].append(sid)
    for j in range(len(tk)):
        idx1[(key(tk[:j] + tk[j + 1:]), h, ct)].append((sid, tk[j]))
FX = rx[~rx.index.isin(used) & (rx.core.fillna("") != "") & (rx.hn.fillna("") != "")]
hits = []
for x, c, h, ct in zip(FX.index, FX.core, FX.hn, FX.city):
    tk = c.split()
    for w in NOISE & set(tk):
        rest = [t for t in tk if t != w]
        for sid in idx.get((key(rest), h, ct), []):
            hits.append((sid, x, w, "sx_adds"))
        for sid, dropped in idx1.get((key(rest), h, ct), []):
            if dropped != w:
                hits.append((sid, x, w, "swap"))
    for w in NOISE:
        for sid in idx.get((key(tk + [w]), h, ct), []):
            hits.append((sid, x, w, "sx_drops"))
h2 = pd.DataFrame(hits, columns=["s1", "sx", "w", "kind"]).drop_duplicates(["s1", "sx"])
h2 = h2[~h2.sx.duplicated(keep=False)]                 # exactly one S1
h2["cand"] = [x in cand.get(s, ()) for s, x in zip(h2.s1, h2.sx)]
n1, nx = fr_key.raw_names("test", set(h2.s1), [1]).reindex(h2.s1), fr_key.raw_names("test", set(h2.sx), [2, 3]).reindex(h2.sx)
h2["lf"] = np.asarray(lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx))))
ok = h2[~h2.lf.isin([3, 4])]
print(f"(2) free noise-suffix SX (same hn + city, one S1): {len(ok):,}; candidates {int(ok.cand.sum()):,}; NEVER candidates {int((~ok.cand).sum()):,}")
print(pd.crosstab([ok.w, ok.kind], ok.cand).to_string())
ok.to_parquet(W / f"p70/noise_nc_{base}.parquet", index=False)
