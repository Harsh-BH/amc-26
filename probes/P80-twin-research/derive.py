"""Copy-of-copy test: an ambiguous address-less x whose RAW name is noised (≠ every twin S1's raw name) and equals the raw
name of an addressed copy at exactly one twin — does that twin own x? Under iid noise it's a coincidence (~uniform)."""
import sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
S = sys.argv[1]; W = "data/work/"
x = pd.read_parquet(S + "/amb.parquet"); x = x[x.owner.notna()]
tw = pd.read_parquet(S + "/s1tw.parquet")
n1 = pq.read_table(W + "train_source1.parquet", columns=["business_name"])["business_name"]
tw["raw"] = n1.take(pa.array(tw.row.to_numpy())).to_pylist(); del n1
raw1 = dict(zip(tw.s1, tw.raw))
names = {}
for i in (2, 3):
    xi = x[x.src == i]
    nm = pq.read_table(W + f"train_source{i}.parquet", columns=["business_name"])["business_name"]
    names.update(zip(xi.sx, nm.take(pa.array(xi.row.to_numpy())).to_pylist())); del nm
x["raw"] = x.sx.map(names)
# addressed copies of twin S1s that appear in some x's twin list
need = set(s for t in x.tw for s in t)
g = pq.read_table(W + "gt_pairs.parquet"); g = g.filter(pc.is_in(g["s1"], value_set=pa.array(list(need)))).to_pandas()
cop = []
for i in (2, 3):
    t = pq.read_table(W + f"train_source{i}.parquet", columns=["entity_id", "business_name", "business_address"])
    gi = g[g.sx.str.startswith(f"S{i}-")]
    idx = pc.index_in(pa.array(gi.sx.tolist()), value_set=t["entity_id"]).to_numpy(zero_copy_only=False)
    ad = pc.utf8_lower(pc.utf8_trim_whitespace(t["business_address"].take(pa.array(idx)))).to_pylist()
    cop.append(pd.DataFrame({"s1": gi.s1.to_numpy(), "raw": t["business_name"].take(pa.array(idx)).to_pylist(),
                             "addr": [a not in ("", "null", "<null>") for a in ad]})); del t
cop = pd.concat(cop); cop = cop[cop.addr]
by = cop.groupby("s1").raw.apply(set).to_dict()
rows = []
for sx_, r, t, o in zip(x.sx, x.raw, x.tw, x.owner):
    noised = all(r != raw1[s] for s in t)
    hit = [s for s in t if r in by.get(s, ())]
    rows.append((noised, len(hit), hit[0] == o if len(hit) == 1 else None, len(t), o in t))
d = pd.DataFrame(rows, columns=["noised", "nhit", "hit_own", "K", "own_in"])
print("x noised (raw ≠ every twin raw):", round(d.noised.mean(), 3), " n", len(d))
for nz in (True, False):
    e = d[(d.noised == nz) & (d.nhit == 1)]
    print(f"noised={nz}: unique twin with an addressed copy of identical raw name: n {len(e)}; that twin owns x {e.hit_own.mean():.3f}; "
          f"uniform baseline {(1 / e.K).mean():.3f}; own_in {e.own_in.mean():.3f}")
