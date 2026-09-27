"""An SX raw-identical (name AND address) to an S1: who owns it (train)? Does our test output assign it to that identical S1?"""
import sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
W = "data/work/"; split = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else None
t = pq.read_table(W + f"{split}_source1.parquet", columns=["entity_id", "business_name", "business_address", "country"])
h1 = pc.binary_join_element_wise(*[t[c].cast(pa.large_string()) for c in ("business_name", "business_address", "country")], pa.scalar("\t", pa.large_string()))
s1 = dict(zip(h1.to_pylist(), t["entity_id"].to_pylist())); del t, h1
hits = []
for i in (2, 3):
    t = pq.read_table(W + f"{split}_source{i}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
    hx = pc.binary_join_element_wise(*[t[c].cast(pa.large_string()) for c in ("business_name", "business_address", "country")], pa.scalar("\t", pa.large_string()))
    m = pc.is_in(hx, value_set=pa.array(list(s1.keys())))
    idx = pa.array(np.flatnonzero(m.to_numpy(zero_copy_only=False)))
    hits += [(e, s1[h], c) for e, h, c in zip(t["entity_id"].take(idx).to_pylist(), hx.take(idx).to_pylist(), t["country"].take(idx).to_pylist())]
    del t, hx
d = pd.DataFrame(hits, columns=["sx", "s1_same", "country"])
print(f"{split}: SX raw-identical to an S1 (name+address+country): {len(d):,}", d.country.value_counts().to_dict())
if split == "train":
    g = pq.read_table(W + "gt_pairs.parquet"); g = g.filter(pc.is_in(g["sx"], value_set=pa.array(d.sx.tolist()))).to_pandas()
    d["owner"] = d.sx.map(dict(zip(g.sx, g.s1)))
    print("  owned by the identical S1:", round((d.owner == d.s1_same).mean(), 4), " by another S1:", round((d.owner.notna() & (d.owner != d.s1_same)).mean(), 4), " by none:", round(d.owner.isna().mean(), 4))
if sub:
    want = set(d.sx); asg = {}
    with open(sub) as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            for s in b.split(",") if b else ():
                if s in want: asg[s] = a
    d["asg"] = d.sx.map(asg)
    print("  our output: assigned to the identical S1:", d.groupby("country").apply(lambda e: int((e.asg == e.s1_same).sum())).to_dict(),
          " to another S1:", d.groupby("country").apply(lambda e: int((e.asg.notna() & (e.asg != e.s1_same)).sum())).to_dict())
