"""P58: where are the owners of the address-less ambiguous SX? Breakdown by key-group size (no row materialisation).
    probes/P58-twin-cands/py.sh probes/P58-twin-cands/cov58.py [c ...]"""
import sys
import pandas as pd, pyarrow.parquet as pq
from frame58 import W, key, noaddr_sx

for c in sys.argv[1:] or ("India", "US"):
    s1 = pq.read_table(W / "train_source1.parquet", columns=["entity_id", "business_name", "country"],
                       filters=[("country", "=", c)]).to_pandas()
    s1["k"] = [key(x) for x in s1.business_name]
    s1["kfull"] = s1.groupby("k").entity_id.transform("size")
    k1 = s1.set_index("entity_id").k
    sx = noaddr_sx(c)
    sx["k_own"] = [key(x) for x in sx.business_name]
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "am"])
    amk = d[d.am].assign(k=lambda t: k1.reindex(t.s1).to_numpy()).set_index("sx").k
    sx["k"] = amk.reindex(sx.entity_id).fillna(sx.set_index("entity_id").k_own).to_numpy()
    sx["in_dump"] = sx.entity_id.isin(set(d.sx))
    sx["kfull"] = s1.groupby("k").size().reindex(sx.k).fillna(0).to_numpy(int)
    sx = sx[sx.kfull >= 2]
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("sx", "in", list(set(sx.entity_id)))]).to_pandas().set_index("sx").s1
    sx["own"] = gt.reindex(sx.entity_id).to_numpy()
    sx["own_k"] = k1.reindex(sx.own).to_numpy()
    sx["own_in_g"] = sx.own_k == sx.k
    fr = set(zip(d.s1, d.sx))
    sx["own_in_set"] = [(o, x) in fr for o, x in zip(sx.own, sx.entity_id)]
    sx["b"] = pd.cut(sx.kfull, [1, 2, 3, 5, 10, 30, 100, 10**7])
    t = sx.groupby("b", observed=True).agg(n=("own", "size"), owned=("own", lambda v: v.notna().mean()),
                                           in_g=("own_in_g", "mean"), in_set=("own_in_set", "mean"), dump=("in_dump", "mean"))
    t["set|g"] = sx[sx.own_in_g].groupby("b", observed=True).own_in_set.mean()
    print(c, len(sx), "\n", t.round(3).to_string(), flush=True)
