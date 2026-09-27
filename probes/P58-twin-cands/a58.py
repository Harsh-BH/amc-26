"""P58 quick signal check on ex_{c}: (a) twin S1 address status, (b) SX-name tokens outside the group key that occur in
the twin S1's address. Owner rate per row class, SX-level uniqueness.  probes/P58-twin-cands/py.sh probes/P58-twin-cands/a58.py"""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from frame58 import W, key, NA

for c in sys.argv[1:] or ("India", "US"):
    ex = pd.read_parquet(W / f"p58/ex_{c}.parquet", columns=["sx", "s1", "sx_name", "y", "kfull", "own_in", "in_set", "holdout"])
    s1 = pq.read_table(W / "train_source1.parquet", columns=["entity_id", "business_name", "business_address"],
                       filters=[("entity_id", "in", list(set(ex.s1)))]).to_pandas().set_index("entity_id")
    a = s1.business_address.reindex(ex.s1).fillna("")
    ex["s1_na"] = a.str.strip().str.upper().isin(NA).to_numpy()
    ex["s1_null"] = a.str.strip().str.upper().isin({"NULL", "<NULL>"}).to_numpy()
    gk = set()
    kk = [set(key(n).split()) for n in s1.business_name.reindex(ex.s1)]
    xs = [set(key(n).split()) for n in ex.sx_name]
    at = [set(key(v).split()) for v in a]
    extra = [x - k for x, k in zip(xs, kk)]
    ex["n_extra"] = [len(e) for e in extra]
    ex["hit"] = [len(e & t) for e, t in zip(extra, at)]
    o = ex[ex.own_in]
    for col in ("s1_na", "s1_null"):
        t = o.groupby(col).y.agg(["mean", "size"])
        sxl = o.groupby("sx")[col].transform("sum")
        u = o[(o[col]) & (sxl == 1)]
        print(c, col, t.round(3).to_dict(), f"| SX with exactly one such twin: {u.sx.nunique():,}, it owns {u.y.mean():.3f}", flush=True)
    t = o.groupby(o.hit > 0).y.agg(["mean", "size"])
    hx = o.groupby("sx").hit.transform(lambda v: (v > 0).sum())
    u = o[(o.hit > 0) & (hx == 1)]
    print(c, "extra-token hit", t.round(3).to_dict(), f"| SX with extra tokens {o[o.n_extra > 0].sx.nunique():,}; "
          f"unique hit twin {u.sx.nunique():,}, owns {u.y.mean():.3f}; touching holdout {u[u.holdout].sx.nunique():,}", flush=True)
    # unconditional (owner may be outside the group)
    ua = ex[(ex.hit > 0) & (ex.groupby("sx").hit.transform(lambda v: (v > 0).sum()) == 1)]
    print(c, f"unconditional unique hit: {ua.sx.nunique():,} SX, owns {ua.y.mean():.3f}", flush=True)
