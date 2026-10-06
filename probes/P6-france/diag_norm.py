"""Label-free normalisation diagnostics per test country: how often do the P1/rules fields agree between an S1 and
its P3 candidates? France far below US/India on a field = a French normalisation gap (no fitting, counts only)."""
import resource, sys
import numpy as np, pandas as pd

def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

for c in sys.argv[1:]:
    s1 = pd.read_parquet(f"data/work/rf_test_s1_{c}.parquet", columns=["id", "core", "state", "city", "hn", "places"])
    sx = pd.read_parquet(f"data/work/rf_test_sx_{c}.parquet", columns=["id", "core", "state", "city", "hn", "places", "is_web"])
    cand = pd.read_parquet(f"data/work/p3/cands_test_{c}.parquet", columns=["s1", "sx", "block_rank_s1"])
    i = pd.Index(s1.id).get_indexer(cand.s1); j = pd.Index(sx.id).get_indexer(cand.sx)
    del cand["s1"], cand["sx"]
    top = (cand.block_rank_s1 == 0).to_numpy()
    out = {"S1": len(s1), "SX": len(sx), "pairs/S1": len(i) / len(s1)}
    for f in ("core", "city", "hn", "state"):
        a, b = s1[f].to_numpy(object)[i], sx[f].to_numpy(object)[j]
        eq = (a == b) & (a != "")
        out[f"s1_{f}"] = (s1[f] != "").mean(); out[f"sx_{f}"] = (sx[f] != "").mean()
        out[f"top1_{f}_eq"] = eq[top].mean()
        out[f"any_{f}_eq"] = np.bincount(i[eq], minlength=len(s1)).astype(bool).mean()
        del a, b, eq
    print(c, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()), f"peak={peak():.2f}GB", flush=True)
    del s1, sx, cand, i, j
