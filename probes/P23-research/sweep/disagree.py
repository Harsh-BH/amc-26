# Aggregates only: where do the CE (bge pooled) and the GBDT disagree on each country's test band?
import sys, re, numpy as np, pandas as pd
sys.path[:0] = ["/home/harsh1/github-repos/amc-26", "/home/harsh1/github-repos/amc-26/probes/P4b-rerank"]
import rerank as rr
W = "/home/harsh1/github-repos/amc-26/data/work"
LEG = {"France": r"\b(?:sarl|sas|sasu|eurl|sci|sa|ei|snc|eirl)\b", "US": r"\b(?:llc|inc|corp|ltd|co|lp|llp|pllc)\b",
       "India": r"\b(?:pvt|ltd|llp|private|limited|co)\b"}
for c in sys.argv[1:]:
    d = pd.read_parquet(f"{W}/p13/prob_test_{c}_blend.parquet", columns=["s1", "sx", "p_gbdt"])
    d = d[(d.p_gbdt > 0.02) & (d.p_gbdt < 0.98)]
    r = pd.read_parquet(f"{W}/p13/r/r_bge_pool_test_{c}.parquet")
    d = d.merge(r, on=["s1", "sx"], how="inner")
    q1, qx = rr.rows("test", c, d.s1, d.sx)
    ta, tb = rr.texts("test", c, q1, qx)
    na = pd.Series([t.split(" | ")[0] for t in ta]); nb = pd.Series([t.split(" | ")[0] for t in tb])
    aa = pd.Series([t.split(" | ", 1)[1] if " | " in t else "" for t in ta]); ab = pd.Series([t.split(" | ", 1)[1] if " | " in t else "" for t in tb])
    la, lb = na.str.contains(LEG[c]).to_numpy(), nb.str.contains(LEG[c]).to_numpy()
    asym = la != lb
    # address token count asymmetry (proxy for a dropped segment, e.g. missing region)
    ca, cb = aa.str.count(",").to_numpy(), ab.str.count(",").to_numpy()
    seg = ca != cb
    dis = (d.r.to_numpy() > 0.5) != (d.p_gbdt.to_numpy() > 0.5)
    print(f"{c}: band {len(d):,} | disagree {dis.mean():.3f} | legal-asym share {asym.mean():.3f} (disagree {dis[asym].mean():.3f} vs {dis[~asym].mean():.3f})"
          f" | addr-segment-count differs {seg.mean():.3f} (disagree {dis[seg].mean():.3f} vs {dis[~seg].mean():.3f})"
          f" | CE>GBDT share among disagreements {((d.r.to_numpy()>0.5)&dis).sum()/max(dis.sum(),1):.3f}", flush=True)
