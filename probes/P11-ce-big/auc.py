"""Holdout band AUC of each CE's r (and of the logit sum with m2_big p), per half and country: why a CE wins or loses.
    auc.py <tag> [<tag> ...]      # 'ce5' = #5's CE (P4b cache + P4's extra cache)"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P11-ce-big")
import ce
from sklearn.metrics import roc_auc_score

big = ce._big()
d = pd.read_parquet(ce.PROB, columns=["s1", "sx", "p", "is_true", "country"])
d = d[(d.p > ce.LO) & (d.p < ce.HI) & d.country.isin(["India", "US"])].reset_index(drop=True)
def R(tag):
    if tag == "ce5":
        return big._r(d[["s1", "sx"]])
    return d[["s1", "sx"]].merge(pd.read_parquet(ce.OUT / f"r_{tag}_holdout.parquet"), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
y, lp = d.is_true.to_numpy(), ce.rr.logit(d.p.to_numpy())
print(f"band {len(d):,}, true {y.mean():.3f}, AUC p {roc_auc_score(y, lp):.4f}")
for tag in sys.argv[1:]:
    r = R(tag); lr = ce.rr.logit(r)
    out = [f"{tag}: AUC r {roc_auc_score(y, r):.4f}", f"p+r {roc_auc_score(y, lp + lr):.4f}"]
    for c in ("India", "US"):
        k = (d.country == c).to_numpy()
        out.append(f"{c} r {roc_auc_score(y[k], r[k]):.4f} p+r {roc_auc_score(y[k], lp[k] + lr[k]):.4f}")
    lo = (d.p < 0.5).to_numpy()
    out.append(f"p<.5: r {roc_auc_score(y[lo], r[lo]):.4f}  p>=.5: r {roc_auc_score(y[~lo], r[~lo]):.4f}")
    print("  ".join(out), flush=True)
