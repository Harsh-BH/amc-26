"""N66: India/US empty-S1 rescue. An empty S1 pays for a wrong add only if it is a true singleton, so its best pair
deserves a lower bar than the global t. Holdout (#8c-era P23 dump): add the best argmax pair (p >= tau) to empty S1."""
import numpy as np, pandas as pd
def fb(tp, npred, nt):
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0); R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
    f = np.where((P + R) > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0)
    return np.where((nt == 0) & (npred == 0), 1.0, f)
for c in ("India", "US"):
    d = pd.read_parquet(f"data/work/p23/pairs_{c}.parquet"); d = d[d.holdout]
    s = pd.read_parquet(f"data/work/p23/s1ctx_{c}.parquet"); idx = s.index
    pos = idx.get_indexer(d.s1); y = d.y.to_numpy(); pred = d.pred.to_numpy(); nt = s.n_true.to_numpy(); n = len(s)
    per = lambda pr: fb(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), nt)
    f0 = per(pred)
    empty = np.bincount(pos[pred], minlength=n) == 0
    owned = set(d.sx[pred])
    c_ = d[empty[pos] & d.am.to_numpy() & ~d.sx.isin(owned).to_numpy()].sort_values("p", ascending=False).drop_duplicates("s1")
    print(f"== {c}: empty S1 {empty.sum()} ({empty.mean():.4f}), truth-singleton among them {(nt[empty]==0).mean():.3f}; with a free argmax cand {len(c_)}")
    hv = (np.arange(n) % 2)
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6):
        add = np.zeros(len(d), bool); k = c_[c_.p >= tau].index; add[d.index.get_indexer(k)] = True
        dF = per(pred | add) - f0
        print(f"  tau {tau}: adds {add.sum():5d} prec {y[add].mean():.3f}  dF {dF.mean():+.6f}  (h0 {dF[hv==0].mean():+.6f} h1 {dF[hv==1].mean():+.6f})")
