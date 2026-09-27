"""N69b: free address-less SX in the #8c holdout dump: truth of the SX's argmax S1 by its p and the gap to its 2nd S1,
plus the holdout dF of adding it (caps ignored; per half)."""
import numpy as np, pandas as pd
def fb(tp, npred, nt):
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0); R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
    f = np.where((P + R) > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0)
    return np.where((nt == 0) & (npred == 0), 1.0, f)
for c in ("India", "US"):
    ex = pd.read_parquet(f"data/work/n69/ex_train_{c}.parquet")
    d = pd.read_parquet(f"data/work/p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "pred", "p", "y", "am"])
    free = set(ex.sx[ex.pred_s1 == ""])
    z = d[d.sx.isin(free)].sort_values(["sx", "p"], ascending=[True, False])
    z["rk"] = z.groupby("sx").cumcount()
    top = z[z.rk == 0].set_index("sx"); sec = z[z.rk == 1].set_index("sx").p
    top["p2"] = sec.reindex(top.index).fillna(0); top["ncand"] = z.groupby("sx").size()
    top = top[top.holdout]
    top["gap"] = top.p - top.p2
    print(f"== {c}: free addr-less SX with a holdout top-1: {len(top):,}  top1 true {top.y.mean():.3f}")
    top["pb"] = pd.cut(top.p, [0, .05, .2, .4, .6, .8, 1.01]); top["gb"] = pd.cut(top.p2, [-.01, .01, .05, .2, 1.01])
    print(top.pivot_table(index="pb", columns="gb", values="y", aggfunc=["mean", "size"], observed=True).round(3).to_string())
    # holdout dF of adding top-1 for rule p>=a & p2<b
    s = pd.read_parquet(f"data/work/p23/s1ctx_{c}.parquet"); idx = s.index; nt = s.n_true.to_numpy(); n = len(s)
    h = d[d.holdout]; pos = idx.get_indexer(h.s1); y = h.y.to_numpy(); pred = h.pred.to_numpy()
    per = lambda pr: fb(np.bincount(pos[pr & y], minlength=n), np.bincount(pos[pr], minlength=n), nt)
    f0 = per(pred); key = pd.MultiIndex.from_arrays([h.s1, h.sx]); hv = np.arange(n) % 2
    for a, b in ((.2, .01), (.3, .01), (.4, .02), (.3, .05), (.5, .05), (.5, .2), (.6, .2)):
        r = top[(top.p >= a) & (top.p2 < b)]
        add = key.isin(pd.MultiIndex.from_arrays([r.s1, r.index]))
        dF = per(pred | add) - f0
        print(f"   p>={a} p2<{b}: adds {add.sum():5d} prec {y[add].mean():.3f}  dF {dF.mean():+.6f} (h0 {dF[hv==0].mean():+.6f} h1 {dF[hv==1].mean():+.6f})")
