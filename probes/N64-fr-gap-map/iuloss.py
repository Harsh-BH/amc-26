# N64: India/US holdout loss decomposition on P23's #8c-era dump (pred = shipped decision). Units = S1-F lost.
import numpy as np, pandas as pd
def fb(tp, npred, ntrue):
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0); R = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0)
    f = np.where((P + R) > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0)
    return np.where((ntrue == 0) & (npred == 0), 1.0, f)
for c in ("India", "US"):
    d = pd.read_parquet(f"data/work/p23/pairs_{c}.parquet"); d = d[d.holdout]
    s = pd.read_parquet(f"data/work/p23/s1ctx_{c}.parquet")
    pr = d[d.pred]; tp = pr[pr.y].groupby("s1").size().reindex(s.index).fillna(0).to_numpy(); npred = pr.groupby("s1").size().reindex(s.index).fillna(0).to_numpy()
    nt = s.n_true.to_numpy(); f = fb(tp, npred, nt); n = len(s)
    fp = npred - tp; fn = nt - tp
    print(f"== {c}: S1 {n}  F {f.mean():.5f} (s1ctx {s.f05.mean():.5f}) loss/S1 {1-f.mean():.5f}")
    f_nofp = fb(tp, tp, nt); f_nofn = fb(tp + 0, npred, tp + 0)  # no FN: pretend ntrue = tp (only when tp>0 or nt==0 handled)
    print(f"   remove all FP -> {f_nofp.mean():.5f} (+{f_nofp.mean()-f.mean():.5f}); recover all FN -> {fb(nt+fp*0, npred-fp+fn+fp, nt).mean():.5f}")
    we = (npred == 0) & (nt > 0); wn = (npred > 0) & (tp == 0) & (nt == 0); wrongall = (npred > 0) & (tp == 0) & (nt > 0)
    print(f"   wrong-empty {we.mean():.4f} (loss {we.mean():.5f}) | singleton predicted {wn.mean():.4f} | all-wrong nonempty {wrongall.mean():.4f}")
    part = ~we & ~wn & ~wrongall
    print(f"   partial loss (fp>0 or fn>0 w/ some tp): {(1-f)[part].sum()/n:.5f}  of which fp>0 only {(1-f)[part & (fp>0) & (fn==0)].sum()/n:.5f}, fn>0 only {(1-f)[part & (fn>0) & (fp==0)].sum()/n:.5f}, both {(1-f)[part & (fn>0) & (fp>0)].sum()/n:.5f}")
    # FN location
    fnp = d[d.y & ~d.pred]; print(f"   FN pairs/S1 {fn.sum()/n:.4f}: in dump {len(fnp)/n:.4f} (argmax {fnp.am.mean():.2f}, p>=.5 {(fnp.p>=.5).mean():.2f}), outside dump {(fn.sum()-len(fnp))/n:.4f}")
