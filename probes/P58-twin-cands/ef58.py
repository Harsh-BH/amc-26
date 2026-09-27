"""P58: expected-F0.5 decode of the expanded twin SX (d58.apply_ef), OOS scores per half (as d58.run), (lam, ps) tuned on
the fit half. Also P(singleton | base-empty) on the holdout.   probes/P58-twin-cands/py.sh probes/P58-twin-cands/ef58.py [arm]"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
import d58
from d58 import fb, LB_W, W

arm = sys.argv[1] if len(sys.argv) > 1 else "rawp"
cols = d58.ARMS[arm]
D = {c: d58.load(c) for c in LB_W}
for c, (d, ex) in D.items():
    e = d["npred"] == 0
    print(c, f"base-empty holdout S1 {e.mean():.4f}, singleton among them {(d['n_true'][e] == 0).mean():.3f}", flush=True)
S = {}
for rep in (0, 1):
    m = {c: ~ex[f"h{rep}"].to_numpy() for c, (d, ex) in D.items()}
    b = lgb.train(d58.PRM, lgb.Dataset(pd.concat([ex.loc[m[c], cols] for c, (d, ex) in D.items()]),
                                       np.concatenate([ex.y.to_numpy()[m[c]] for c, (d, ex) in D.items()])), 150)
    for c, (d, ex) in D.items():
        S[(c, rep)] = b.predict(ex[cols])
res, out = {}, {}
LAM, PS = [0.0, 0.25, 0.5, 1.0], [0.3, 0.5, 0.7, 0.9, 1.0]
for rep in (1, 0):
    fit = 1 - rep
    for c, (d, ex) in D.items():
        f0 = fb(d["tp"], d["npred"], d["n_true"])
        sf, sr = ex[f"h{fit}"].to_numpy(), ex[f"h{rep}"].to_numpy()
        subf, subr = ex[sf].reset_index(drop=True), ex[sr].reset_index(drop=True)
        gv = {(l, p): (d58.apply_ef(d, subf, S[(c, fit)][sf], l, p) - f0)[d["half"] == fit].mean() for l in LAM for p in PS}
        bt = max(gv, key=gv.get)
        dl = (d58.apply_ef(d, subr, S[(c, rep)][sr], *bt) - f0)[d["half"] == rep]
        res[(c, rep)] = (dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl)), bt, int((dl > 0).sum()), int((dl < 0).sum()))
for rep in (1, 0):
    lb = sum(LB_W[c] * res[(c, rep)][0] for c in LB_W)
    se = np.sqrt(sum((LB_W[c] * res[(c, rep)][1]) ** 2 for c in LB_W))
    print(f"[ef-{arm}] fit h{1 - rep} -> report h{rep}: " + " | ".join(
        f"{c} {res[(c, rep)][0]:+.6f} ± {res[(c, rep)][1]:.6f} ((lam, ps) {res[(c, rep)][2]}, +{res[(c, rep)][3]}/-{res[(c, rep)][4]})"
        for c in LB_W) + f" | LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:.1f} SE)", flush=True)
    out[f"ef_{arm}_h{rep}"] = dict(lb=lb, se=se, **{c: list(res[(c, rep)]) for c in LB_W})
json.dump(out, open(W / f"p58/ef58_{arm}.json", "w"), indent=1, default=float)
