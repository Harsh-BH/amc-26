"""P58: is any slice of the expanded twin SX above break-even? 2-fold (SX hash) OOS scores of the d58 'raw' / 'rawp' arms;
per SX best twin: precision (best twin owns it) by score bucket; unique raw-closest twin precision (P49 T3, now on full groups).
    probes/P58-twin-cands/py.sh probes/P58-twin-cands/calib58.py"""
import numpy as np, pandas as pd, lightgbm as lgb
import d58

D = {c: d58.load(c) for c in d58.LB_W}
for arm in ("raw", "rawp"):
    cols = d58.ARMS[arm]
    X = pd.concat([ex.assign(c=c) for c, (d, ex) in D.items()], ignore_index=True)
    fo = (pd.util.hash_pandas_object(X.sx, index=False).to_numpy() % 2).astype(int)
    s = np.zeros(len(X))
    for f in (0, 1):
        b = lgb.train(d58.PRM, lgb.Dataset(X.loc[fo != f, cols], X.y[fo != f].to_numpy()), 150)
        s[fo == f] = b.predict(X.loc[fo == f, cols])
    X["s"] = s
    best = X.loc[X.groupby("sx").s.idxmax()]
    best["b"] = pd.cut(best.s, [0, .3, .5, .6, .7, .8, .9, 1])
    t = best.groupby(["c", "b"], observed=True).agg(n=("y", "size"), prec=("y", "mean"), ho=("holdout", "sum"), new=("in_set", lambda v: (v == 0).mean()))
    print(arm, "\n", t.round(3).to_string(), flush=True)
for c, (d, ex) in D.items():
    u = ex[ex.d_min_uniq > 0]
    print(c, f"unique raw-closest twin: {u.sx.nunique():,} SX, owns {u.y.mean():.3f}; given owner in group {u[u.own_in].y.mean():.3f}; "
          f"new pair {(u.in_set == 0).mean():.3f}; holdout {u.holdout.sum():,}", flush=True)
