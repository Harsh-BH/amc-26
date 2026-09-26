"""P35 a2: precision (holdout rows only) of the pairs each twin rule would touch, by p bin and twin strength."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load
from twins import twin_frame, T

for c in ("India", "US"):
    S = load(c); d = S["d"]
    am = d.am.to_numpy()
    tw = twin_frame(d, am, 0.71, 0.85)
    k = d.merge(tw, on=["s1", "sx"], how="left")
    k["has"] = k.tw_p.notna()
    # per SX: twin-supported S1 count, and does the SX's argmax have support
    k["n_sup"] = k.groupby("sx").has.transform("sum")
    k["am_sup"] = k.groupby("sx").has.transform(lambda s: False) if False else \
        k.sx.map(k[k.am & k.has].drop_duplicates("sx").set_index("sx").has).fillna(False).astype(bool)
    h = k[k.holdout & k.noaddr]
    print(f"\n==== {c}: address-less holdout pairs {len(h):,}, with twin support {int(h.has.sum()):,}")
    A = h[h.am & ~h.pred & h.has]
    print("A raise (am, p<t, supported):", len(A), "prec", round(A.y.mean(), 3))
    print(A.groupby([pd.cut(A.p, [0, .1, .3, .5, .71]), A.n_sup.clip(upper=2)], observed=True).y.agg(["size", "mean"]).round(3).to_string())
    print(" by tw_p", A.groupby(pd.cut(A.tw_p, [.7, .9, .99, .999, 1]), observed=True).y.agg(["size", "mean"]).round(3).to_dict())
    print(" by tw_sim", A.groupby(pd.cut(A.tw_sim, [.84, .95, .999, 1]), observed=True).y.agg(["size", "mean"]).round(3).to_dict())
    print(" by tw_x", A.groupby("tw_x").y.agg(["size", "mean"]).round(3).to_dict())
    B = h[~h.am & h.has]
    print("B switch (non-am, supported):", len(B), "prec", round(B.y.mean(), 3))
    print(B.groupby([B.am_sup, B.n_sup.clip(upper=3)]).y.agg(["size", "mean"]).round(3).to_string())
    print(B[~B.am_sup].groupby(pd.cut(B.p[~B.am_sup], [0, .1, .3, .5, .71, 1]), observed=True).y.agg(["size", "mean"]).round(3).to_string())
    C = h[h.pred & ~h.has]
    print("C guard (pred, unsupported):", len(C), "prec", round(C.y.mean(), 3))
    print(C.groupby(C.n_sup.clip(upper=2)).y.agg(["size", "mean"]).round(3).to_string())
    P = h[h.pred & h.has]
    print("pred & supported:", len(P), "prec", round(P.y.mean(), 3), "| pred & multi-sup", round(P.y[P.n_sup > 1].mean(), 3), int((P.n_sup > 1).sum()))
