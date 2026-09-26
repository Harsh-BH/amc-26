"""P35 a1: how often does a confident addressed twin under a candidate S1 of an address-less SX point at its true S1?
Per address-less holdout-competing SX: outcome of #8c (TP / lowp FN / stolen FN / FP / TN-empty) x twin presence."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load
from twins import twin_frame, T

for c in ("India", "US"):
    S = load(c); d = S["d"]
    am = d.am.to_numpy()
    for tau1, ts in ((0.9, 0.9), (0.9, 1.0), (0.71, 0.85)):
        tw = twin_frame(d, am, tau1, ts)
        k = d.merge(tw, on=["s1", "sx"], how="left")
        k["has"] = k.tw_p.notna()
        a = k[k.noaddr]
        # per address-less SX: does its true S1 (if candidate) have twin support? does its argmax?
        g = a.groupby("sx")
        sxf = pd.DataFrame({"any_y": g.y.any(), "am_y": g.apply(lambda x: (x.am & x.y).any(), include_groups=False),
                            "pred": g.pred.any(), "pred_y": g.apply(lambda x: (x.pred & x.y).any(), include_groups=False),
                            "tw_true": g.apply(lambda x: (x.has & x.y).any(), include_groups=False),
                            "tw_any": g.has.any(), "tw_am": g.apply(lambda x: (x.has & x.am).any(), include_groups=False),
                            "n_tw_s1": g.has.sum(), "hold": g.holdout.any()})
        sxf["out"] = np.select([sxf.pred_y, sxf.pred & ~sxf.pred_y & sxf.any_y, sxf.pred & ~sxf.any_y,
                                sxf.any_y & sxf.am_y, sxf.any_y], ["TP", "FP_stolen", "FP", "FN_lowp", "FN_stolen"], "TN")
        tab = sxf.groupby("out").agg(n=("tw_any", "size"), tw_any=("tw_any", "mean"), tw_true=("tw_true", "mean"),
                                     tw_am=("tw_am", "mean"), multi=("n_tw_s1", lambda s: (s > 1).mean()))
        print(f"\n== {c} tau1 {tau1} sim {ts}: address-less SX {len(sxf):,}; twin support pairs {int(a.has.sum()):,}, "
              f"precision of supported pairs y {a.y[a.has].mean():.3f} (am {a.y[a.has & a.am].mean():.3f}, "
              f"non-am {a.y[a.has & ~a.am].mean():.3f})")
        print(tab.round(3).to_string())
