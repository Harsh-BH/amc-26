"""P35 a11: empty-S1 rescue on the #8c yardstick. An S1 with no accepted pair takes its best candidate (argmax SX
only, or also an SX not accepted anywhere) if p >= t_r < t. Both halves; all SX and address-less-only variants."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load, gain
from fit import LBW

S = [load(c, ["s1", "sx", "holdout", "y", "p", "am", "pred", "noaddr"]) for c in ("India", "US")]
for mode in ("argmax", "unaccepted"):
    for only_na in (False, True):
        for tr in (0.4, 0.5, 0.6, 0.65):
            out = {}
            for s in S:
                d = s["d"]; p, am, pred = d.p.to_numpy(), d.am.to_numpy(), d.pred.to_numpy()
                s1c = pd.factorize(d.s1)[0]
                empty = np.bincount(s1c, pred)[s1c] == 0
                sx_acc = pd.Series(pred).groupby(pd.factorize(d.sx)[0]).transform("any").to_numpy()
                ok = am if mode == "argmax" else ~sx_acc
                m = empty & ok & (p >= tr) & ~pred
                if only_na:
                    m &= d.noaddr.to_numpy()
                # one pair per S1 (its best eligible), one S1 per SX (highest p)
                idx = np.flatnonzero(m)
                k = pd.DataFrame({"i": idx, "s": s1c[idx], "x": d.sx.to_numpy()[idx], "p": p[idx]})
                k = k.sort_values("p", ascending=False).drop_duplicates("s").drop_duplicates("x")
                new = pred.copy(); new[k.i.to_numpy()] = True
                out[s["c"]] = (gain(s, new), len(k), float(d.y.to_numpy()[k.i.to_numpy()].mean()) if len(k) else np.nan)
            lb = {h: sum(LBW[c] * out[c][0][h] for c in LBW) for h in ("h0", "h1")}
            print(f"{mode:10s} noaddr-only {only_na!s:5s} t_r {tr}: " + " | ".join(
                f"{c} h0 {out[c][0]['h0']:+.6f} h1 {out[c][0]['h1']:+.6f} ± {out[c][0]['se1']:.6f} ({out[c][1]} pairs, prec {out[c][2]:.3f})"
                for c in LBW) + f" | LB-eq h0 {lb['h0']:+.6f} h1 {lb['h1']:+.6f}", flush=True)
