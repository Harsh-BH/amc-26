"""P35 a9: the brief's rules on the #8c yardstick. (a) propagate: an address-less SX's twin-supported candidate X
(twin p >= tau1) with p >= floor becomes its argmax at p' = max(p, t) (accepted); (b) conflict guard: an accepted
address-less pair with no twin while another candidate S1 of the SX has one is dropped. Both halves + SE."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load, gain
from gdec import decide
from fit import LBW

T = 0.71
S = []
for c in ("India", "US"):
    s = load(c, ["s1", "sx", "holdout", "y", "p", "pred", "noaddr"])
    s["x"] = pd.read_parquet(f"data/work/p35/gx_{c}.parquet", columns=["tw_p", "n_sup", "rank"])
    s["x"]["p"] = s["d"].p.to_numpy()[s["x"].index.to_numpy()]
    S.append(s)
rules = {f"propagate floor {fl} n_sup {ns}": (fl, ns) for fl in (0.3, 0.5, 0.6) for ns in (1, 9)}
rules["guard"] = None
for name, r in rules.items():
    out = {}
    for s in S:
        x = s["x"]; p = x.p.to_numpy().copy()
        if r is None:
            base = s["d"].pred.to_numpy()[x.index.to_numpy()]
            drop = base & (x.tw_p.to_numpy() == 0) & (x.n_sup.to_numpy() > 0)
            p[drop] = 0.0
        else:
            fl, ns = r
            m = (x.tw_p.to_numpy() > 0) & (x.p.to_numpy() >= fl) & (x.n_sup.to_numpy() <= ns)
            p[m] = np.maximum(p[m], 0.999)                      # wins the SX's argmax and clears t
        out[s["c"]] = gain(s, decide(s["d"], x, p, T))
    lb = {h: sum(LBW[c] * out[c][h] for c in LBW) for h in ("h0", "h1")}
    print(f"{name:28s} " + " | ".join(f"{c} h0 {out[c]['h0']:+.6f} h1 {out[c]['h1']:+.6f} ± {out[c]['se1']:.6f}" for c in LBW)
          + f" | LB-eq h0 {lb['h0']:+.6f} h1 {lb['h1']:+.6f}", flush=True)
