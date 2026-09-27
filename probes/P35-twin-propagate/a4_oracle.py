"""P35 a4: headroom. Oracle re-decisions restricted to address-less SX (holdout rows scored, others unchanged):
 O1 accept/reject only #8c's argmax pair (a10's space); O2 any candidate (re-assignment allowed); O3 = O2 on all SX."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load, gain

for c in ("India", "US"):
    S = load(c, ["s1", "sx", "holdout", "y", "am", "pred", "noaddr", "p"]); d = S["d"]
    na, y, am, pred = d.noaddr.to_numpy(), d.y.to_numpy(), d.am.to_numpy(), d.pred.to_numpy()
    o1 = np.where(na, am & y, pred)
    o2 = np.where(na, y, pred)
    o3 = y.copy()
    for n, m in (("O1 accept-only (noaddr)", o1), ("O2 reassign (noaddr)", o2), ("O3 reassign (all)", o3)):
        g = gain(S, m)
        print(f"{c} {n}: all {g['all']:+.5f}  h1 {g['h1']:+.5f}")
    # split of O2 by #8c state of the address-less SX
    sxs = d[na].groupby("sx").agg(ya=("y", "any"), amy=("y", lambda s: False))
