"""LOCO vs LB calibration: rescue90 (LB +0.00031) and gc70 (LB +0.00034) in France-F0.5 units, both directions."""
import json, sys
import numpy as np
sys.path[:0] = ["probes/P37-fr-complete", "probes/P28-addr-anchor"]
import p37 as P
fd = json.load(open("probes/P37-fr-complete/loco2.json"))["fr_dens"]
for src, tgt in (("US", "India"), ("India", "US")):
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = (a[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
    ntrue, tb = a["t2"] + a["t3"], P.TS[(src, tgt)]
    tm = {float(t): P.inv_dens(p, am, tb, g, n) for t, g in fd.items()}
    per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue).mean()
    b = am & (p >= tb)
    r = b | P.rescue(s1, p, am, b, sib, tm[0.9])
    g = r | P.gc_add(s1, sx, is2, p, am, r, sib, tm[0.7], "oneall")
    print(f"{src}->{tgt}: rescue90 {per(r) - per(b):+.5f} (pairs {int((r & ~b).sum())}, prec {y[r & ~b].mean():.3f}); "
          f"gc70 {per(g) - per(r):+.5f}", flush=True)
    del a
print("LB in France-F0.5 units (/0.1497): rescue90 +%.5f, gc70 +%.5f" % (0.00031 / 0.1497, 0.00034 / 0.1497))
