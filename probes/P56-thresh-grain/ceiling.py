"""P56: how much any re-thresholding of sub44r's p'' could win. Oracle: pairs with p'' in [lo, hi) are decided by their
label (accept iff true), every other pair keeps the base (p'' >= 0.71). Per-S1 F0.5 delta per half, LB-eq.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/ceiling.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm1 as A1

D = {c: A1.load(c) for c in A1.LB_W}
out = {}
for lo, hi in ((0.3, 0.97), (0.5, 0.9), (0.6, 0.82), (0.66, 0.76)):
    r = {}
    for c, d in D.items():
        band = (d["pa"] >= lo) & (d["pa"] < hi)
        acc = np.where(band, d["y"], d["pa"] >= A1.T0)
        g, n = d["g"], d["n"]
        x = A1.fb(np.bincount(g[acc & d["y"]], minlength=n), np.bincount(g[acc], minlength=n), d["n_true"]) - d["base"]
        r[c] = [float(x[d["half"] == h].mean()) for h in (0, 1)] + [int(band.sum())]
    lb = [sum(A1.LB_W[c] * r[c][h] for c in D) for h in (0, 1)]
    out[f"{lo}-{hi}"] = lb
    print(f"oracle band [{lo}, {hi}): LB-eq h0 {lb[0]:+.6f} h1 {lb[1]:+.6f}",
          {c: f"{v[0]:+.6f}/{v[1]:+.6f} pairs {v[2]}" for c, v in r.items()}, flush=True)
