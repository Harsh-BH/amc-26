import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import LB_W, state
bins = [0, .05, .2, .4, .6, .71, .8, .9, .97, 1]
for c in LB_W:
    s = state(c); pr = s["pairs"]; a = pr[pr.am]
    y = a.y.to_numpy().astype(float)
    for col in ("p", "pa"):
        p = np.clip(a[col].to_numpy(), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        br = ((p - y) ** 2).mean()
        band = (p > .05) & (p < .97)
        b = pd.cut(p, bins)
        g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
        ece = (g.n * (g.p - g.y).abs()).sum() / g.n.sum()
        eceb = (g.n * (g.p - g.y).abs())[1:-1].sum() / g.n[1:-1].sum()
        print(f"{c} {col:2s}: logloss {ll:.5f} brier {br:.6f} ECE {ece:.5f} ECE(0.05-0.97) {eceb:.4f} n_band {band.sum()}")
        print("   ", " ".join(f"{str(i)}:{r.n}/{r.p:.3f}/{r.y:.3f}" for i, r in g.iterrows()))
