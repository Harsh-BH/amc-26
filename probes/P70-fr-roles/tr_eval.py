"""Paired holdout test: shipped India/US combiner columns (P50 e_zoo COLS: P44 FINAL + P45 zoo) vs the same + trace features.
Same LightGBM params / residual on lp / hc guard / t .71; fit on one half, report on the other; per-S1 paired delta x LB weight."""
import os, sys
os.environ["P44_VARIANT"] = "r"
import numpy as np, pandas as pd
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P70-fr-roles"]
import ev as E, meta as MT, te44  # noqa: E402
import zoo50 as Z  # noqa: E402
import e_zoo  # noqa: E402
import traces as TRm  # noqa: E402
LB = {"India": 809986 / 1732544, "US": 663106 / 1732544}
D = MT.prep(te44.MODELS)
Z.attach(D)
for c in E.C:
    t = pd.read_parquet(f"data/work/p70/tr_train_{c}.parquet")
    assert len(t) == len(D[c][0])
    for k in TRm.TR:
        D[c][0][k] = t[k].to_numpy(np.float32)
arms = {"shipped": list(e_zoo.COLS), "+traces": list(e_zoo.COLS) + TRm.TR}
sel = sys.argv[1:] or list(arms)
out = {}
for fit in (0, 1):
    rep = 1 - fit
    F = {}
    for nm in arms:
        fp = MT.gbdt(arms[nm])
        s = fp({c: D[c][0].half.to_numpy() == fit for c in E.C}, D)
        F[nm] = {c: E.per_s1(D[c][0], D[c][1], s[c] >= E.T) for c in E.C}
    tot, var = 0.0, 0.0
    parts = []
    for c in E.C:
        m = D[c][1].half.to_numpy() == rep
        d = (F["+traces"][c] - F["shipped"][c])[m]
        mu, se = LB[c] * d.mean(), LB[c] * d.std(ddof=1) / np.sqrt(m.sum())
        tot += mu; var += se ** 2
        parts.append(f"{c} {mu:+.6f} ± {se:.6f} (+{int((d > 0).sum())}/-{int((d < 0).sum())}); shipped F {F['shipped'][c][m].mean():.5f}")
    print(f"fit h{fit} -> report h{rep}: +traces vs shipped LB-eq {tot:+.6f} ± {np.sqrt(var):.6f} ({tot / np.sqrt(var):+.1f} SE) | " + " | ".join(parts), flush=True)
