"""P36 shared state: the #34 India/US holdout decision (#8c + P34's guarded a10 stacker), out of fold.

S1 of half h take the stacked p' of the booster fit on half 1-h (P34's ho_p_{c}_h{1-h}.npy), so every holdout S1 is
scored by a booster that never saw it. acc = #8c argmax & p' >= T (P34's decision).
    probes/P36-iu-residual/py.sh probes/P36-iu-residual/base.py      # self-check: reproduces P34's half-1 F0.5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p36"
sys.path[:0] = [str(ROOT / "probes/P34-a10-stack"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import a10 as A  # noqa: E402

T = A.T
LB_W = {"India": 0.468, "US": 0.383}
half = lambda ids: (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2   # = run.half
NA = {"", "NULL", "<NULL>", "N/A", "NA", "NONE", "-"}


def state(c: str) -> dict:
    """pairs: every holdout-S1 candidate row (s1, sx, y, p_m2, r, p = #8c, am, pred8c, g = S1 index, pa = p' | NaN,
    acc = #34 decision); ctx: n_true / f05_8c per scored S1; h: half per S1; f34: per-S1 F0.5 under acc."""
    ctx = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true", "f05"])
    ids = ctx.index.to_numpy()
    h = half(ids)
    gi = pd.Series(np.arange(len(ids)), index=ids)
    pr = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p", "am", "pred"])
    pr = pr[pr.holdout].drop(columns="holdout").reset_index(drop=True)
    pr["g"] = gi.reindex(pr.s1).to_numpy()
    assert not pr.g.isna().any()
    ho = pd.read_parquet(W / f"p34/ho_{c}.parquet", columns=["s1", "sx", "p"])
    p0, p1 = (np.load(W / f"p34/ho_p_{c}_h{fh}.npy") for fh in (0, 1))
    hs = h[gi.reindex(ho.s1).to_numpy()]
    ho["pa"] = np.where(hs == 1, p0, p1)                 # out of fold
    pr = pr.merge(ho[["s1", "sx", "pa"]], on=["s1", "sx"], how="left")
    assert pr.pa.notna().sum() == len(ho) and pr.am[pr.pa.notna()].all() and not pr.pa[pr.am].isna().any()
    pr["acc"] = pr.am & (pr.pa >= T)
    g, y, acc = pr.g.to_numpy(), pr.y.to_numpy(), pr.acc.to_numpy()
    n = len(ids)
    f34 = A.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), ctx.n_true.to_numpy())
    f8 = A.fb(np.bincount(g[pr.pred & y], minlength=n), np.bincount(g[pr.pred], minlength=n), ctx.n_true.to_numpy())
    assert np.allclose(f8, ctx.f05.to_numpy()), "the #8c yardstick does not reproduce"
    return dict(c=c, ids=ids, h=h, pairs=pr, ctx=ctx, n_true=ctx.n_true.to_numpy(), f8=f8, f34=f34)


def paired(d: np.ndarray, h: np.ndarray, rep: int) -> tuple:
    """(mean, SE) of a per-S1 delta over the S1 of half rep."""
    x = d[h == rep]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


if __name__ == "__main__":
    for c, want in (("India", 0.990200), ("US", 0.989601)):
        s = state(c)
        f1 = s["f34"][s["h"] == 1].mean()
        print(c, f"half-1 F0.5 #8c {s['f8'][s['h'] == 1].mean():.6f} -> #34 {f1:.6f} (P34: {want}); "
                 f"all {s['f8'].mean():.6f} -> {s['f34'].mean():.6f}")
        assert abs(f1 - want) < 5e-7
