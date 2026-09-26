"""P35: paired per-S1 F0.5 on the #8c holdout yardstick (data/work/p35/ho_<c> + p23/s1ctx_<c>).

    S = load("India"); f = f_of(S, pred)          # pred: bool mask over S["d"] rows (only holdout rows count)
    d = f - S["f0"]; report on S["h"] == 1 (run.half's report half) with SE = std / sqrt(n)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
half = lambda ids: (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2   # = run.half


def fb(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(c: str, cols=None) -> dict:
    d = pd.read_parquet(W / f"p35/ho_{c}.parquet", columns=cols)
    ctx = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true", "f05"])
    ids = ctx.index.to_numpy()
    g = pd.Index(ids).get_indexer(d.s1)
    assert ((g >= 0) == d.holdout.to_numpy()).all()
    return dict(c=c, d=d, g=g, n=len(ids), n_true=ctx.n_true.to_numpy(), f0=ctx.f05.to_numpy(), h=half(ids))


def f_of(S: dict, pred: np.ndarray) -> np.ndarray:
    g, y = S["g"], S["d"].y.to_numpy()
    k = pred & (g >= 0)
    return fb(np.bincount(g[k & y], minlength=S["n"]), np.bincount(g[k], minlength=S["n"]), S["n_true"])


def gain(S: dict, pred: np.ndarray) -> dict:
    """Paired delta vs #8c per half: mean, SE; plus the LB-equivalent weight is applied by the caller."""
    dl = f_of(S, pred) - S["f0"]
    out = {}
    for hv in (0, 1):
        x = dl[S["h"] == hv]
        out[f"h{hv}"], out[f"se{hv}"] = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    out["all"] = float(dl.mean())
    return out


if __name__ == "__main__":
    for c in ("India", "US"):
        S = load(c, ["s1", "sx", "holdout", "y", "pred"])
        f = f_of(S, S["d"].pred.to_numpy())
        assert np.abs(f - S["f0"]).max() < 1e-9, np.abs(f - S["f0"]).max()
        print(c, "reproduces #8c:", round(f.mean(), 6), "half1", round(f[S["h"] == 1].mean(), 6))
