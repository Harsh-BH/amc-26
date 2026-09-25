"""P4 matcher: fit / predict, the fixed decision rule (each S2/S3 to its argmax S1, one global threshold swept for
macro F0.5) and the label-free Fellegi-Sunter baseline.

    m = fit(X, y)                  # LightGBM (M2), early-stopped on (Xv, yv) when given
    p = predict(m, X)              # float32 match probability per pair
    best = sweep(pairs, truth_n)   # pairs: s1, sx, p, is_true -> {"t", "f05", ...}
    .venv/bin/python probes/P4-match/model.py     # self-check
"""
import numpy as np
import pandas as pd

DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=12)


def feats(X: pd.DataFrame) -> list:
    return [c for c in X.columns if c not in DROP]


def fit(X, y, Xv=None, yv=None, rounds: int = 2000, params: dict | None = None, cols=None):
    """LightGBM. X / Xv: DataFrames, or float32 arrays with `cols` = their feature names."""
    import lightgbm as lgb
    if isinstance(X, pd.DataFrame):
        cols = cols or feats(X)
        X = X[cols].to_numpy(np.float32)
        Xv = None if Xv is None else Xv[cols].to_numpy(np.float32)
    d = lgb.Dataset(X, np.asarray(y, np.float32), feature_name=list(cols), free_raw_data=True)
    kw = {}
    if Xv is not None:
        kw = dict(valid_sets=[lgb.Dataset(Xv, np.asarray(yv, np.float32), reference=d)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return lgb.train({**PARAMS, **(params or {})}, d, rounds, **kw)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X[model.feature_name()], num_threads=PARAMS["num_threads"]).astype(np.float32)


def argmax_sx(s1: np.ndarray, sx: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bool mask: the pair is its SX's highest-p candidate (ties -> the first; one S1 per SX)."""
    o = np.lexsort((-p, sx))
    first = np.r_[True, sx[o][1:] != sx[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


def f05_per_s1(tp: np.ndarray, npred: np.ndarray, ntrue: np.ndarray) -> np.ndarray:
    """Macro-F0.5 terms per S1, exactly metric.f_beta: singletons 1 iff empty, tp = 0 -> 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def sweep(pairs: pd.DataFrame, truth_n: pd.Series, ts=None, argmax: bool = True, by=None) -> dict:
    """pairs: s1, sx, p, is_true (any superset of the scored S1's pairs; S1 outside truth_n are ignored after the
    argmax, so non-scored S1 still compete for their SX). truth_n: true-match count for EVERY scored S1 (0 for
    singletons) -- true pairs missing from `pairs` are recall losses. by: optional Series S1 -> group for per-group F.
    Returns the best global threshold and its macro F0.5 (+ per group, pair precision / recall)."""
    ts = np.round(np.arange(0.02, 0.99, 0.01), 2) if ts is None else ts
    s1, sx, p, y = (pairs[c].to_numpy() for c in ("s1", "sx", "p", "is_true"))
    keep = argmax_sx(pd.factorize(s1)[0], pd.factorize(sx)[0], p) if argmax else np.ones(len(p), bool)
    idx = pd.Index(truth_n.index)
    k = idx.get_indexer(s1[keep])
    ok = k >= 0
    k, pk, yk = k[ok], p[keep][ok], y[keep][ok].astype(bool)
    ntrue = truth_n.to_numpy(float)
    res = []
    for t in ts:
        m = pk >= t
        npred = np.bincount(k[m], minlength=len(idx)).astype(float)
        tp = np.bincount(k[m & yk], minlength=len(idx)).astype(float)
        res.append((t, f05_per_s1(tp, npred, ntrue).mean(), tp.sum() / max(npred.sum(), 1), tp.sum() / ntrue.sum()))
    r = pd.DataFrame(res, columns=["t", "f05", "pair_precision", "pair_recall"])
    b = r.loc[r.f05.idxmax()].to_dict()
    if by is not None:
        m = pk >= b["t"]
        npred = np.bincount(k[m], minlength=len(idx)).astype(float)
        tp = np.bincount(k[m & yk], minlength=len(idx)).astype(float)
        f = pd.Series(f05_per_s1(tp, npred, ntrue), index=idx)
        b |= {f"f05_{g}": float(v) for g, v in f.groupby(by.reindex(idx).to_numpy()).mean().items()}
    b["curve"] = r
    return b


def m0_score(X: pd.DataFrame) -> np.ndarray:
    """M0: hand-weighted rule score (no labels): name evidence + address evidence + exact keys."""
    s = (0.35 * X.name_cos + 0.15 * X.core_tset + 0.25 * X.addr_cos + 0.15 * X.addr_rule.clip(0, 7) / 7
         + 0.10 * np.maximum(X.core_eq, X.web_eq))
    return s.fillna(0).to_numpy(np.float32)


def _fs_levels(X: pd.DataFrame) -> dict:
    return {"name": np.digitize(X.name_cos.fillna(0), [0.3, 0.5, 0.7, 0.9]),
            "core": np.select([X.core_eq == 1, X.core_tset >= 0.9, X.core_tset >= 0.7], [3, 2, 1], 0),
            "addr": np.digitize(X.addr_cos.fillna(0), [0.2, 0.4, 0.6, 0.8]) + 5 * X.addr_empty_x.to_numpy(int),
            "hn": np.select([X.hn_eq == 1, X.hn_diff == 1], [2, 1], 0),
            "city": np.select([X.city_eq == 1, X.city_both == 1], [2, 1], 0),
            "state": np.select([X.state_eq == 1, X.state_diff == 1], [2, 1], 0),
            "web": np.select([X.web_eq == 1, X.web_in == 1], [2, 1], 0)}


_FS_N = {"name": 5, "core": 4, "addr": 10, "hn": 3, "city": 3, "state": 3, "web": 3}


def fs_score(params: tuple, X: pd.DataFrame) -> np.ndarray:
    lam, m, u = params
    L = _fs_levels(X)
    lm = np.log(lam) + sum(np.log(m[k][v]) for k, v in L.items())
    lu = np.log(1 - lam) + sum(np.log(u[k][v]) for k, v in L.items())
    return (1 / (1 + np.exp(lu - lm))).astype(np.float32)


def fs_fit(X: pd.DataFrame, iters: int = 30, prior: float = 0.2, fix_lam: float | None = None) -> tuple:
    """M3: Fellegi-Sunter fitted by EM, no labels (the model Splink fits): conditionally independent comparison
    levels, m / u probabilities per level. Returns (lambda, m, u) for fs_score.
    Free EM drifts to the same-name cluster at full density (lambda 0.58, AUC 0.60); fix_lam = #SX / #pairs (each SX
    has at most one parent, label-free) holds it at AUC 0.87 -- still below the hand-weighted M0 (0.96)."""
    L = _fs_levels(X)
    ramp = lambda n, a, b: np.linspace(a, b, n) / np.linspace(a, b, n).sum()
    lam, m, u = fix_lam or prior, {k: ramp(n, 1, 2) for k, n in _FS_N.items()}, {k: ramp(n, 2, 1) for k, n in _FS_N.items()}
    for _ in range(iters):
        g = fs_score((lam, m, u), X).astype(np.float64)
        lam = g.mean() if fix_lam is None else fix_lam
        for k, v in L.items():
            m[k] = (np.bincount(v, g, _FS_N[k]) + 1e-3) / (g.sum() + 1e-3 * _FS_N[k])
            u[k] = (np.bincount(v, 1 - g, _FS_N[k]) + 1e-3) / ((1 - g).sum() + 1e-3 * _FS_N[k])
    return lam, m, u


def _check():
    # 3 S1: A has 2 true SX, B is a singleton, C has 1 true SX that blocking missed
    pairs = pd.DataFrame({"s1": ["A", "A", "B", "B", "A"], "sx": ["x1", "x2", "x3", "x1", "x4"],
                          "p": [0.9, 0.8, 0.6, 0.5, 0.3], "is_true": [1, 1, 0, 0, 0]})
    tn = pd.Series({"A": 2, "B": 0, "C": 1})
    b = sweep(pairs, tn)
    # t in (0.6, 0.8]: A gets {x1, x2} (F=1), B empty (1), C empty (0) -> 2/3
    assert abs(b["f05"] - 2 / 3) < 1e-9 and 0.6 < b["t"] <= 0.8, b
    # x1's argmax is A, so B never gets x1 even at a low threshold
    assert argmax_sx(np.array([0, 1]), np.array([0, 0]), np.array([.9, .95])).tolist() == [False, True]
    assert np.allclose(f05_per_s1(np.array([2.]), np.array([3.]), np.array([2.])), 0.7142857, atol=1e-6)  # PDF example
    print("model self-check ok")


if __name__ == "__main__":
    _check()
