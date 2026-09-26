"""P25 evaluation on the #8c yardstick: P22's ev.py (imported unchanged) with P25's model sets.

Each config is a pair-level p for M2 built from the variants' holdout p (same rows: #6's pruned pairs), then P22's
run_variant: bge CE blend (P13 w refit on half 0; e5 fallback) -> US stacker (shipped | refit on half 0) -> argmax,
t 0.71 | t swept on half 0 -> per-S1 F0.5 on half 1, paired vs #8c as shipped and vs the control put through the
same procedure. Configs:
  alone_<m>    the variant by itself (in place of m2_big)
  avg_<m>      equal-weight logit mean of m2_big and the variant
  avg_bag      m2_big + bag1..4           avg_boost  m2_big + dart, goss, extra, shallow, deep
  avg_data     m2_big + bag3 + bag4 (the fold 6-9 models: the only ones with a clearly lower band corr, 0.96 vs 0.98;
               added after the six-model eval, so post hoc)
  avg_all      m2_big + every variant     avg_var    every variant, no m2_big
  lr_all       logistic stack on logits of m2_big + every variant (fit on the half-0 S1's kept pairs, both countries)
  nn_all       the same with non-negative weights (L-BFGS-B)
"""
import gc, json, time

import numpy as np
import pandas as pd

import ev

BAG = ("bag1", "bag2", "bag3", "bag4")
BOOST = ("dart", "goss", "extra", "shallow", "deep")
C = ev.C
lg, sg = ev.lg, ev.sg


def configs(models) -> dict:
    ms = tuple(models)
    out = {f"alone_{m}": ("alone", (m,)) for m in ms} | {f"avg_{m}": ("avg", ("lgb", m)) for m in ms}
    for tag, grp in (("bag", BAG), ("boost", BOOST), ("data", ("bag3", "bag4"))):
        g = tuple(m for m in ms if m in grp)
        if len(g) > 1:
            out[f"avg_{tag}"] = ("avg", ("lgb",) + g)
    if len(ms) > 1:
        out |= {"avg_all": ("avg", ("lgb",) + ms), "avg_var": ("avg", ms), "lr_all": ("lr", ("lgb",) + ms),
                "nn_all": ("nn", ("lgb",) + ms)}
    return out


def fit_nn(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """min mean logloss of sigmoid(Z w + b), w >= 0 -> [w..., b]."""
    from scipy.optimize import minimize
    Z = Z.astype(np.float64)
    yf = y.astype(np.float64)

    def f(x):
        z = Z @ x[:-1] + x[-1]
        l = np.logaddexp(0, z) - yf * z
        g = 1 / (1 + np.exp(-z)) - yf
        return l.mean(), np.r_[Z.T @ g, g.sum()] / len(yf)

    k = Z.shape[1]
    x0 = np.r_[np.full(k, 1.0 / k), 0.0]
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * k + [(None, None)])
    return r.x


def build(ds: dict, P: dict, cfg: dict) -> tuple:
    """{tag: {c: p}} and the fitted weights."""
    V, W = {}, {}
    tr = {c: ev.half0_rows(ds[c]) for c in C}
    for tag, (kind, ms) in cfg.items():
        if kind == "alone":
            V[tag] = {c: P[c][ms[0]] for c in C}
            continue
        Z = {c: np.column_stack([lg(P[c][m]) for m in ms]) for c in C}
        if kind == "avg":
            V[tag] = {c: sg(Z[c].mean(1)) for c in C}
            continue
        Zt = np.concatenate([Z[c][tr[c]] for c in C])
        yt = np.concatenate([ds[c]["y"][tr[c]] for c in C])
        if kind == "lr":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Zt, yt)
            x = np.r_[lr.coef_[0], lr.intercept_[0]]
        else:
            x = fit_nn(Zt, yt)
        W[tag] = dict(zip(list(ms) + ["_b"], [round(float(v), 4) for v in x]))
        V[tag] = {c: sg(Z[c] @ x[:-1] + x[-1]) for c in C}
        del Zt, yt
    return V, W


def pair_stats(ds: dict, P: dict, models) -> dict:
    """Pair level, half-1 S1's kept pairs: logloss of each model, and its logit correlation with m2_big on the band."""
    out = {}
    for m in ("lgb",) + tuple(models):
        r = {}
        for c in C:
            d = ds[c]
            h1 = d["pos"] >= 0
            h1[h1] &= d["h"]["half"][d["pos"][h1]] == 1
            p = np.clip(P[c][m][h1].astype(np.float64), 1e-7, 1 - 1e-7)
            y = d["y"][h1]
            r[f"logloss_{c}"] = float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))
            b = ev.band(P[c]["lgb"]) & h1
            r[f"corr_band_{c}"] = float(np.corrcoef(lg(P[c][m][b]), lg(P[c]["lgb"][b]))[0, 1])
            r[f"mean_abs_dlogit_band_{c}"] = float(np.mean(np.abs(lg(P[c][m][b]) - lg(P[c]["lgb"][b]))))
        out[m] = {k: round(v, 6) for k, v in r.items()}
        print("pair", m, out[m], flush=True)
    return out


def noce(ds: dict, pv: dict, ts=np.round(np.arange(0.50, 0.951, 0.01), 2)) -> tuple:
    """M2 p alone (no CE, no stacker): per-S1 F0.5 (India then US) at the t best on half 0, and that t."""
    per = {c: ev.f05(ds[c], pv[c], list(ts)) for c in C}
    allf = {t: np.concatenate([per[c][t] for c in C]) for t in ts}
    hv = np.concatenate([ds[c]["h"]["half"] for c in C])
    t0 = max(ts, key=lambda t: allf[t][hv == 0].mean())
    return allf[t0], float(t0)


def evaluate(models, name="p25", only=None):
    """P22's evaluate loop over P25's configs (only = a subset of config tags)."""
    t00 = time.time()
    ds = {c: ev.load(c) for c in C}
    P = {c: ev.preds(ds[c], c, models) for c in C}
    cfg = configs(models)
    if only:
        cfg = {k: v for k, v in cfg.items() if k in only}
    out = {"pair": pair_stats(ds, P, models)}
    V, W = build(ds, P, cfg)
    out["weights"] = W
    print("weights", json.dumps(W), flush=True)
    hv = np.concatenate([ds[c]["h"]["half"] for c in C])
    grp = np.concatenate([[c] * ds[c]["h"]["n"] for c in C])
    h1 = hv == 1
    ship = []
    for c, d in ds.items():                                  # #8c exactly as shipped
        p = ev.blended(d, d["p_lgb"], ev.W8)
        ship.append(ev.f05(d, ev.stacked(d, p) if c == "US" else p, ev.T8))
    ship = np.concatenate(ship)
    ctl_res, ctl = ev.run_variant(ds, {c: P[c]["lgb"] for c in C}, "ctl")
    ctl_noce, _ = noce(ds, {c: P[c]["lgb"] for c in C})
    out |= {"control_8c_shipped": {"f05_half1": float(ship[h1].mean()), "f05_all": float(ship.mean())},
            "control_refit": ctl_res | {k: float(v[h1].mean()) for k, v in ctl.items()}}
    for tag, pv in V.items():
        b = {c: ev.band(pv[c]) for c in C}
        new = {c: b[c] & np.isnan(ds[c]["r_bge"]) for c in C}
        r, f = ev.run_variant(ds, pv, tag)
        fn, r["t_noce"] = noce(ds, pv)
        r["vs_ctl_noce"] = ev.paired(fn, ctl_noce, h1)
        r |= {"band": int(sum(b[c].sum() for c in C)),
              "band_no_bge_holdoutS1": int(sum((new[c] & (ds[c]["pos"] >= 0)).sum() for c in C)),
              "band_no_ce_holdoutS1": int(sum((new[c] & np.isnan(ds[c]["r_e5"]) & (ds[c]["pos"] >= 0)).sum() for c in C))}
        for k, v in f.items():
            r[f"f05_half1_{k}"] = float(v[h1].mean())
            r[f"vs_ctl_{k}"] = ev.paired(v, ctl[k], h1)
            r[f"vs_8c_{k}"] = ev.paired(v, ship, h1)
            for c in C:
                r[f"vs_ctl_{k}_{c}"] = ev.paired(v, ctl[k], h1 & (grp == c))["gain"]
                r[f"vs_8c_{k}_{c}"] = ev.paired(v, ship, h1 & (grp == c))["gain"]
        out[tag] = r
        print(tag, json.dumps({k: r[k] for k in ("vs_8c_refit_t71", "vs_ctl_refit_t71", "vs_8c_ship_t71",
                                                 "band_no_bge_holdoutS1")}, default=float), flush=True)
        gc.collect()
    out["runtime_s"], out["peak_gb"] = round(time.time() - t00), round(ev.peak(), 2)
    (ev.OUT / f"eval_{name}.json").write_text(json.dumps(out, default=float, indent=1))
    return out
