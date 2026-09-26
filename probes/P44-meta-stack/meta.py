"""P44 meta-learner bench on #34's argmax pairs (am = #8c's argmax, kept; acc = am & s >= t_c).

Every arm is fit on holdout half `fit` (India + US pooled, no country column) and reported on the other half vs #34
(P36's out-of-fold p', t 0.71), both directions. t_c (per country) is tuned on the fit half's S1 with scores that are
out of fold inside the fit half (2-fold by S1), so t never sees in-sample scores; the reported model is refit on the
whole fit half.
Arms: a_*  the report's logistic on [p_m2, p_hn, |diff|, min, max, has_hn, exact_hn] (+ CE r, + #8c p)
      b_*  P34's a10 residual stacker + those columns (guard kept; t 0.71 and tuned)
      o_*  our combiners: a10+all inputs, monotone GBDT, logistic on logits with interactions, isotonic per bucket,
           2-level stack on #34's p'.
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/meta.py hn21 [m2hn]   (models = attached p_<name> columns)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402
import a10  # noqa: E402  (P34, via ev's sys.path)

lg = E.lg
sig = lambda z: 1 / (1 + np.exp(-z))
PRM = dict(a10.PRM)
GRID = np.round(np.arange(0.40, 0.951, 0.01), 2)


def prep(models: list) -> dict:
    """Per country (A, S) with the derived inputs every arm draws from."""
    import arm_c
    D = {}
    for c in E.C:
        A, S = E.load(c)
        if "freq" not in A:
            A["freq"] = arm_c.name_freq(c, A)
            A.to_parquet(E.OUT / f"A_{c}.parquet")
        A["p_m2"] = sig(A.lm2.to_numpy())
        A["has_hn"] = A.hn_lev.notna().astype(np.float32)
        A["exact_hn"] = ((A.hn_cls == 0) & (A.has_hn > 0)).astype(np.float32)
        A["r_na"] = A.r.isna().astype(np.float32)
        A["r0"] = A.r.fillna(0.0)
        A["lr"] = lg(A.r.to_numpy())
        A["lq"] = lg(A.q.to_numpy())
        A["lpa"] = lg(A.pa.to_numpy())
        A["lfreq"] = np.log1p(A.freq.fillna(1).to_numpy())
        A["tw_any"] = (A.n_twins > 0).astype(np.float32)
        for m in models:
            A[f"l_{m}"] = lg(A[f"p_{m}"].to_numpy())
        D[c] = (A, S)
    return D


def inner(S: pd.DataFrame, A: pd.DataFrame, fit: int) -> np.ndarray:
    """2-fold split of the fit half's S1 (by S1 index parity after a fixed shuffle) -> fold id per A row, -1 outside."""
    rng = np.random.default_rng(44)
    f = rng.integers(0, 2, len(S))
    f[S.half.to_numpy() != fit] = -1
    return f[A.g.to_numpy()]


def run_arm(D: dict, name: str, fitpred, fixed_t=None, res=None):
    """fitpred(train_rows: {c: bool mask}, D) -> {c: scores for all rows of A}. Reports both directions."""
    t0 = time.time()
    for fit in (0, 1):
        rep = 1 - fit
        fold = {c: inner(D[c][1], D[c][0], fit) for c in E.C}
        # scores out of fold inside the fit half, for t
        oof = {c: np.full(len(D[c][0]), np.nan) for c in E.C}
        for k in ((0, 1) if fixed_t is None else ()):
            s = fitpred({c: fold[c] == 1 - k for c in E.C}, D)
            for c in E.C:
                oof[c][fold[c] == k] = s[c][fold[c] == k]
        full = fitpred({c: D[c][0].half.to_numpy() == fit for c in E.C}, D)
        r, ts = {}, {}
        for c in E.C:
            A, S = D[c]
            # tune t on the fit half: rows outside it get score -inf-ish, S1 outside it are ignored by tune_t
            so = np.where(np.isnan(oof[c]), 0.0, oof[c])
            ts[c] = fixed_t if fixed_t is not None else E.tune_t(A, S, so, fit, GRID)
            r[c] = E.delta(A, S, full[c] >= ts[c], rep)
        lb, se = E.lbeq(r)
        row = {"arm": name, "fit": fit, "rep": rep, "lbeq": lb, "se": se, "z": lb / se, "t": ts,
               **{f"d_{c}": r[c][0] for c in E.C}, **{f"se_{c}": r[c][1] for c in E.C},
               **{f"updown_{c}": r[c][2:] for c in E.C}, "s": round(time.time() - t0)}
        if res is not None:
            res.append(row)
        print(f"{name:34s} h{fit}->h{rep} t={ts}: LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:+.1f} SE) | "
              + " | ".join(f"{c} {r[c][0]:+.6f} ± {r[c][1]:.6f} (+{r[c][2]}/-{r[c][3]})" for c in E.C), flush=True)


# ---------- model families ----------
def logistic(cols, interactions=(), C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def X_of(A):
        X = A[list(cols)].to_numpy(np.float64)
        ex = [A[a].to_numpy(np.float64) * A[b].to_numpy(np.float64) for a, b in interactions]
        X = np.column_stack([X] + ex) if ex else X
        return np.nan_to_num(X)

    def fp(tr, D):
        X = np.vstack([X_of(D[c][0])[tr[c]] for c in E.C])
        y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C])
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
        fp.coef = dict(zip(list(cols) + [f"{a}*{b}" for a, b in interactions], m.coef_[0].round(3).tolist()))
        return {c: m.predict_proba(sc.transform(X_of(D[c][0])))[:, 1] for c in E.C}
    return fp


def gbdt(cols, init="lp", guard=True, rounds=200, mono=None, cat=("cat",), **kw):
    """a10-style residual LightGBM on logit(init); guard = sibling-offset pairs keep #8c's p (P34)."""
    import lightgbm as lgb
    P = dict(PRM, **kw)
    if mono:
        P["monotone_constraints"] = [mono.get(c, 0) for c in cols]
        P["monotone_constraints_method"] = "advanced"
    cats = [c for c in cat if c in cols]

    def fp(tr, D):
        X = pd.concat([D[c][0].loc[tr[c], list(cols)] for c in E.C])
        y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C])
        base = np.concatenate([D[c][0][init].to_numpy()[tr[c]] for c in E.C])
        b = lgb.train(P, lgb.Dataset(X, y, init_score=base, categorical_feature=cats, free_raw_data=True), rounds)
        out = {}
        for c in E.C:
            A = D[c][0]
            s = sig(b.predict(A[list(cols)], raw_score=True) + A[init].to_numpy())
            out[c] = np.where(A.hc.to_numpy() == 1, A.p.to_numpy(), s) if guard else s
        return out
    return fp


def iso_bucket(score="pa", bucket="cat"):
    """Isotonic calibration of `score` per bucket (fit half), so one t cuts every bucket at the same precision."""
    from sklearn.isotonic import IsotonicRegression

    def fp(tr, D):
        s = np.concatenate([D[c][0][score].to_numpy()[tr[c]] for c in E.C])
        b = np.concatenate([D[c][0][bucket].to_numpy()[tr[c]] for c in E.C])
        y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in E.C])
        iso = {k: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[b == k], y[b == k])
               for k in np.unique(b) if (b == k).sum() >= 500}
        out = {}
        for c in E.C:
            A = D[c][0]
            v, bb = A[score].to_numpy(), A[bucket].to_numpy()
            o = v.copy()
            for k, m in iso.items():
                if (bb == k).any():
                    o[bb == k] = m.predict(v[bb == k])
            out[c] = o
        return out
    return fp


def ident(col):
    return lambda tr, D: {c: D[c][0][col].to_numpy() for c in E.C}


if __name__ == "__main__":
    models = sys.argv[1:]
    D = prep(models)
    res = []
    run_arm(D, "control_34_pa_t071", ident("pa"), fixed_t=E.T, res=res)          # must be 0 exactly
    run_arm(D, "control_34_pa_tuned", ident("pa"), res=res)
    run_arm(D, "control_a10_refit", gbdt(a10.COLS), fixed_t=E.T, res=res)        # == #34 up to the inner split
    HN2 = ["has_hn", "exact_hn"]
    for m in models:
        pm = f"p_{m}"
        for c in E.C:
            A = D[c][0]
            A[f"diff_{m}"] = (A.p_m2 - A[pm]).abs()
            A[f"min_{m}"] = np.minimum(A.p_m2, A[pm])
            A[f"max_{m}"] = np.maximum(A.p_m2, A[pm])
        rep = ["p_m2", pm, f"diff_{m}", f"min_{m}", f"max_{m}"] + HN2
        run_arm(D, f"a_m2alone_{m}", ident("p_m2"), res=res)
        run_arm(D, f"a_{m}alone", ident(pm), res=res)
        run_arm(D, f"a_report_{m}", logistic(rep), res=res)
        run_arm(D, f"a_report+r_{m}", logistic(rep + ["r0", "r_na"]), res=res)
        run_arm(D, f"a_report+r+p8_{m}", logistic(rep + ["r0", "r_na", "p"]), res=res)
        run_arm(D, f"b_a10+report_{m}_t071", gbdt(a10.COLS + [f"l_{m}", f"diff_{m}", f"min_{m}", f"max_{m}"] + HN2),
                fixed_t=E.T, res=res)
        run_arm(D, f"b_a10+report_{m}_tuned", gbdt(a10.COLS + [f"l_{m}", f"diff_{m}", f"min_{m}", f"max_{m}"] + HN2),
                res=res)
    # ---- our own combiners ----
    L = [f"l_{m}" for m in models]
    HNF = ["hn_d", "hn_cls", "hn_lev", "hn_lendiff", "has_hn", "exact_hn"]
    CTX = ["lq", "lfreq", "tw_any"]
    ALL = a10.COLS + L + HNF + CTX
    run_arm(D, "o_a10+all_t071", gbdt(ALL), fixed_t=E.T, res=res)
    run_arm(D, "o_a10+all_tuned", gbdt(ALL), res=res)
    run_arm(D, "o_a10+all_noguard", gbdt(ALL, guard=False), res=res)
    from ctx import CTX2
    for m in models:
        run_arm(D, f"o_a10+l_{m}_t071", gbdt(a10.COLS + [f"l_{m}"]), fixed_t=E.T, res=res)
    run_arm(D, "o_a10+all+ctx_t071", gbdt(ALL + CTX2), fixed_t=E.T, res=res)
    run_arm(D, "o_a10+all+ctx_tuned", gbdt(ALL + CTX2), res=res)
    run_arm(D, "o_a10+all+ctx_400r", gbdt(ALL + CTX2, rounds=400), fixed_t=E.T, res=res)
    mono = {k: 1 for k in ["lp", "lm2", "r"] + L}
    run_arm(D, "o_mono_gbdt", gbdt(ALL, mono=mono, rounds=300, num_leaves=15), res=res)
    run_arm(D, "o_stack2_on_pa", gbdt(ALL + ["lpa"], init="lpa", rounds=100), res=res)
    LOG = ["lp", "lm2", "lr", "r_na", "lq", "has_hn", "exact_hn", "noaddr", "tw_any", "lfreq", "n_conf_other",
           "s1_max_other", "n_cand_sx"] + L
    for c in E.C:
        A = D[c][0]
        A["lr"] = A.lr.where(A.r.notna(), 0.0)
        A["p2m0"] = A.p2m.fillna(A.p2)
        A["lp2"] = lg(A.p2m0.to_numpy())
        A["hn_bad"] = ((A.hn_cls >= 1) & (A.hn_cls != 2)).astype(np.float32)
    LOG += ["lp2", "hn_bad"]
    inter = [("lp", "has_hn"), ("lp", "noaddr"), ("lp", "hn_bad"), ("lp", "lp2")] + [(l, "has_hn") for l in L]
    run_arm(D, "o_logit_inter", logistic(LOG, inter), res=res)
    run_arm(D, "o_iso_cat_pa", iso_bucket("pa", "cat"), res=res)
    run_arm(D, "o_iso_hncls_pa", iso_bucket("pa", "hn_cls"), res=res)
    out = E.OUT / f"meta_{'_'.join(models) or 'none'}.json"
    out.write_text(json.dumps(res, default=float, indent=0))
    print("wrote", out, flush=True)
