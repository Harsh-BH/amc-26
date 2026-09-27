"""P52 arm 3: adversarial validation of the France-chain second-stage features, then per-frame quantile ranks for the
most shifted ones, and a refit of P39's frame scorer (LightGBM on P39's 'full' columns) on one LOCO frame's labels,
applied to the other. Decode on top of the current recipe analogue (P42 density-mapped bars + P48 veto):
  add    free pairs (argmax, non-SIB3, not capped out, under the caps) with s >= tau join
  addrm  add, and recipe pairs with s < 0.3 leave
Variants: raw (P39 'full'), rank5 (top-5 France-shifted columns -> per-frame percentile rank), drop5 (dropped).

    .venv/bin/python probes/P52-fr-calib/arm3.py      # -> probes/P52-fr-calib/arm3.json
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p52 as P  # noqa: E402
import solution_p42 as S  # noqa: E402

W = P.W / "p39"
FULL = ["lt", "q", "rank_src", "n_acc_src", "n_acc_oth", "n_live_src", "n_live_oth", "best_oth_lt", "best_src_ex_lt", "is2",
        "hn_rel", "kind", "nm_ratio", "nm_tset", "city", "state_eq", "is_web", "addr_eq", "grp_sim", "grp_hn"]
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
FR = {"US_India": "loco_US_India", "India_US": "loco_India_US"}


def X(d, cols):
    return d[cols].astype(np.float32).to_numpy()


def adversarial(frames, fr, n=150_000, seed=0):
    """France vs the pooled LOCO targets (and target vs target): cross-fitted AUC and gain share per column."""
    rng = np.random.default_rng(seed)
    pick = lambda d: d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]
    out = {}
    for name, a, b in (("France_vs_loco", pd.concat([pick(f) for f in frames.values()]), pick(fr)),
                       ("US_tgt_vs_India_tgt", pick(frames["India_US"]), pick(frames["US_India"]))):
        Xa, Xb = X(a, FULL), X(b, FULL)
        Xx, yy = np.vstack([Xa, Xb]), np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
        fold = rng.integers(0, 2, len(Xx))
        pr, gain = np.zeros(len(Xx)), np.zeros(len(FULL))
        for k in range(2):
            m = lgb.train(dict(PAR, num_threads=8), lgb.Dataset(Xx[fold != k], yy[fold != k]), 200)
            pr[fold == k] = m.predict(Xx[fold == k])
            gain += m.feature_importance("gain")
        uni = {c: round(abs(roc_auc_score(yy, Xx[:, i]) - 0.5) + 0.5, 3) for i, c in enumerate(FULL)}
        g = gain / gain.sum()
        out[name] = {"auc": round(float(roc_auc_score(yy, pr)), 4),
                     "top": [(FULL[i], round(float(g[i]), 3), uni[FULL[i]]) for i in np.argsort(-g)[:8]]}
        print(name, out[name], flush=True)
    return out


def rank_cols(d, cols):
    d = d.copy()
    for c in cols:
        d[c] = d[c].rank(pct=True).astype(np.float32)
    return d


def decode(z, st, ref, s, tau, mode):
    """On top of the recipe: add free pairs with s >= tau up to the caps (best s first); addrm also drops ref pairs s < 0.3."""
    s1, is2, n = z["s1"], z["is2"], z["n"]
    base = ref & ~(s < 0.3) if mode == "addrm" else ref.copy()
    free = z["free"] & ~base
    n2, n3 = S.counts(s1, is2, base, n)
    c = np.flatnonzero(free & (s >= tau))
    o, rk = S.rank_in(s1 * 2 + is2, s, c)
    take = o[rk < np.where(is2[o], CUR_CAP[0] - n2[s1[o]], CUR_CAP[1] - n3[s1[o]])]
    base[take] = True
    return base


CUR_CAP = P.CUR["cap"]


def main():
    t0 = time.time()
    frames = {dr: pd.read_parquet(W / f"{FR[dr]}.parquet", columns=FULL + ["y"]) for dr in FR}
    fr = pd.read_parquet(W / "france.parquet", columns=FULL)
    out = {"adv": adversarial(frames, fr)}
    top5 = [c for c, _, _ in out["adv"]["France_vs_loco"]["top"][:5]]
    out["top5"] = top5
    del fr
    Z, R = {}, {}
    for dr in FR:
        _, z = P.frames(dr)
        assert len(z["s1"]) == len(frames[dr]) and (z["y"] == frames[dr].y.to_numpy()).all()
        p = S.loco_p(z["g"], z["r"], P.CUR["tau"], "chain")
        rb = P.ref_bars(dr, p, z, "chain")
        st, ref = P.run(z, p, rb, steps=True)
        _, z["free"] = S.base(z["s1"], z["is2"], p, z["am"], z["sib"], rb["t"], CUR_CAP)
        Z[dr], R[dr] = z, (st, ref, P.fvec(z, ref))
    variants = {"raw": (FULL, False), "rank5": (FULL, True), "drop5": ([c for c in FULL if c not in top5], False)}
    for a, b in (("India_US", "US_India"), ("US_India", "India_US")):
        for vn, (cols, rk) in variants.items():
            da, db = frames[a], frames[b]
            if rk:
                da, db = rank_cols(da, top5), rank_cols(db, top5)
            m = lgb.train(PAR, lgb.Dataset(X(da, cols), da.y.to_numpy()), 400)
            s = m.predict(X(db, cols))
            zb = Z[b]
            st, ref, fref = R[b]
            res = {"auc_B": round(float(roc_auc_score(zb["y"], s)), 4)}
            for mode in ("add", "addrm"):
                for tau in (0.8, 0.9, 0.95):
                    res[f"{mode}:{tau}"] = P.cmp(zb, ref, decode(zb, st, ref, s, tau, mode), fref)
            key = f"train {a} labels -> {b}"
            out.setdefault(key, {})[vn] = res
            print(key, vn, json.dumps(res), f"{time.time() - t0:.0f}s peak {P.peak():.2f} GB", flush=True)
            (HERE / "arm3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
