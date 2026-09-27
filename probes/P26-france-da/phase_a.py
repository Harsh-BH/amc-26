"""P26 Phase A: adversarial validation, France test vs US/India train, on the pruned candidate lists (laptop, samples).
France = P14's IDF-fixed X (w_idfx) restricted to its #6-pruner keys (p9 file, m2_big p attached); train = a random
pair sample of P4's train X, kept where the pruner keeps it (q >= 0.03 | rule_acc), m2_big p attached.
Reports: domain AUC (5-fold, LightGBM) for France vs train, France vs US, France vs India, US vs India; per-feature
univariate AUC + SMD; AUC on true-pair-like (p >= 0.5) vs look-alike (p < 0.5) rows; AUC after dropping candidate sets.
    .venv/bin/python probes/P26-france-da/phase_a.py        (~1 GB, a few minutes)
-> data/work/p26/phase_a.json, data/work/p26/phase_a_feats.csv
"""
import json, resource, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p26"
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
N = 120_000                                   # pruned pairs per domain
rng = np.random.default_rng(0)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def sample(files, rate, keep_fn, cols):
    parts = []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=200_000, columns=cols):
            d = b.to_pandas()
            d = d[rng.random(len(d)) < rate]
            d = keep_fn(d)
            if len(d):
                parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    cols = [c for c in pq.read_schema(W / "p4/X/train_US_sx_000.parquet").names if c not in DROP]
    m2 = lgb.Booster(model_file=str(W / "p4/m2_big.txt"))
    pr = lgb.Booster(model_file=str(W / "p10/pruner.txt"))
    assert m2.feature_name() == cols, "m2_big features = X columns"
    # France: pruned keys with m2_big p
    k = pd.read_parquet(W / "p14/w_idfx/p9/m2_test_France_d2b6569f_6a23c2ec_0.03.parquet")
    key = pd.Series(k.p.to_numpy(), index=(k.q1.to_numpy(np.int64) << 32) | k.qx.to_numpy(np.int64))
    del k

    def fr_keep(d):
        kk = (d.q1.to_numpy(np.int64) << 32) | d.qx.to_numpy(np.int64)
        p = key.reindex(kk).to_numpy()
        return d.assign(p=p)[~np.isnan(p)]

    fr = sample(sorted((W / "p14/w_idfx/p4/X").glob("test_France_sx_*.parquet")), N / 1.30e6 * 1.02, fr_keep,
                ["q1", "qx"] + cols)
    del key
    print(f"France sample {len(fr):,} ({time.time() - t0:.0f}s, peak {peak():.2f} GB)", flush=True)

    def tr_keep(d):
        q = pr.predict(d[pr.feature_name()].to_numpy(np.float32))
        d = d[(q >= 0.03) | (d.rule_acc.to_numpy() > 0)]
        return d.assign(p=m2.predict(d[cols].to_numpy(np.float32)).astype(np.float32)) if len(d) else d

    tr = {}
    for c, n in (("US", 18.93e6), ("India", 12.47e6)):
        tr[c] = sample(sorted((W / "p4/X").glob(f"train_{c}_sx_*.parquet")), N / n * 3.3, tr_keep,
                       ["q1", "qx", "is_true", "fold"] + cols)
        print(f"{c} sample {len(tr[c]):,} (true share {tr[c].is_true.mean():.3f}; {time.time() - t0:.0f}s, "
              f"peak {peak():.2f} GB)", flush=True)
    return fr, tr, cols


def cv_auc(a, b, cols, imp=False, folds=5):
    X = pd.concat([a[cols], b[cols]], ignore_index=True).to_numpy(np.float32)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    oof, gain = np.zeros(len(y)), np.zeros(len(cols))
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               verbose=-1, num_threads=4)
    for trn, tst in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        m = lgb.train(prm, lgb.Dataset(X[trn], y[trn]), 200)
        oof[tst] = m.predict(X[tst])
        gain += m.feature_importance("gain")
    out = {"auc": roc_auc_score(y, oof)}
    if imp:
        out["gain"] = pd.Series(gain / gain.sum(), index=cols)
    return out, oof


if __name__ == "__main__":
    t0 = time.time()
    fr, tr, cols = main()
    both = pd.concat([tr["US"], tr["India"]], ignore_index=True)
    # balance: train pool mixed at test's non-France mix 55% India / 45% US
    n = min(len(fr), len(tr["US"]), len(tr["India"]))
    mix = pd.concat([tr["India"].sample(int(0.55 * n), random_state=0), tr["US"].sample(int(0.45 * n), random_state=0)],
                    ignore_index=True)
    fr = fr.sample(n, random_state=0).reset_index(drop=True)
    res = {"n_per_domain": n}
    r, oof = cv_auc(fr, mix, cols, imp=True)
    res["auc_France_vs_train"] = r["auc"]
    # true-pair-like vs look-alike (m2_big p) -- AUC within each stratum from the same OOF scores
    y = np.r_[np.ones(len(fr)), np.zeros(len(mix))]
    p = np.r_[fr.p.to_numpy(), mix.p.to_numpy()]
    for tag, m in (("p>=0.5", p >= 0.5), ("p<0.5", p < 0.5), ("p>=0.95", p >= 0.95), ("band0.02-0.98", (p > .02) & (p < .98))):
        res[f"auc_France_vs_train_{tag}"] = roc_auc_score(y[m], oof[m])
        res[f"share_{tag}_France"], res[f"share_{tag}_train"] = float(m[y == 1].mean()), float(m[y == 0].mean())
    res["auc_France_vs_US"] = cv_auc(fr, tr["US"].sample(n, random_state=1), cols)[0]["auc"]
    res["auc_France_vs_India"] = cv_auc(fr, tr["India"].sample(n, random_state=1), cols)[0]["auc"]
    res["auc_US_vs_India"] = cv_auc(tr["US"].sample(n, random_state=2), tr["India"].sample(n, random_state=2), cols)[0]["auc"]
    # per-feature: univariate AUC (France vs train mix), SMD, gain share
    rows = []
    for c in cols:
        a, b = fr[c].to_numpy(np.float64), mix[c].to_numpy(np.float64)
        v = np.r_[a, b]
        ok = ~np.isnan(v)
        u = roc_auc_score(y[ok], v[ok]) if ok.all() or ok.sum() > 1000 else np.nan
        smd = (np.nanmean(a) - np.nanmean(b)) / np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2 + 1e-12)
        # same stat US vs India (is the feature a country id already inside train?)
        a2, b2 = tr["US"][c].to_numpy(np.float64), tr["India"][c].to_numpy(np.float64)
        smd2 = (np.nanmean(a2) - np.nanmean(b2)) / np.sqrt((np.nanvar(a2) + np.nanvar(b2)) / 2 + 1e-12)
        rows.append({"feat": c, "uni_auc": u, "abs_uni": abs(u - 0.5) if u == u else np.nan, "smd_fr_vs_train": smd,
                     "smd_us_vs_india": smd2, "gain": r["gain"][c], "nan_fr": np.isnan(a).mean(), "nan_tr": np.isnan(b).mean()})
    F = pd.DataFrame(rows).sort_values("gain", ascending=False)
    F.to_csv(OUT / "phase_a_feats.csv", index=False)
    print(F.head(25).round(3).to_string(index=False), flush=True)
    # drop sets
    sets = {"-context": ["s1_n", "s1_gap", "s1_margin", "sx_n", "sx_gap", "sx_margin", "block_rank_s1", "block_rank_sx"],
            "-chain": ["chain_1", "chain_x"], "-state": ["state_eq", "state_both", "state_diff"],
            "-place": ["place_n_1", "place_n_x", "place_inter"], "-indic": ["indic_x"]}
    top = F.feat.head(10).tolist()
    for i in (3, 5, 10):
        sets[f"-top{i}gain"] = top[:i]
    for tag, dcols in sets.items():
        keep = [c for c in cols if c not in dcols]
        res[f"auc_France_vs_train{tag}"] = cv_auc(fr, mix, keep, folds=3)[0]["auc"]
        print(f"{tag}: {res[f'auc_France_vs_train{tag}']:.4f}", flush=True)
    res |= {"runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2), "top_gain": top}
    (OUT / "phase_a.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
