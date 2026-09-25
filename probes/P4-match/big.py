"""Improvement round (MERGE-2): the single-stage M2 recipe trained on MORE folds, blended with P4b's cross-encoder
exactly like submission #4, compared with #4 on P5's report half.

    big.py train [2,3,4,5]      # M2 params, early stop on the fold-1 sample -> data/work/p4/m2_big.txt
    big.py holdout              # -> data/work/p4/prob_holdout_big.parquet (prob_holdout schema; argmax over ALL S1)
    flock /tmp/amc-gpu.lock big.py ce     # CE r for new band pairs (not in p4b/r_holdout) -> p4/r_holdout_big.parquet
    big.py eval                 # #4 control (M2) vs big, same code: blend fit + D3 tuned on half 0, report half 1
    big.py test                 # (only if adopted) prob_test_<c>_big for France / India / US
    flock /tmp/amc-gpu.lock big.py blendtest India    # (adopted) CE blend with the refit w -> prob_test_<c>_big_blend

Training rows are held as float16 LightGBM Sequences (1.6 GB for folds 2-5 instead of 3.3 GB of float32);
LightGBM bins them itself, so the float32 matrix never exists. P4b / P5 code is imported from the main checkout
(read-only): rerank.py (blend, fit_blend, d3, apply, texts, score), bench.py (load, select, metrics, per_s1).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = Path("/home/harsh1/github-repos/amc-26")       # P4b / P5 code lives on main (commit d8e7e11)
sys.path[:0] = [str(HERE), str(ROOT / "probes/P3-block"), str(ROOT)]
import model as M                                      # before anything puts src/ (which has a model.py) first
from featrun import trim
from full import peak_gb
from train import COUNTRIES, ID, OFF1, OFF2, arrays, load, with_ids, xfiles

OUT = ROOT / "data/work/p4"
BIG = OUT / "m2_big.txt"
PROB = OUT / "prob_holdout_big.parquet"
RX = OUT / "r_holdout_big.parquet"                     # CE scores for band pairs P4b's cache lacks
P4B = ROOT / "data/work/p4b"


def _p4b():
    sys.path.insert(0, str(MAIN / "probes/P4b-rerank"))
    import rerank
    return rerank


def _bench():
    sys.path.insert(0, str(MAIN / "probes/P5-decide"))
    import bench
    return bench


def train(folds="2,3,4,5", val_frac=0.25):
    """M2 (model.PARAMS, 127 leaves, lr 0.05) on the given train folds; fold 1 sample for early stopping."""
    import lightgbm as lgb
    from ledger import log
    folds = [int(f) for f in str(folds).split(",")]
    assert not {0, 1} & set(folds), "never train on folds 0/1"
    t0 = time.time()

    class Seq(lgb.Sequence):
        def __init__(self, a):
            self.a, self.batch_size = a, 65_536

        def __getitem__(self, i):
            return self.a[i].astype(np.float64)       # LightGBM samples rows as double

        def __len__(self):
            return len(self.a)

    cols = [c for c in pq.read_schema(xfiles("train", "India")[0]).names if c not in ID]
    seqs, ys = [], []
    for c in COUNTRIES:
        for p in xfiles("train", c):
            t = pq.read_table(p, columns=cols + ["is_true", "fold"], filters=[("fold", "in", folds)])
            a = np.empty((t.num_rows, len(cols)), np.float16)
            for k, col in enumerate(cols):
                a[:, k] = t.column(col).to_numpy()
            seqs.append(Seq(a))
            ys.append(t.column("is_true").to_numpy())
            del t
            trim()
    y = np.concatenate(ys).astype(np.float32)
    del ys
    print(f"train {len(y):,} pairs ({y.mean():.3f} true) folds {folds}, {len(cols)} feats, load {time.time() - t0:.0f}s,"
          f" peak {peak_gb():.2f} GB", flush=True)
    va = load((1,), frac=val_frac)                     # the SAME fold-1 sample the ladder used (hash on q1)
    Xva, yva = arrays(va, cols)
    del va
    d = lgb.Dataset(seqs, label=y, feature_name=cols, free_raw_data=True)
    dv = lgb.Dataset(Xva, yva.astype(np.float32), reference=d)
    d.construct()
    del seqs
    trim()
    print(f"dataset built {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
    t1 = time.time()
    m = lgb.train(M.PARAMS, d, 8000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    fit_s = time.time() - t1
    m.save_model(str(BIG))
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    print(f"m2_big: {m.num_trees()} trees, fit {fit_s:.0f}s, best val logloss "
          f"{m.best_score['valid_0']['binary_logloss']:.5f}, peak {peak_gb():.2f} GB", flush=True)
    log("exp", probe="P4-match", subproblem="matcher", name="M2-big-train", world="train",
        metrics={"n_train": int(len(y)), "trees": m.num_trees(), "val_logloss": round(m.best_score["valid_0"]["binary_logloss"], 6),
                 "fit_s": round(fit_s)},
        config={"params": M.PARAMS, "train_folds": folds, "early_stop": "fold-1 25% S1 sample, patience 100",
                "storage": "float16 lgb.Sequence", "top_gain": (imp / imp.sum()).round(4).head(10).to_dict()},
        hypothesis="4x the training pairs lets the same M2 recipe recover more true pairs", verdict="inconclusive",
        runtime_s=round(time.time() - t0), peak_gb=peak_gb())


def holdout():
    """p for every pair of every SX that has a fold-0 candidate (so the argmax runs over ALL its train S1),
    written for the fold-0 pairs in prob_holdout's schema."""
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(BIG))
    t0, parts, n = time.time(), [], 0
    for c in COUNTRIES:
        for p in xfiles("train", c):
            X = pd.read_parquet(p)
            f0 = X.fold.to_numpy() == 0
            rel = np.isin(X.qx.to_numpy(), np.unique(X.qx.to_numpy()[f0]))
            X = X[rel].reset_index(drop=True)
            pr = M.predict(m, X)
            n += len(X)
            qx = X.qx.to_numpy()
            am = M.argmax_sx(X.q1.to_numpy(), qx, pr)
            best = pd.Series(pr).groupby(qx).transform("max").to_numpy(np.float32)
            k = X.fold.to_numpy() == 0
            d = pd.DataFrame({"q1": X.q1.to_numpy()[k], "qx": qx[k], "p": pr[k], "is_true": X.is_true.to_numpy()[k],
                              "am": am[k], "sx_best_p": best[k], "blocked": X.blocked.to_numpy()[k] > 0})
            parts.append(with_ids(d, "train", c).assign(country=c))
            del X
            trim()
        print(f"{c}: scored, {n / (time.time() - t0):,.0f} pairs/s, peak {peak_gb():.2f} GB", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d[["s1", "sx", "p", "is_true", "country", "am", "sx_best_p", "blocked"]].to_parquet(PROB, index=False)
    print(f"{PROB.name}: {len(d):,} pairs ({n:,} scored for the argmax) in {time.time() - t0:.0f}s", flush=True)


def _r(ids: pd.DataFrame) -> np.ndarray:
    """CE score per (s1, sx) from P4b's cache, then this round's extra cache; NaN if neither has it."""
    r = ids.merge(pd.read_parquet(P4B / "r_holdout.parquet"), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
    if RX.exists():
        r2 = ids.merge(pd.read_parquet(RX), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
        r = np.where(np.isnan(r), r2, r)
    return r


def ce():
    """CE r for the big model's band pairs (0.02 < p < 0.98, CE countries) that no cache has. GPU: run under flock."""
    rr = _p4b()
    t0 = time.time()
    d = pd.read_parquet(PROB, columns=["s1", "sx", "p", "country"])
    lo, hi = rr.CFG["band"]
    d = d[(d.p > lo) & (d.p < hi) & d.country.isin(rr.CFG["countries"])].reset_index(drop=True)
    miss = np.isnan(_r(d[["s1", "sx"]]))
    d = d[miss]
    print(f"band pairs {len(miss):,}, without a cached r {len(d):,}", flush=True)
    out = [pd.read_parquet(RX)] if RX.exists() else []
    tok = m = None
    for c, g in d.groupby("country"):
        q1, qx = rr.rows("train", c, g.s1, g.sx)
        ta, tb = rr.texts("train", c, q1, qx)
        if m is None:
            tok, m = rr.load_model(P4B / "ce.pt")
        t1 = time.time()
        out.append(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "r": rr.score(ta, tb, tok, m)}))
        print(f"{c}: {len(g):,} pairs at {len(g) / (time.time() - t1):,.0f} pairs/s", flush=True)
    if out:
        pd.concat(out, ignore_index=True).drop_duplicates(["s1", "sx"]).to_parquet(RX, index=False)
    print(f"ce done {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


def _blend_eval(path: Path):
    """rerank.evaluate's 'blend-0.02-0.98' run on one probability file: w fit on band pairs of half 0, D3 (t, t0)
    tuned on half 0, report half 1. Returns (metrics, cfg, w, per-S1 F on every holdout S1)."""
    rr, B = _p4b(), _bench()
    B.SRC["_x"] = path
    d, S = B.load("_x")
    r = _r(pd.read_parquet(path, columns=["s1", "sx"]))
    p, y, h = d.p.to_numpy(), d.is_true.to_numpy(), S.half.to_numpy()[d.s1.to_numpy()]
    lo, hi = rr.CFG["band"]
    b = (p > lo) & (p < hi) & ~np.isnan(r)
    inband = (p > lo) & (p < hi) & d.country.isin(rr.CFG["countries"]).to_numpy()
    w = rr.fit_blend(p[b & (h == 0)], r[b & (h == 0)], y[b & (h == 0)])
    dd = d.assign(p=rr.blend(p, r, w, lo, hi))
    c = rr.d3(B, dd, S)
    sel = B.select(dd, c)
    m = B.metrics(dd, S, sel, 1)
    m["band_pairs"], m["band_unscored"] = int(inband.sum()), int((inband & np.isnan(r)).sum())
    f, _, _ = B.per_s1(dd, S, sel)
    base = B.metrics(d, S, B.select(d, rr.d3(B, d, S)), 1)["f05"]      # the same file without the CE
    return m | {"f05_no_ce": base}, c, w, f, S


def eval_():
    from ledger import log
    t0 = time.time()
    res = {}
    for nm, path in (("control-M2", OUT / "prob_holdout_m2.parquet"), ("big", PROB)):
        m, c, w, f, S = _blend_eval(path)
        res[nm] = (m, c, w, f)
        print(nm, f"t={c['t']:.2f} t0={c['t0']:.2f} w={np.round(w, 3).tolist()}",
              {k: round(float(v), 4) for k, v in m.items()}, flush=True)
    h1 = S.half.to_numpy() == 1
    dif = (res["big"][3] - res["control-M2"][3])[h1]
    gain, se = float(dif.mean()), float(dif.std(ddof=1) / np.sqrt(len(dif)))
    adopt = gain >= 0.001 and gain > 3 * se
    print(f"big vs #4 control on half 1: {gain:+.5f} +- {se:.5f} (1 se), {(dif > 0).sum():,} S1 up, "
          f"{(dif < 0).sum():,} down -> {'ADOPT' if adopt else 'do not adopt'}", flush=True)
    for nm, (m, c, w, _) in res.items():
        log("exp", probe="P4-match", subproblem="matcher+rerank", name=f"improve-{nm}-ce-D3", world="holdout-dense-half1",
            metrics={k: round(float(v), 4) for k, v in m.items()}
            | ({"gain_vs_control": round(gain, 5), "gain_se": round(se, 5)} if nm == "big" else {}),
            config={"t": c["t"], "t0": c["t0"], "blend_w": [round(x, 4) for x in w], "band": [0.02, 0.98],
                    "src": "prob_holdout_m2 (#4)" if nm != "big" else "prob_holdout_big (M2 on folds 2-5)",
                    "code": "P4b rerank.fit_blend/blend/d3 + P5 bench.select/metrics (main d8e7e11)"},
            hypothesis="M2 on 4x the training folds, CE-blended like #4, beats #4 by >= 0.001 and > 3 paired se"
            if nm == "big" else "#4 control reproduced on the same code (expected 0.9881)",
            verdict=("keep" if adopt else "drop") if nm == "big" else "keep", runtime_s=round(time.time() - t0),
            peak_gb=peak_gb())
    json.dump({nm: {"metrics": {k: float(v) for k, v in m.items()}, "cfg": c, "w": w}
               for nm, (m, c, w, _) in res.items()} | {"gain": gain, "se": se, "adopt": adopt},
              open(OUT / "improve_eval.json", "w"), indent=1)


def test():
    """Adopted only: big-model p for every test pair (argmax over all test S1) -> prob_test_<c>_big (no GPU)."""
    import lightgbm as lgb
    from train import predict_split
    ev = json.load(open(OUT / "improve_eval.json"))
    assert ev["adopt"] or ev.get("adopt_master"), "test files are written only for an adopted model"
    m = lgb.Booster(model_file=str(BIG))
    for c in ("France", "India", "US"):
        t0 = time.time()
        d = with_ids(predict_split(lambda X: M.predict(m, X), "test", c), "test", c)
        d[["s1", "sx", "p", "am", "sx_best_p", "blocked"]].to_parquet(OUT / f"prob_test_{c}_big.parquet", index=False)
        print(f"prob_test_{c}_big: {len(d):,} pairs in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
        del d
        trim()


W_BIG = [0.678, 0.715, 0.338]           # MERGE-2 ruling: the refit weights (improve_eval.json), rounded like #4's CFG


def blendtest(c, n_check=5_000):
    """Own process, GPU (flock): #4's CE blend on prob_test_<c>_big with W_BIG -> prob_test_<c>_big_blend
    (p = blended, p_gbdt = the big model's p). CE countries only.
    CE scores are REUSED where #4 already computed them: #4's rescore blended M2's band (0.02 < p_m2 < 0.98) with
    CFG w and kept only p', so r is recovered exactly by inverting the blend, r = sigmoid((logit p' - a logit p_m2
    - c) / b); a fresh CE score of n_check of those pairs checks the inversion. Only big-band pairs outside M2's band
    reach the GPU."""
    rr = _p4b()
    ev = json.load(open(OUT / "improve_eval.json"))
    assert (ev["adopt"] or ev.get("adopt_master")) and c in rr.CFG["countries"]
    t0 = time.time()
    lo, hi = rr.CFG["band"]
    d = pq.read_table(OUT / f"prob_test_{c}_big.parquet")
    m2 = pq.read_table(OUT / f"prob_test_{c}_m2.parquet", columns=["s1", "sx", "p"])          # #4's GBDT input
    b4 = pq.read_table(P4B / f"prob_test_{c}_blend.parquet", columns=["s1", "sx", "p"])       # #4's output
    for t in (m2, b4):                               # same rows, same order: all three come from P4's X file order
        assert t.column("s1").equals(d.column("s1")) and t.column("sx").equals(d.column("sx")), "row order"
    p, pm, pb = (t.column("p").to_numpy().astype(np.float64) for t in (d, m2, b4))
    del m2, b4
    a, b, cc = rr.CFG["w"]
    old = (pm > lo) & (pm < hi)
    assert np.allclose(pb[~old], pm[~old]), "outside #4's band p' == p_m2"
    r = np.full(len(p), np.nan, np.float32)
    r[old] = 1 / (1 + np.exp(-(rr.logit(pb[old]) - a * rr.logit(pm[old]) - cc) / b))
    new = (p > lo) & (p < hi) & ~old
    s1, sx = d.column("s1").to_pandas(), d.column("sx").to_pandas()
    tok, m = rr.load_model(P4B / "ce.pt")
    chk = np.random.default_rng(0).choice(np.flatnonzero(old), min(n_check, int(old.sum())), replace=False)
    for rows_, nm in ((chk, "check"), (np.flatnonzero(new), "new")):
        if not len(rows_):
            continue
        q1, qx = rr.rows("test", c, s1.iloc[rows_], sx.iloc[rows_])
        rs = rr.score(*rr.texts("test", c, q1, qx), tok, m)
        if nm == "check":
            dz = np.abs(rr.logit(rs) - rr.logit(r[rows_]))
            print(f"{c}: inverted vs fresh CE on {len(rows_):,} #4 band pairs: |d logit r| median {np.median(dz):.2e}, "
                  f"p99 {np.percentile(dz, 99):.2e}, max {dz.max():.2e}; |d r| max {np.abs(rs - r[rows_]).max():.2e}",
                  flush=True)
            assert np.percentile(dz, 99) < 0.05, "blend inversion does not reproduce the cached CE scores"
        else:
            r[rows_] = rs
    inband = (p > lo) & (p < hi)
    out = d.to_pandas()
    out["p_gbdt"] = out.p
    out["p"] = rr.blend(p.astype(np.float32), r, W_BIG, lo, hi)
    out.to_parquet(OUT / f"prob_test_{c}_big_blend.parquet", index=False)
    print(f"prob_test_{c}_big_blend: {len(out):,} pairs, band {int(inband.sum()):,} = {int((inband & old).sum()):,} "
          f"reused from #4 + {int(new.sum()):,} scored now; {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
    return {"pairs": len(out), "band": int(inband.sum()), "reused": int((inband & old).sum()), "scored": int(new.sum())}


def profile(t_fr: str, t_fr4: str = "0.82"):
    """Test answer profile per country under the adopted decision (P5 select: each SX to its argmax S1 over all
    test S1 via `am`, keep p >= t, S1 answers only if its best p >= t0), for the big files and for shipped #4.
    India / US: blended p, (t, t0) tuned on the holdout tune half; France: GBDT p at P6's label-free t_France."""
    from ledger import log
    sys.path.insert(0, str(MAIN / "probes/P5-decide"))
    from decide import select
    ev = json.load(open(OUT / "improve_eval.json"))
    runs = {"big": ({"India": OUT / "prob_test_India_big_blend.parquet", "US": OUT / "prob_test_US_big_blend.parquet",
                     "France": OUT / "prob_test_France_big.parquet"}, ev["big"]["cfg"], float(t_fr)),
            "sub4": ({"India": P4B / "prob_test_India_blend.parquet", "US": P4B / "prob_test_US_blend.parquet",
                      "France": OUT / "prob_test_France_m2.parquet"}, ev["control-M2"]["cfg"], float(t_fr4))}
    for nm, (files, cfg, tf) in runs.items():
        m = {}
        for c, f in files.items():
            d = pd.read_parquet(f, columns=["s1", "sx", "p", "am"])
            cf = {"excl": True, "t": tf, "t0": tf} if c == "France" else {"excl": True, "t": cfg["t"], "t0": cfg["t0"]}
            sel = select(d, cf)
            n1 = pq.ParquetFile(ROOT / f"data/work/p3/txt_test/{c}_s1.parquet").metadata.num_rows
            nx = d.sx.nunique()
            m |= {f"pairs_{c}": len(d), f"matches_per_s1_{c}": round(float(sel.sum() / n1), 4),
                  f"s1_empty_{c}": round(float(1 - d.s1[sel].nunique() / n1), 4),
                  f"sx_assigned_{c}": round(float(sel.sum() / nx), 4), f"t_{c}": cf["t"], f"t0_{c}": cf["t0"]}
            del d, sel
            trim()
        print(nm, m, flush=True)
        log("exp", probe="P4-match", subproblem="matcher+rerank", name=f"test-profile-{nm}", world="test", metrics=m,
            config={"files": {c: str(f.relative_to(ROOT)) for c, f in files.items()},
                    "decision": "P5 select excl (am) + t + t0; France t0 = t", "blend_w": W_BIG if nm == "big" else "#4 CFG"},
            hypothesis="the big model's test answers keep the holdout profile (~3.36 matches/S1, ~5.7% empty)",
            verdict="keep", runtime_s=0, peak_gb=peak_gb())


if __name__ == "__main__":
    {"train": lambda: train(*sys.argv[2:3]), "holdout": holdout, "ce": ce, "eval": eval_, "test": test,
     "blendtest": lambda: blendtest(sys.argv[2]), "profile": lambda: profile(*sys.argv[2:4])}[sys.argv[1]]()
