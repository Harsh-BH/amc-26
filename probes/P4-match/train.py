"""Model ladder on the full-density train features (featrun.py feat train <c>), scored dense on the holdout.

    train.py ladder            # M0 / M1 / M2 / M3: fit on folds >= 2, early-stop on fold 1, score fold 0
    train.py final             # the kept model -> data/work/p4/{m2.txt, prob_holdout.parquet, prob_dev.parquet}

Scoring streams every X file (one file = every pair of its SX), predicts ALL pairs, and applies the argmax per SX
two ways: over every train S1 (test-faithful: on test all S1 compete) and over the scored fold's S1 only
(= P3's cands_holdout_full files, which hold holdout S1 only). The global threshold is swept for macro F0.5 over
EVERY S1 of the fold, true pairs missing from the candidates counting as recall losses.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P3-block"), str(ROOT)]
import model as M
from full import peak_gb
from featrun import OUT, TXT, WORK

COUNTRIES = ["India", "US"]
ID = ["q1", "qx", "is_true", "fold"]


def xfiles(split, c):
    return sorted((OUT / "X").glob(f"{split}_{c}_sx_*.parquet"))


def load(folds, countries=COUNTRIES, frac: float = 1.0, cols=None, seed=0) -> pd.DataFrame:
    """Train pairs of the given S1 folds (all countries), optionally an S1-level sample."""
    parts = []
    for ci, c in enumerate(countries):
        for p in xfiles("train", c):
            d = pq.read_table(p, columns=cols and ID + cols, filters=[("fold", "in", list(folds))]).to_pandas()
            if frac < 1:
                h = (d.q1.to_numpy(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000)
                d = d[h < frac * 1000]
            parts.append(d.assign(country=np.int8(ci)))
    return pd.concat(parts, ignore_index=True)


def truth_counts(fold) -> dict:
    """{country: Series q1 -> number of true matches} for EVERY S1 of the fold (0 = singleton)."""
    gt = pd.read_parquet(WORK / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    fo = pd.read_parquet(WORK / "s1_folds.parquet")
    out = {}
    for c in COUNTRIES:
        ids1 = pd.read_parquet(TXT["train"] / f"{c}_s1.parquet", columns=["entity_id"]).entity_id
        f = fo.set_index("entity_id").fold.reindex(ids1).to_numpy()
        q = np.flatnonzero(f == fold)
        out[c] = pd.Series(gt.reindex(ids1.iloc[q]).fillna(0).to_numpy(), index=q)
    return out


OFF1, OFF2 = 10_000_000, 100_000_000          # per-country offsets: S1 / SX row ids unique across countries


def score_folds(scorer, folds=(0, 1), countries=COUNTRIES) -> pd.DataFrame:
    """Stream every X file, score all pairs, keep rows of S1 in `folds` with both argmax flags."""
    keep = []
    for ci, c in enumerate(countries):
        for p in xfiles("train", c):
            X = pd.read_parquet(p)
            pr = scorer(X)
            qx, fo = X.qx.to_numpy(), X.fold.to_numpy()
            am_all = M.argmax_sx(X.q1.to_numpy(), qx, pr)
            sel = np.isin(fo, folds)
            best = pd.Series(pr).groupby(qx).transform("max").to_numpy()
            d = pd.DataFrame({"s1": X.q1.to_numpy()[sel].astype(np.int64) + ci * OFF1,
                              "sx": qx[sel].astype(np.int64) + ci * OFF2, "fold": fo[sel], "country": np.int8(ci),
                              "p": pr[sel], "is_true": X.is_true.to_numpy()[sel], "am_all": am_all[sel],
                              "sx_best_p": best[sel], "blocked": X.blocked.to_numpy()[sel]})
            d["am_fold"] = False
            for f in folds:                                   # argmax among this fold's S1 only
                m = d.fold.to_numpy() == f
                d.loc[m, "am_fold"] = M.argmax_sx(d.s1.to_numpy()[m], d.sx.to_numpy()[m], d.p.to_numpy()[m])
            keep.append(d)
            del X
    return pd.concat(keep, ignore_index=True)


def truth_all(fold) -> tuple:
    tc = truth_counts(fold)
    tn = pd.concat([tc[c].set_axis(tc[c].index + ci * OFF1) for ci, c in enumerate(COUNTRIES)])
    grp = pd.Series(np.repeat(COUNTRIES, [len(tc[c]) for c in COUNTRIES]), index=tn.index)
    return tn, grp


def evaluate(S: pd.DataFrame, fold: int, tn: pd.Series, grp: pd.Series) -> dict:
    """Dense macro F0.5 on one fold at one global threshold; argmax over all S1 ('all') or the fold's S1 ('fold')."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = S[S.fold == fold]
    out = {"auc": roc_auc_score(d.is_true, d.p), "ap": average_precision_score(d.is_true, d.p)}
    for how in ("all", "fold"):
        b = M.sweep(d[d[f"am_{how}"]], tn, argmax=False, by=grp)
        out |= {f"f05_{how}": b["f05"], f"t_{how}": b["t"], f"prec_{how}": b["pair_precision"],
                f"rec_{how}": b["pair_recall"]} | {f"f05_{how}_{c}": b[f"f05_{c}"] for c in COUNTRIES}
    return out


def arrays(d: pd.DataFrame, cols) -> tuple:
    return d[cols].to_numpy(np.float32), d.is_true.to_numpy()


def chunked(f, n=400_000):
    """Score a big frame in row chunks (sklearn models upcast to float64)."""
    return lambda X: np.concatenate([f(X.iloc[a:a + n]) for a in range(0, len(X), n)]) if len(X) else np.zeros(0, np.float32)


def ladder(train_folds=(2,), val_frac=0.25, only=None):
    from ledger import log
    from featrun import trim
    t0 = time.time()
    tr = load(train_folds)
    cols = M.feats(tr)
    small = tr.sample(n=min(len(tr), 600_000), random_state=0)
    n_tr, pos = len(tr), float(tr.is_true.mean())
    Xtr, ytr = arrays(tr, cols)
    del tr
    trim()
    va = load((1,), frac=val_frac)
    Xva, yva = arrays(va, cols)
    del va
    trim()
    print(f"train {n_tr:,} pairs ({pos:.3f} true), val {len(yva):,}, {len(cols)} feats, "
          f"load {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
    tn, grp = truth_all(0)
    n_all = sum(pq.ParquetFile(p).metadata.num_rows for c in COUNTRIES for p in xfiles("train", c))
    res = {}

    def run(name, scorer, hyp, cfg, fit_s=0.0):
        t = time.time()
        S = score_folds(scorer, folds=(0,))
        secs = time.time() - t
        m = evaluate(S, 0, tn, grp) | {"score_pairs_per_s": n_all / secs}
        res[name] = m
        print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)
        log("exp", probe="P4-match", subproblem="matcher", name=name, world="holdout-dense-full",
            metrics={k: round(float(v), 4) for k, v in m.items()}, config=cfg, hypothesis=hyp,
            verdict="keep" if max(m["f05_all"], m["f05_fold"]) > 0.8215 else "drop",
            runtime_s=round(fit_s + secs, 1), peak_gb=peak_gb())
        return S

    todo = only.split(",") if only else ["M0", "M3", "M1", "M2"]
    if "M0" in todo:
        run("M0-hand-rule", M.m0_score, "a hand-weighted name+address score with argmax per SX",
            {"weights": "0.35 name_cos + 0.15 core_tset + 0.25 addr_cos + 0.15 addr_rule/7 + 0.10 max(core_eq, web_eq)"})
    if "M3" in todo:
        t = time.time()
        fsp = M.fs_fit(small)
        run("M3-fellegi-sunter-em", lambda X: M.fs_score(fsp, X), "label-free FS-EM (Splink's model) as a baseline",
            {"levels": "name_cos/core/addr_cos/hn/city/state/web", "fit": "EM on 600k train pairs, no labels",
             "lambda": round(float(fsp[0]), 4)}, time.time() - t)
    if "M1" in todo:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        t = time.time()
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(small[cols].fillna(-1), small.is_true)
        run("M1-logreg", chunked(lambda X: lr.predict_proba(X[cols].fillna(-1))[:, 1].astype(np.float32)),
            "linear model on the same features", {"fit": f"600k train pairs, folds {list(train_folds)}", "fillna": -1},
            time.time() - t)
    del small
    if "M2" in todo:
        t = time.time()
        m2 = M.fit(Xtr, ytr, Xva, yva, rounds=3000, cols=cols)
        fit_s = time.time() - t
        del Xtr, Xva
        trim()
        imp = pd.Series(m2.feature_importance("gain"), index=m2.feature_name()).sort_values(ascending=False)
        print(f"M2 {m2.num_trees()} trees in {fit_s:.0f}s; top gain:", (imp / imp.sum()).round(4).head(25).to_dict(),
              flush=True)
        m2.save_model(str(OUT / "m2_ladder.txt"))
        S = run("M2-lightgbm", lambda X: M.predict(m2, X), "GBDT on name/address/context/rule features (Foursquare 1st)",
                {"params": M.PARAMS, "trees": m2.num_trees(), "train_folds": list(train_folds), "n_train": n_tr,
                 "top_gain": (imp / imp.sum()).round(4).head(15).to_dict()}, fit_s)
        S.to_parquet(OUT / "S_ladder_m2.parquet", index=False)
        (OUT / "imp_m2.json").write_text(json.dumps((imp / imp.sum()).round(5).to_dict(), indent=0))
    (OUT / "ladder.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- stage 2 (Foursquare-style re-scoring)
P1DIR = OUT / "p1"


def _rank_desc(key: np.ndarray, p: np.ndarray) -> np.ndarray:
    import block as B
    o = np.lexsort((-p, key))
    r = np.empty(len(p), np.float32)
    r[o] = B.ranks(key[o])
    return r


def _side(key: np.ndarray, p: np.ndarray, pre: str) -> dict:
    """Competition among one S1's (or one SX's) candidates, by stage-1 probability."""
    u, inv = np.unique(key, return_inverse=True)
    mx = np.full(len(u), -1.0, np.float32)
    np.maximum.at(mx, inv, p)
    rk = _rank_desc(key, p)
    top = rk == 0
    sec = np.full(len(u), -1.0, np.float32)
    np.maximum.at(sec, inv[~top], p[~top])
    other = np.where(top, sec[inv], mx[inv])
    return {f"{pre}_p1_rank": rk, f"{pre}_p1_other": other, f"{pre}_p1_margin": p - other,
            f"{pre}_p1_n05": np.bincount(inv, p >= 0.5)[inv].astype(np.float32),
            f"{pre}_p1_sum": np.bincount(inv, p)[inv].astype(np.float32)}


def p1_pass(ma, mb=None, mb_fold=2, split="train", countries=COUNTRIES):
    """Stage-1 probability for EVERY pair of a split, out of fold on train: model A everywhere except the rows A was
    trained on (fold mb_fold), which get model B (trained elsewhere); test gets A. Then the SX side per file and
    the S1 side over all files of the country -> data/work/p4/p1/<X file name>."""
    P1DIR.mkdir(parents=True, exist_ok=True)
    for c in countries:
        q1s, lens, files = [], [], sorted((OUT / "X").glob(f"{split}_{c}_sx_*.parquet"))
        p1s = []
        for p in files:
            X = pd.read_parquet(p)
            p1 = M.predict(ma, X)
            if mb is not None:
                f2 = X.fold.to_numpy() == mb_fold
                p1[f2] = M.predict(mb, X[f2])
            q1s.append(X.q1.to_numpy()), p1s.append(p1), lens.append(len(X))
            pd.DataFrame({"p1": p1, **_side(X.qx.to_numpy(), p1, "sx")}).to_parquet(P1DIR / p.name, index=False)
            del X
        q1, p1 = np.concatenate(q1s), np.concatenate(p1s)
        s1 = pd.DataFrame(_side(q1, p1, "s1"))
        for p, a, b in zip(files, np.cumsum([0] + lens[:-1]), np.cumsum(lens)):
            d = pd.read_parquet(P1DIR / p.name)
            pd.concat([d, s1.iloc[a:b].reset_index(drop=True)], axis=1).to_parquet(P1DIR / p.name, index=False)
        print(f"p1 pass {split} {c}: {len(q1):,} pairs, peak {peak_gb():.2f} GB", flush=True)


def write_prob_dev():
    """prob_dev.parquet (fold 1 = the dev world's fold) from the stage-1 pass: p = M2 (never trained on fold 1),
    am = the SX's argmax over ALL train S1, sx_best_p = the SX's best p."""
    parts = []
    for c in COUNTRIES:
        for p in xfiles("train", c):
            X = pq.read_table(p, columns=["q1", "qx", "is_true", "fold", "blocked"], filters=[("fold", "==", 1)]
                              ).to_pandas()
            Z = pq.read_table(P1DIR / p.name, columns=["p1", "sx_p1_rank", "sx_p1_other"]).to_pandas()
            Z = Z[pq.read_table(p, columns=["fold"]).column(0).to_numpy() == 1].reset_index(drop=True)
            parts.append(with_ids(X.assign(p=Z.p1.to_numpy(), am=Z.sx_p1_rank.to_numpy() == 0,
                                           sx_best_p=np.maximum(Z.p1, Z.sx_p1_other).to_numpy(np.float32)),
                                  "train", c).assign(country=c))
    d = pd.concat(parts, ignore_index=True)
    d["blocked"] = d.blocked > 0
    d[["s1", "sx", "p", "is_true", "country", "am", "sx_best_p", "blocked"]].to_parquet(OUT / "prob_dev.parquet",
                                                                                        index=False)
    print(f"prob_dev: {len(d):,} pairs", flush=True)


def load2(folds, frac=1.0):
    """Stage-2 rows: the X features + the stage-1 competition features, for the given S1 folds."""
    parts = []
    for ci, c in enumerate(COUNTRIES):
        for p in xfiles("train", c):
            X, Z = pd.read_parquet(p), pd.read_parquet(P1DIR / p.name)
            m = np.isin(X.fold.to_numpy(), folds)
            if frac < 1:
                m &= (X.q1.to_numpy(np.uint64) * np.uint64(2654435761)) % np.uint64(1000) < frac * 1000
            parts.append(pd.concat([X[m].reset_index(drop=True), Z[m].reset_index(drop=True)], axis=1)
                         .assign(country=np.int8(ci)))
            del X, Z
    return pd.concat(parts, ignore_index=True)


def score2(model) -> pd.DataFrame:
    """score_folds for the stage-2 model (reads X + p1 files together)."""
    keep = []
    for ci, c in enumerate(COUNTRIES):
        for p in xfiles("train", c):
            X = pd.concat([pd.read_parquet(p), pd.read_parquet(P1DIR / p.name)], axis=1)
            pr = M.predict(model, X)
            qx, fo = X.qx.to_numpy(), X.fold.to_numpy()
            am_all = M.argmax_sx(X.q1.to_numpy(), qx, pr)
            sel = np.isin(fo, (0, 1))
            d = pd.DataFrame({"s1": X.q1.to_numpy()[sel].astype(np.int64) + ci * OFF1,
                              "sx": qx[sel].astype(np.int64) + ci * OFF2, "fold": fo[sel], "country": np.int8(ci),
                              "p": pr[sel], "p1": X.p1.to_numpy()[sel], "is_true": X.is_true.to_numpy()[sel],
                              "am_all": am_all[sel], "blocked": X.blocked.to_numpy()[sel]})
            d["am_fold"] = False
            for f in (0, 1):
                m = d.fold.to_numpy() == f
                d.loc[m, "am_fold"] = M.argmax_sx(d.s1.to_numpy()[m], d.sx.to_numpy()[m], d.p.to_numpy()[m])
            keep.append(d)
            del X
    return pd.concat(keep, ignore_index=True)


FAST = {"learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200}


def _fit_fold(fold, name, params, val_frac=0.25):
    """LightGBM on one train fold, early-stopped on a fold-1 sample; cached as data/work/p4/<name>.txt."""
    import lightgbm as lgb
    from featrun import trim
    path = OUT / f"{name}.txt"
    if not path.exists():
        t = time.time()
        tr = load((fold,))
        cols = M.feats(tr)
        Xtr, ytr = arrays(tr, cols)
        del tr
        va = load((1,), frac=val_frac)
        Xva, yva = arrays(va, cols)
        del va
        trim()
        m = M.fit(Xtr, ytr, Xva, yva, rounds=3000, cols=cols, params=params)
        m.save_model(str(path))
        print(f"{name}: {m.num_trees()} trees in {time.time() - t:.0f}s", flush=True)
        del Xtr, Xva, m
        trim()
    return lgb.Booster(model_file=str(path))


def s_from_p1(countries=COUNTRIES) -> pd.DataFrame:
    """score_folds-shaped frame for folds 0/1 from the stage-1 pass files (p = p1, argmax over all S1)."""
    keep = []
    for ci, c in enumerate(countries):
        for p in xfiles("train", c):
            X = pq.read_table(p, columns=["q1", "qx", "fold", "is_true", "blocked"]).to_pandas()
            Z = pq.read_table(P1DIR / p.name, columns=["p1", "sx_p1_rank"]).to_pandas()
            m = np.isin(X.fold.to_numpy(), (0, 1))
            keep.append(pd.DataFrame({"s1": X.q1.to_numpy()[m].astype(np.int64) + ci * OFF1,
                                      "sx": X.qx.to_numpy()[m].astype(np.int64) + ci * OFF2,
                                      "fold": X.fold.to_numpy()[m], "country": np.int8(ci), "p": Z.p1.to_numpy()[m],
                                      "is_true": X.is_true.to_numpy()[m], "am_all": Z.sx_p1_rank.to_numpy()[m] == 0,
                                      "blocked": X.blocked.to_numpy()[m]}))
    d = pd.concat(keep, ignore_index=True)
    d["am_fold"] = d.am_all
    return d


def stage2(fold_a=2, fold_b=6, fold_2=(3,), val_frac=0.25):
    """Stage 1 = fast LightGBM on fold_a (+ a twin on fold_b for fold_a's own rows, so every p1 is out of fold);
    stage 2 = LightGBM on the features + stage-1 competition features (p1 rank / margin / counts per SX and per
    S1), trained on fold_2. Both scored dense on fold 0 with the fixed decision rule."""
    from ledger import log
    from featrun import trim
    t0 = time.time()
    ma = _fit_fold(fold_a, f"m2fast_fold{fold_a}", FAST)
    mb = _fit_fold(fold_b, f"m2fast_fold{fold_b}", FAST)
    if not all((P1DIR / p.name).exists() for c in COUNTRIES for p in xfiles("train", c)):
        t = time.time()
        p1_pass(ma, mb, fold_a)
        n_all = sum(pq.ParquetFile(p).metadata.num_rows for c in COUNTRIES for p in xfiles("train", c))
        print(f"p1 pass: {n_all / (time.time() - t):,.0f} pairs/s", flush=True)
    tn, grp = truth_all(0)
    S = s_from_p1()
    r1 = evaluate(S, 0, tn, grp)
    del S
    print("M2-fast", {k: round(v, 4) for k, v in r1.items()}, flush=True)
    log("exp", probe="P4-match", subproblem="matcher", name="M2-lightgbm-fast", world="holdout-dense-full",
        metrics={k: round(float(v), 4) for k, v in r1.items()},
        config={"params": {**M.PARAMS, **FAST}, "trees": ma.num_trees(), "train_folds": [fold_a]},
        hypothesis="a 63-leaf lr-0.1 model is nearly as good and several times faster to score",
        verdict="inconclusive", runtime_s=round(time.time() - t0, 1), peak_gb=peak_gb())
    tr, va = load2(fold_2), load2((1,), frac=val_frac)
    cols = M.feats(tr)
    Xtr, ytr = arrays(tr, cols)
    del tr
    Xva, yva = arrays(va, cols)
    del va
    trim()
    t = time.time()
    m = M.fit(Xtr, ytr, Xva, yva, rounds=3000, cols=cols, params=FAST)
    fit_s = time.time() - t
    del Xtr, Xva
    trim()
    m.save_model(str(OUT / "m2_stage2.txt"))
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    S = score2(m)
    r = evaluate(S, 0, tn, grp)
    print("M2-stage2", {k: round(v, 4) for k, v in r.items()}, (imp / imp.sum()).round(4).head(20).to_dict(), flush=True)
    log("exp", probe="P4-match", subproblem="matcher", name="M2-lightgbm-stage2", world="holdout-dense-full",
        metrics={k: round(float(v), 4) for k, v in r.items()},
        config={"stage1": f"fast M2 fold {fold_a} (+ fold {fold_b} twin for its rows)", "stage2_train": list(fold_2),
                "params": {**M.PARAMS, **FAST}, "trees": m.num_trees(),
                "top_gain": (imp / imp.sum()).round(4).head(15).to_dict()},
        hypothesis="stage-1 probabilities of the competing S1 / SX (rank, margin, counts) sharpen the per-SX decision",
        verdict="keep" if r["f05_all"] > 0.9814 else "drop", runtime_s=round(time.time() - t0, 1), peak_gb=peak_gb())
    S.to_parquet(OUT / "S_stage2.parquet", index=False)
    return r


# ---------------------------------------------------------------- deliverables
def ids(split, c):
    """(S1 ids, SX ids) as arrow arrays in the q1 / qx row order."""
    import pyarrow as pa
    s1 = pq.read_table(TXT[split] / f"{c}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    files = sorted(TXT[split].glob(f"{c}_sx_*.parquet"))
    sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files])
    return s1, sx


def with_ids(d: pd.DataFrame, split, c, q1="q1", qx="qx") -> pd.DataFrame:
    import pyarrow as pa
    a, b = ids(split, c)
    return d.assign(s1=a.take(pa.array(d[q1].to_numpy())).to_numpy(zero_copy_only=False),   # arrays: d may be a
                    sx=b.take(pa.array(d[qx].to_numpy())).to_numpy(zero_copy_only=False))   # filtered view


def predict_split(scorer, split, c, extra=None) -> pd.DataFrame:
    """Every pair of one country: p, the SX's argmax flag over all its S1, the SX's best p.
    extra(path) -> DataFrame of more columns aligned with the X file (stage-2 features)."""
    out = []
    for p in sorted((OUT / "X").glob(f"{split}_{c}_sx_*.parquet")):
        X = pd.read_parquet(p)
        if extra is not None:
            X = pd.concat([X, extra(p)], axis=1)
        pr = scorer(X)
        qx = X.qx.to_numpy()
        d = pd.DataFrame({"q1": X.q1.to_numpy(), "qx": qx, "p": pr, "am": M.argmax_sx(X.q1.to_numpy(), qx, pr),
                          "sx_best_p": pd.Series(pr).groupby(qx).transform("max").to_numpy(np.float32),
                          "blocked": X.blocked.to_numpy() > 0})
        for k in ("is_true", "fold"):
            if k in X:
                d[k] = X[k].to_numpy()
        out.append(d)
        del X
    return pd.concat(out, ignore_index=True)


def final(stage="1", what="holdout,test"):
    """Deliverables with the kept model: prob_holdout / prob_dev (fold 0 / 1, all their pairs, argmax over ALL train
    S1) and prob_test_<country> (after featrun.py feat test <country>)."""
    import lightgbm as lgb
    what = what.split(",")
    m = lgb.Booster(model_file=str(OUT / ("m2_ladder.txt" if stage == "1" else "m2_stage2.txt")))
    extra = None if stage == "1" else (lambda p: pd.read_parquet(P1DIR / p.name))
    cols = ["s1", "sx", "p", "is_true", "country", "am", "sx_best_p", "blocked"]
    if "holdout" in what:
        parts = []
        for c in COUNTRIES:
            d = predict_split(lambda X: M.predict(m, X), "train", c, extra)
            d = d[np.isin(d.fold.to_numpy(), (0, 1))]
            parts.append(with_ids(d, "train", c).assign(country=c))
        d = pd.concat(parts, ignore_index=True)
        for f, name in ((0, "holdout"), (1, "dev")):
            d[d.fold == f][cols].to_parquet(OUT / f"prob_{name}.parquet", index=False)
        print("prob_holdout / prob_dev written", {f: int((d.fold == f).sum()) for f in (0, 1)}, flush=True)
    if "test" in what:
        ma = lgb.Booster(model_file=str(OUT / "m2fast_fold2.txt"))     # stage 2's own stage-1 model
        for c in ("France", "India", "US"):
            if stage != "1":
                p1_pass(ma, split="test", countries=[c])
            d = with_ids(predict_split(lambda X: M.predict(m, X), "test", c, extra), "test", c)
            d[["s1", "sx", "p", "am", "sx_best_p", "blocked"]].to_parquet(OUT / f"prob_test_{c}.parquet", index=False)
            print(f"prob_test_{c}: {len(d):,} pairs, {d.am.mean():.3f} argmax, peak {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    {"ladder": ladder, "stage2": stage2, "final": final}[sys.argv[1]](*sys.argv[2:])
