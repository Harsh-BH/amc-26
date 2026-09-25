"""P7 stage benchmarks. Each call measures ONE thing in a fresh process, so ru_maxrss is that stage's peak.

    .venv/bin/python probes/P7-scale/bench.py load <src> <engine>        # engine: pandas | pandas_pa | polars | object
    .venv/bin/python probes/P7-scale/bench.py tfidf <n_fit> <n_transform>
    .venv/bin/python probes/P7-scale/bench.py topn <n_q> <n_corpus> <k> [threads] [country]
    .venv/bin/python probes/P7-scale/bench.py feats <n_pairs>
    .venv/bin/python probes/P7-scale/bench.py lgbm <n_rows> <n_feat> <n_trees>
    .venv/bin/python probes/P7-scale/bench.py write <n_s1> <per_s1>
Prints one JSON line: {stage, ..., secs, peak_gb, base_gb}.
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/work"
sys.path[:0] = [str(ROOT / "probes/P3-block"), str(ROOT / "probes/P7-scale"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
from chunks import rss_gb, peak_gb


def out(**kw):
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(json.dumps({**kw, "peak_gb": round(peak_gb(), 3), "child_peak_gb": round(kids, 3),
                      "load": os.getloadavg()[0]}), flush=True)


def load(src, engine):
    p = WORK / f"test_{src}.parquet"
    if engine == "polars":
        import polars as pl
        base = rss_gb(); t = time.time()
        df = pl.read_parquet(p)
        secs = time.time() - t
        size = df.estimated_size() / 1e9
    else:
        import pandas as pd
        base = rss_gb(); t = time.time()
        df = pd.read_parquet(p, dtype_backend="pyarrow") if engine == "pandas_pa" else pd.read_parquet(p)
        if engine == "object":
            df = df.astype(object)
        secs = time.time() - t
        size = df.memory_usage(deep=True).sum() / 1e9
    out(stage="load", src=src, engine=engine, rows=len(df), secs=round(secs, 1), frame_gb=round(size, 3),
        base_gb=round(base, 3), rss_gb=round(rss_gb(), 3))


def tfidf(n_fit, n_tr, chunk="1000000"):
    """Fit P3's name/addr vectorizers on n_fit train S2/S3 rows, transform n_tr test S2 rows chunk by chunk."""
    import pandas as pd
    import scipy.sparse as sp
    import block as B
    n_fit, n_tr, chunk = int(n_fit), int(n_tr), int(chunk)
    cols = ["business_name", "business_address", "country"]
    fit_df = pd.concat([pd.read_parquet(WORK / f"train_source{i}.parquet", columns=cols).sample(n=n_fit // 2, random_state=i)
                        for i in (2, 3)], ignore_index=True)
    base = rss_gb(); t = time.time()
    vecs = B.fit_views(fit_df, B.CFG, sample=n_fit)
    fit_s, fit_peak = time.time() - t, peak_gb()
    del fit_df
    tr = pd.read_parquet(WORK / "test_source2.parquet", columns=cols).iloc[:n_tr]
    t = time.time(); clean_s = 0; mats = {"name": [], "addr": []}
    for o in range(0, len(tr), chunk):
        t1 = time.time(); tx = B.texts(tr.iloc[o:o + chunk], B.CFG); clean_s += time.time() - t1
        for v in mats:
            mats[v].append(vecs[v].transform(tx[v]).astype(np.float32).tocsr())
    X = {v: sp.vstack(m, format="csr") for v, m in mats.items()}
    tr_s = time.time() - t
    out(stage="tfidf", n_fit=n_fit, n_tr=n_tr, chunk=chunk, fit_s=round(fit_s, 1), fit_peak_gb=round(fit_peak, 3),
        vocab={v: len(getattr(x, "vocabulary_", ())) for v, x in vecs.items()}, transform_s=round(tr_s, 1), clean_s=round(clean_s, 1),
        nnz_per_row={v: round(x.nnz / x.shape[0], 1) for v, x in X.items()},
        csr_gb={v: round((x.data.nbytes + x.indices.nbytes + x.indptr.nbytes) / 1e9, 3) for v, x in X.items()},
        base_gb=round(base, 3))


def topn(n_q, n_c, k, threads="15", country="India", direction="f", max_df="1.0", view="na"):
    """One sp_matmul_topn call pattern at a given size, P3's fitted vectorizers, test rows of one country.
    direction f: n_q S1 queries against n_c SX rows; r: n_q SX queries against n_c S1 rows (the S1 pool)."""
    import pickle
    import pandas as pd
    import block as B
    from sparse_dot_topn import sp_matmul_topn
    n_q, n_c, k, threads = int(n_q), int(n_c), int(k), int(threads)
    cfg = B.CFG
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    cols = ["business_name", "business_address", "country"]
    rd = lambda s, n: pd.read_parquet(WORK / f"test_{s}.parquet", columns=cols, filters=[("country", "==", country)]).iloc[:n]
    s1, sx = (rd("source1", n_q), rd("source2", n_c)) if direction == "f" else (rd("source2", n_q), rd("source1", n_c))
    if float(max_df) < 1:
        _prune_mats(B, float(max_df))              # keep mask from the query side's df
    Q = B.mats(B.texts(s1, cfg), {view}, vecs, cfg)[view]
    C = B.mats(B.texts(sx, cfg), {view}, vecs, cfg)[view]
    CT = C.T.tocsr()
    base = rss_gb(); t = time.time()
    R = sp_matmul_topn(Q, CT, top_n=k, n_threads=threads)
    secs = time.time() - t
    out(stage="topn", view=view, direction=direction, country=country, n_q=Q.shape[0], n_c=C.shape[0], k=k,
        threads=threads, max_df=float(max_df), secs=round(secs, 2),
        ns_per_cell=round(secs * 1e9 * threads / (Q.shape[0] * C.shape[0]), 4),
        q_per_s=round(Q.shape[0] / secs), nnz_q=round(Q.nnz / Q.shape[0], 1), nnz_c=round(C.nnz / C.shape[0], 1),
        out_nnz=R.nnz, flops=_flops(Q, C), ns_per_flop=round(secs * 1e9 * threads / max(_flops(Q, C), 1), 3),
        base_gb=round(base, 3))


def _df(X, ncol=None) -> np.ndarray:
    """Per-feature document frequency of a CSR matrix."""
    return np.bincount(X.indices, minlength=ncol or X.shape[1]).astype(np.float64)


def _flops(Q, C) -> int:
    """Multiply-adds of Q @ C.T in a row-by-row sparse product (Gustavson): sum over features of dfQ * dfC."""
    return int(_df(Q) @ _df(C, Q.shape[1]))


def flops(country, max_dfs="1.0,0.1,0.05,0.02,0.01", chunk="500000"):
    """Exact multiply-add count of P3's blocker (B.CFG: na view, state partition) for one test country, forward and
    reverse, when features with within-country df > max_df are dropped. Only per-group column counts are kept.
    (Ledger rows before 05:10 used P3's vocabulary vectorizer and train-df max_df; this is the current version.)"""
    import pickle
    import pandas as pd
    import block as B
    cfg = B.CFG
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    cols = ["business_name", "business_address", "country"]
    t0 = time.time()
    df = {}                                   # (side, state group) -> per-feature df
    n = {}
    for side, srcs in (("a", ["source1"]), ("b", ["source2", "source3"])):
        for s in srcs:
            d = pd.read_parquet(WORK / f"test_{s}.parquet", columns=cols, filters=[("country", "==", country)])
            for o in range(0, len(d), int(chunk)):
                t = B.texts(d.iloc[o:o + int(chunk)], cfg)
                X = B.mats(t, {"na"}, vecs, cfg)["na"]
                for g, rows in B._groups(t.st).items():
                    key = (side, g)
                    df[key] = df.get(key, 0) + _df(X[rows])
                    n[key] = n.get(key, 0) + len(rows)
            del d
    n_a = sum(v for (s, g), v in n.items() if s == "a"); n_b = sum(v for (s, g), v in n.items() if s == "b")
    frac = np.maximum(sum(v for (s, g), v in df.items() if s == "a") / n_a,
                      sum(v for (s, g), v in df.items() if s == "b") / n_b)
    res = {}
    for mx in map(float, max_dfs.split(",")):
        keep = (frac <= mx).astype(np.float64)
        A = {g: v * keep for (s, g), v in df.items() if s == "a"}
        Bx = {g: v * keep for (s, g), v in df.items() if s == "b"}
        z = np.zeros(len(keep))
        a_all, b_all = sum(A.values()), sum(Bx.values())
        a_loose, b_nost = A.get("", z), Bx.get("", z)
        if cfg.get("partition"):
            fwd = sum(A[g] @ (Bx.get(g, z) + b_nost) for g in A if g) + a_loose @ b_all
            rev = sum(Bx[g] @ ((A[g] + a_loose) if g in A else a_all) for g in Bx if g) + b_nost @ a_all
        else:
            fwd = rev = a_all @ b_all
        res[mx] = {"fwd": float(fwd), "rev": float(rev), "nopart": float(a_all @ b_all)}
    out(stage="flops", country=country, n_a=sum(v for (s, g), v in n.items() if s == "a"),
        n_b=sum(v for (s, g), v in n.items() if s == "b"), groups=len({g for s, g in df}),
        loose_a=n.get(("a", ""), 0), nost_b=n.get(("b", ""), 0), partition=cfg.get("partition"),
        translit=cfg.get("translit"), flops=res, secs=round(time.time() - t0, 1))


def feats(n_pairs, chunk="1000000", workers="15", country="India", per_s1="30"):
    """Pair-feature throughput: a Foursquare-style feature set (rapidfuzz cpdist + TF-IDF row cosines) over
    n_pairs synthetic candidate pairs (per_s1 SX per S1, same country), streamed in `chunk`-pair pieces."""
    import pickle
    import pandas as pd
    import block as B
    from rapidfuzz import distance, fuzz, process
    n_pairs, chunk, workers, per_s1 = int(n_pairs), int(chunk), int(workers), int(per_s1)
    cfg = {**B.CFG, "translit": "anyascii"}
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    cols = ["business_name", "business_address", "country"]
    rd = lambda s: pd.read_parquet(WORK / f"test_{s}.parquet", columns=cols, filters=[("country", "==", country)])
    a, b = rd("source1"), rd("source2").iloc[:2_000_000]
    rng = np.random.default_rng(0)
    qi = np.repeat(rng.integers(0, len(a), n_pairs // per_s1), per_s1)
    ci = rng.integers(0, len(b), len(qi))
    ua, ub = np.unique(qi), np.unique(ci)             # only records that appear in a pair get cleaned
    ta, tb = B.texts(a.iloc[ua], cfg), B.texts(b.iloc[ub], cfg)
    qi, ci = np.searchsorted(ua, qi), np.searchsorted(ub, ci)
    Xa = {v: vecs[v].transform(ta[v]).astype(np.float32).tocsr() for v in ("name", "addr")}
    Xb = {v: vecs[v].transform(tb[v]).astype(np.float32).tocsr() for v in ("name", "addr")}
    na_, nb_ = ta.name.to_numpy(object), tb.name.to_numpy(object)
    aa_, ab_ = ta.addr.to_numpy(object), tb.addr.to_numpy(object)
    del a, b, ta, tb
    SC = {"name": [fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio,
                   distance.JaroWinkler.normalized_similarity, distance.Levenshtein.normalized_distance],
          "addr": [fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio]}
    base = rss_gb(); t0 = time.time(); per = {}
    got = []
    for o in range(0, len(qi), chunk):
        i, j = qi[o:o + chunk], ci[o:o + chunk]
        cols_ = []
        for side, (A_, B_) in (("name", (na_, nb_)), ("addr", (aa_, ab_))):
            x, y = A_[i].tolist(), B_[j].tolist()
            for sc in SC[side]:
                t = time.time()
                cols_.append(process.cpdist(x, y, scorer=sc, workers=workers, dtype=np.float32))
                per[f"{side}.{sc.__name__}"] = per.get(f"{side}.{sc.__name__}", 0) + time.time() - t
            t = time.time()
            cols_.append(np.asarray(Xa[side][i].multiply(Xb[side][j]).sum(1)).ravel().astype(np.float32))
            per[f"{side}.tfidf_cos"] = per.get(f"{side}.tfidf_cos", 0) + time.time() - t
            cols_.append(np.array([len(s) for s in x], np.float32) - np.array([len(s) for s in y], np.float32))
        got.append(np.column_stack(cols_))
    secs = time.time() - t0
    F = np.vstack(got)
    out(stage="feats", n_pairs=len(qi), n_feat=F.shape[1], chunk=chunk, workers=workers, secs=round(secs, 1),
        pairs_per_s=round(len(qi) / secs), per_feature_s={k: round(v, 2) for k, v in per.items()},
        feat_gb=round(F.nbytes / 1e9, 3), base_gb=round(base, 3))


def lgbm(n_rows, n_feat="30", n_trees="500", threads="15", n_train="1000000", leaves="63"):
    """LightGBM train on n_train synthetic rows, then predict n_rows in 1M-row chunks (the streaming pattern)."""
    import lightgbm as lgb
    n_rows, n_feat, n_trees, threads, n_train, leaves = map(int, (n_rows, n_feat, n_trees, threads, n_train, leaves))
    rng = np.random.default_rng(0)
    X = rng.random((n_train, n_feat), dtype=np.float32)
    y = ((X[:, 0] * X[:, 1] + np.sin(6 * X[:, 2]) * X[:, 3] + 0.3 * rng.standard_normal(n_train)) > 0.4).astype(int)
    t = time.time()
    m = lgb.train({"objective": "binary", "num_leaves": leaves, "learning_rate": 0.05, "num_threads": threads,
                   "verbose": -1, "max_bin": 255}, lgb.Dataset(X, y), num_boost_round=n_trees)
    train_s = time.time() - t
    del X, y
    base = rss_gb(); t = time.time(); done = 0
    while done < n_rows:
        c = min(1_000_000, n_rows - done)
        m.predict(rng.random((c, n_feat), dtype=np.float32), num_threads=threads)
        done += c
    secs = time.time() - t
    out(stage="lgbm", n_train=n_train, n_feat=n_feat, n_trees=n_trees, leaves=leaves, threads=threads,
        train_s=round(train_s, 1), n_rows=n_rows, predict_s=round(secs, 1), rows_per_s=round(n_rows / secs),
        base_gb=round(base, 3))


def write(n_s1="1732544", per_s1="30", engine="polars"):
    """Write candidate_pairs.tsv-shaped output (n_s1 rows, per_s1 ids each) from int row indices."""
    import pandas as pd
    import polars as pl
    n_s1, per_s1 = int(n_s1), int(per_s1)
    rng = np.random.default_rng(0)
    s1_ids = pd.read_parquet(WORK / "test_source1.parquet", columns=["entity_id"]).entity_id.to_numpy()[:n_s1]
    sx_ids = pl.read_parquet(WORK / "test_source2.parquet", columns=["entity_id"])["entity_id"]
    qi = np.repeat(np.arange(n_s1, dtype=np.int32), rng.integers(0, 2 * per_s1, n_s1))
    ci = rng.integers(0, len(sx_ids), len(qi)).astype(np.int32)
    path = Path("/tmp/p7_write.tsv")
    base = rss_gb(); t = time.time()
    if engine == "chunks":
        from chunks import write_lists
        write_lists(path, "candidate_entity_ids", s1_ids, qi, sx_ids.to_numpy(), ci)
    elif engine == "polars":
        p = pl.DataFrame({"q": qi, "sx": sx_ids.gather(ci)}).group_by("q").agg(pl.col("sx").str.join(","))
        full = pl.DataFrame({"q": np.arange(n_s1, dtype=np.int32), "source1_entity_id": s1_ids})
        full = full.join(p, on="q", how="left").sort("q").select(
            "source1_entity_id", pl.col("sx").fill_null("").alias("candidate_entity_ids"))
        full.write_csv(path, separator="\t", quote_style="never")
    else:
        sx = pd.Series(sx_ids.to_numpy()[ci]).groupby(qi).agg(",".join)
        pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_ids": sx.reindex(range(n_s1), fill_value="").to_numpy()}
                     ).to_csv(path, sep="\t", index=False)
    secs = time.time() - t
    out(stage="write", engine=engine, n_s1=n_s1, n_pairs=len(qi), secs=round(secs, 1),
        file_mb=round(path.stat().st_size / 1e6), base_gb=round(base, 3))
    path.unlink()


def _prune_mats(B, max_df: float):
    """Monkeypatch B.mats to drop features whose document frequency in the FIRST matrix it builds (the S1 pool of
    the current country, built by B.index) exceeds max_df. Norms stay those of the full vector (partial cosine).
    Call once per country (the keep mask is per country)."""
    orig, keep = B._orig_mats if hasattr(B, "_orig_mats") else B.mats, {}
    B._orig_mats = orig

    def mats(t, views, vecs, cfg):
        out_ = orig(t, views, vecs, cfg)
        for v, X in out_.items():
            if v not in keep:
                keep[v] = (np.bincount(X.indices, minlength=X.shape[1]) <= max_df * X.shape[0]).astype(np.float32)
            X = X.multiply(keep[v][None, :]).tocsr()
            X.eliminate_zeros()
            out_[v] = X.astype(np.float32)
        return out_
    B.mats = mats


def _pieces_topk(B, size: int):
    """Monkeypatch B.topk: a corpus piece with more than `size` rows is split into <= size-row pieces (built once per
    piece and cached), so the per-thread accumulator stays in cache. Exact up to ties (bench.py pieces)."""
    orig, cache = B._orig_topk if hasattr(B, "_orig_topk") else B.topk, {}
    B._orig_topk = orig

    def topk(Q, CTs, k, cfg):
        out_ = []
        for cols, CT in CTs:
            n = CT.shape[1]
            if n <= size:
                out_.append((cols, CT)); continue
            if id(CT) not in cache:
                CC = CT.tocsc()
                cache[id(CT)] = (CT, [(cols + o if np.isscalar(cols) else np.asarray(cols)[o:o + size],
                                       CC[:, o:o + size].tocsr()) for o in range(0, n, size)])
            out_ += cache[id(CT)][1]
        return orig(Q, out_, k, cfg)
    B.topk = topk


def recall(max_df="1.0", world="dev", vecs_file="vecs_hash.pkl"):
    """Pair recall + candidates/S1 of P3's candidates() on a world, with the within-country df prune."""
    import pickle
    import block as B
    from worlds import load_world
    s1, sx, truth = load_world(world)
    vecs = pickle.loads((WORK / "p3" / vecs_file).read_bytes())
    t = time.time(); parts = []
    for c in sorted(set(s1.country)):
        if float(max_df) < 1:
            _prune_mats(B, float(max_df))
        parts.append(B.candidates(s1[s1.country == c], sx[sx.country == c], B.CFG, vecs, log=lambda *_: None))
    secs = time.time() - t
    import pandas as pd
    cand = pd.concat(parts)
    got = set(zip(cand.s1, cand.sx))
    tp = [(a, b) for a, bs in truth.items() for b in bs]
    hit = np.array([p in got for p in tp])
    s1c = np.array([s for s, _ in tp])
    full = pd.Series(hit).groupby(s1c).all().mean()
    out(stage="recall", world=world, max_df=float(max_df), pair_recall=round(float(hit.mean()), 4),
        s1_full_recall=round(float(full), 4), cands_per_s1=round(len(cand) / len(s1), 2), secs=round(secs, 1))


def pieces(n_q="20000", country="India", sizes="0,250000,125000"):
    """P3's B.topk for n_q SX queries against the WHOLE S1 pool of a country (what a no-state SX does), with the
    pool passed as one piece (0) or split into pieces of <= size rows. Results must be identical; time must drop."""
    import pickle
    import pandas as pd
    import block as B
    cfg = B.CFG
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    cols = ["business_name", "business_address", "country"]
    rd = lambda s: pd.read_parquet(WORK / f"test_{s}.parquet", columns=cols, filters=[("country", "==", country)])
    A = B.mats(B.texts(rd("source1"), cfg), {"na"}, vecs, cfg)["na"]
    Q = B.mats(B.texts(rd("source2").sample(n=int(n_q), random_state=0), cfg), {"na"}, vecs, cfg)["na"]
    res, ref = {}, None
    for size in map(int, sizes.split(",")):
        size = size or A.shape[0]
        CTs = [(o, A[o:o + size].T.tocsr()) for o in range(0, A.shape[0], size)]
        t = time.time()
        qi, ci, sc = B.topk(Q, CTs, 10, cfg)
        res[size] = round(time.time() - t, 2)
        got = set(zip(qi.tolist(), ci.tolist()))
        scores = pd.Series(np.round(sc, 5)).groupby(qi).agg(tuple)          # per-query sorted score list
        if ref is None:
            ref, ref_sc = got, scores
        res[f"{size}_same_pairs"] = round(len(got & ref) / max(len(ref), 1), 4)   # tied members may swap
        res[f"{size}_same_scores"] = round(float((scores == ref_sc.reindex(scores.index)).mean()), 4)
    out(stage="pieces", country=country, n_q=int(n_q), pool=A.shape[0], secs=res, threads=cfg["threads"])


def block(country, shard="500000", workers="1", vecs_file="vecs_hash.pkl", max_df="1.0", max_sx="0", piece="0"):
    """P3's current blocker (B.CFG: reverse na top-10, state partition, derive keep_r/keep_f) on ONE full test
    country, timing texts / index / search separately. workers > 1 builds SX shard texts in a pmap fork pool."""
    import pickle
    import pandas as pd
    import block as B
    from chunks import pmap
    cfg = {**B.CFG, "shard": int(shard)}
    workers = int(workers)
    vecs = pickle.loads((WORK / "p3" / vecs_file).read_bytes())
    if float(max_df) < 1:
        _prune_mats(B, float(max_df))
    if int(piece):
        _pieces_topk(B, int(piece))
    cols = ["entity_id", "business_name", "business_address", "country"]
    rd = lambda s: pd.read_parquet(WORK / f"test_{s}.parquet", columns=cols, filters=[("country", "==", country)])
    a = rd("source1"); b = pd.concat([rd("source2"), rd("source3")], ignore_index=True)
    if int(max_sx):                         # a random SX subset: reverse-search time is linear in #SX
        b = b.sample(n=min(int(max_sx), len(b)), random_state=0).reset_index(drop=True)
    load_gb = rss_gb()
    t = time.time(); ta = B.texts(a, cfg); t_s1 = time.time() - t
    t = time.time(); ix = B.index(ta, vecs, cfg); t_ix = time.time() - t
    t_sx = 0.0

    def shard_texts():                     # `workers` shards of texts resident at a time, not the whole country
        nonlocal t_sx
        step = cfg["shard"] * max(workers, 1)
        for g0 in range(0, len(b), step):
            t = time.time()
            sub = b.iloc[g0:g0 + step]
            got = pmap(lambda sl: B.texts(sub.iloc[sl], cfg), len(sub), cfg["shard"], workers)
            t_sx += time.time() - t
            yield from got
    ks = {v: cfg["keep_f"] for v in ix["rk"]} | ix["fk"]
    keep, pool, t_search, t_fold = [], None, 0.0, 0.0
    for n, tb in enumerate(shard_texts()):
        t = time.time()
        h = B.search(ix, tb, vecs, cfg)
        t_search += time.time() - t; t = time.time()
        h["qx"] += n * cfg["shard"]
        r = h[h["dir"] == "r"]
        keep.append(r[r["rank"] < cfg["keep_r"]].assign(via="r"))
        pool = B.fold(pd.concat([pool, h]) if pool is not None else h, ks)
        t_fold += time.time() - t
    h = pd.concat(keep + [pool.assign(via="f")], ignore_index=True)
    h = h.sort_values("score", ascending=False).drop_duplicates(["q1", "qx"])
    n_pairs = len(h)
    (WORK / "p7").mkdir(exist_ok=True)
    pd.DataFrame({"q1": h.q1.to_numpy(np.int32), "qx": h.qx.to_numpy(np.int32), "score": h.score.to_numpy(np.float32)}
                 ).to_parquet(WORK / f"p7/cands_{country}_{max_df}_{max_sx}.parquet", index=False)
    out(stage="block", country=country, n_s1=len(a), n_sx=len(b), shard=cfg["shard"], workers=workers, max_df=float(max_df), piece=int(piece),
        rev=cfg["rev"], keep_r=cfg["keep_r"], keep_f=cfg["keep_f"], groups=len(ix["ga"]),
        texts_s1_s=round(t_s1, 1), index_s=round(t_ix, 1), texts_sx_s=round(t_sx, 1), search_s=round(t_search, 1),
        fold_s=round(t_fold, 1), pairs=n_pairs, pairs_per_s1=round(n_pairs / len(a), 2), load_gb=round(load_gb, 3))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
