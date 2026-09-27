"""P80 stage C (rival-style rebuild): a LightGBM over ALL pruned candidates of every train fold, on honest out-of-fold M2 p,
that re-decides each SX's owner. Stage 1 (solution_p80.py) cross-fits M2 on folds {2,3}<->{4,5}; folds 0, 1, 6-9 were
seen by neither booster, so they take the logit-space mean of both (= score_test.py on test). Every p is out of sample for its own pair.

frame C -> <out>/sc_<C>.parquet, one row per pruned candidate (M2 p >= PMIN) of every fold:
  q1 qx fold y s3 p + RAW (four of M2's pair features) + anchors a_n / a_name / a_addr
  (anchors = the S1's OTHER candidates with p >= .9; best token-set ratio of this SX's name / address to theirs).
fit     -> derived competition features (feats), stage C cross-fit the same way (C_A fit {2,3}, C_B fit {4,5}, the rest
  = mean of both, early stop on fold 1), owner re-decision (each SX -> its best-scoring S1, kept at s >= t, caps 5 S2 /
  6 S3), t tuned on fold 1, per-S1 F0.5 on fold 0 vs today's decision (P76 frame: sub54's p54 >= .71, out of fold).
    .venv/bin/python probes/P80-crossfit/stagec.py frame India      # needs out/<stage-1 job>/ (sm_launch.py --fetch)
    .venv/bin/python probes/P80-crossfit/stagec.py fit              # -> probes/P80-crossfit/stagec.json
    .venv/bin/python probes/P80-crossfit/stagec.py test India --test-p <score_test.py out dir>  # -> sc_test_India.parquet
    smoke: add --smoke (1% S1 sample, local smoke boosters out/p80m + OOF out/p80o, --out a scratch dir)
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = ROOT / "data/work"
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P76-segment-hill"), str(ROOT)]
import features as FT  # noqa: E402
from dive import fb  # noqa: E402  (per-S1 F0.5, singletons count, empty = 1.0)
from solution_p80 import keep  # noqa: E402  (stage 1's q1-hash sample)

PMIN, ANC, T_BASE = 0.01, 0.9, 0.71
CAP2, CAP3 = 5, 6
RAW = ["name_tset", "core_tset", "addr_tset", "hn_eq"]
JOBS = {"A": ("p80-cfA-0927-2115", "p80-cfA-r-0927-2123"), "B": ("p80-cfB-0927-2115", "p80-cfB-r-0927-2123")}
LB_W = {"India": 0.468, "US": 0.383}
C = ("India", "US")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, deterministic=True, seed=0, verbose=-1, num_threads=8)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
logit = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
keys = lambda q1, qx: (np.asarray(q1, np.int64) << 32) | np.asarray(qx, np.int64)
ALL = tuple(range(10))


def job_dirs(smoke):
    """{split: (dir holding m2_cf_<split>.txt, dir holding its oof_*.parquet)}; a fetched job holds both."""
    if smoke:
        return {s: (ROOT / "out/p80m", ROOT / "out/p80o") for s in JOBS}
    got = {}
    for s, names in JOBS.items():
        d = [(m.parent, ROOT / "out" / j) for j in names for m in sorted((ROOT / "out" / j).rglob(f"m2_cf_{s}.txt"))]
        if not d:
            sys.exit(f"no fetched stage-1 output for split {s}: sm_launch.py --profile harsh --fetch {names}")
        got[s] = d[0]
    return got


# ------------------------------------------------------------------ frame
def pseudo_fold(q1):
    """--ho rehearsal: fold-0 S1 split into 6 pseudo folds that play the roles of folds 0-5."""
    return ((np.asarray(q1, np.uint64) * np.uint64(2654435761) + np.uint64(12345)) % np.uint64(2**32) % np.uint64(6)).astype(np.int8)


def ho_p(c):
    """m2_big's fold-0 p (P4's holdout world: fold 0 was never trained on), keyed by (q1, qx)."""
    id1, idx = FT.ids("train", c)
    t = pq.read_table(W / "p4/prob_holdout_big.parquet", columns=["s1", "sx", "p"], filters=[("country", "=", c)])
    q1 = pc.index_in(pc.cast(t.column("s1"), pa.large_string()), value_set=id1)
    qx = pc.index_in(pc.cast(t.column("sx"), pa.large_string()), value_set=idx)
    assert q1.null_count == 0 and qx.null_count == 0
    return pd.Series(t.column("p").to_numpy().astype(np.float64), index=keys(q1.to_numpy(), qx.to_numpy()))


def m2_scores(c, frac, ho=False):
    """Pruned (q1, qx, fold, y, s3, p, RAW) over every X file of c; p = OOF on folds 2-5, mean of both boosters elsewhere.
    ho: fold-0 rows only, p = m2_big's holdout p, fold = pseudo_fold (a rehearsal before stage 1 lands)."""
    import lightgbm as lgb
    if ho:
        cols, po = RAW + ["is_s3"], ho_p(c)
    else:
        dirs = job_dirs(frac < 1)
        bA, bB = (lgb.Booster(model_file=str(dirs[s][0] / f"m2_cf_{s}.txt")) for s in ("A", "B"))
        cols = bA.feature_name()
        assert cols == bB.feature_name() and set(RAW + ["is_s3"]) <= set(cols)
    parts, n_all, n_true_all, n_true_kept = [], 0, 0, 0
    t0 = time.time()
    for xf in sorted((W / "p4/X").glob(f"train_{c}_sx_*.parquet")):
        if not ho:
            oofs = sorted({p for _, d in dirs.values() for p in d.rglob(f"oof_*{xf.name}")})
            o = pd.concat([pd.read_parquet(p, columns=["q1", "qx", "p"]) for p in oofs])
            po = pd.Series(o.p.to_numpy(np.float64), index=keys(o.q1, o.qx))
            del o
        assert po.index.is_unique, xf.name
        for b in pq.ParquetFile(xf).iter_batches(300_000, columns=["q1", "qx", "fold", "is_true"] + cols):
            d = b.to_pandas()
            d = d[keep(d.q1.to_numpy(), d.fold.to_numpy(), (0,) if ho else ALL, frac)]
            f = pseudo_fold(d.q1.to_numpy()) if ho else d.fold.to_numpy()
            p = po.reindex(keys(d.q1, d.qx)).to_numpy(np.float64, copy=True)
            both = np.zeros(len(d), bool) if ho else ~np.isin(f, (2, 3, 4, 5))
            assert not np.isnan(p[~both]).any(), f"{xf.name}: fold 2-5 rows without an OOF p"
            if both.any():
                Xb = d.loc[both, cols].to_numpy(np.float32)
                # logit-space mean = score_test.py's test-side p, so fold 0 sees the test distribution
                p[both] = 1 / (1 + np.exp(-(logit(bA.predict(Xb, num_threads=8)) + logit(bB.predict(Xb, num_threads=8))) / 2))
            k = p >= PMIN
            y = d.is_true.to_numpy(bool)
            n_all, n_true_all, n_true_kept = n_all + len(d), n_true_all + y.sum(), n_true_kept + y[k].sum()
            parts.append(pd.DataFrame({"q1": d.q1.to_numpy(np.int32)[k], "qx": d.qx.to_numpy(np.int32)[k],
                                       "fold": f.astype(np.int8)[k], "y": y[k], "s3": d.is_s3.to_numpy()[k] > 0.5,
                                       "p": p[k].astype(np.float32),
                                       **{r: d[r].to_numpy(np.float32)[k] for r in RAW}}))
            del d, b
        print(f"  {c} {xf.name}: {sum(map(len, parts)):,} kept of {n_all:,} | {time.time() - t0:.0f}s peak {peak():.2f} GB",
              flush=True)
    D = pd.concat(parts, ignore_index=True)
    print(f"{c}: {len(D):,} pruned rows of {n_all:,} (p >= {PMIN}); true kept {n_true_kept / n_true_all:.5f}", flush=True)
    return D, dict(rows_all=int(n_all), rows=len(D), true_kept=float(n_true_kept / n_true_all))


def anchor_pairs(q1, anc):
    """(i, j): every row i with every anchor row j != i of the same S1. i comes out sorted."""
    a = np.flatnonzero(anc)
    a = a[np.argsort(q1[a], kind="stable")]
    aq = q1[a]
    lo, hi = np.searchsorted(aq, q1, "left"), np.searchsorted(aq, q1, "right")
    cnt = hi - lo
    i = np.repeat(np.arange(len(q1), dtype=np.int32), cnt)
    j = a[np.repeat(lo, cnt) + np.arange(len(i)) - np.repeat(np.cumsum(cnt) - cnt, cnt)].astype(np.int32)
    m = i != j
    return i[m], j[m]


def anchors(c, D, chunk=1_000_000, split="train"):
    """a_n = the S1's other p >= ANC candidates; a_name / a_addr = max token-set ratio (P3 normalised text) to them."""
    from rapidfuzz import fuzz, process
    fs, _ = FT.sx_files(split, c)
    t = pa.concat_tables([pq.read_table(f, columns=["name", "addr"]) for f in fs])
    name, addr = t.column("name"), t.column("addr")
    del t
    q1, qx, anc = D.q1.to_numpy(), D.qx.to_numpy(), D.p.to_numpy() >= ANC
    D["a_n"] = (np.bincount(q1, weights=anc)[q1] - anc).astype(np.float32)
    i, j = anchor_pairs(q1, anc)
    assert len(i) == D.a_n.sum()
    out = {"a_name": (name, np.full(len(D), np.nan, np.float32)), "a_addr": (addr, np.full(len(D), np.nan, np.float32))}
    for s in range(0, len(i), chunk):
        ii, jj = i[s:s + chunk], j[s:s + chunk]
        for col, (txt, acc) in out.items():
            r = process.cpdist(txt.take(qx[ii]).to_pylist(), txt.take(qx[jj]).to_pylist(), scorer=fuzz.token_set_ratio,
                               workers=8, dtype=np.float32)
            np.fmax.at(acc, ii, r)
    empty = pc.equal(pc.utf8_length(pc.utf8_trim_whitespace(addr.take(qx))), 0).to_numpy(zero_copy_only=False)
    out["a_addr"][1][empty] = np.nan
    for col, (_, acc) in out.items():
        D[col] = acc
    print(f"{c}: anchors on {len(i):,} pairs, peak {peak():.2f} GB", flush=True)


def frame(c, frac, out, ho=False):
    t0 = time.time()
    D, info = m2_scores(c, frac, ho)
    anchors(c, D)
    tag = "sc_ho" if ho else "sc"
    D.to_parquet(out / f"{tag}_{c}.parquet", index=False)
    (out / f"{tag}_{c}.json").write_text(json.dumps(dict(info, sec=round(time.time() - t0)), indent=1))
    print(f"{c} frame done {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


# ------------------------------------------------------------------ stage C
def ranks(key, p):
    """Within key groups ordered by p desc: rank (0 = best), size, p just above / below, the best OTHER row (-1: none)."""
    o = np.lexsort((-p, key))
    ks, ps = key[o], p[o].astype(np.float64)
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    size = np.diff(np.r_[start, len(o)])[gid]
    rk = np.arange(len(o)) - start[gid]
    up = np.r_[np.nan, ps[:-1]]
    up[first] = np.nan
    dn = np.r_[ps[1:], np.nan]
    dn[np.r_[first[1:], True]] = np.nan
    oth = np.where(size > 1, o[np.where(rk == 0, np.minimum(start[gid] + 1, len(o) - 1), start[gid])], -1)
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return {k: v[inv] for k, v in dict(rank=rk, size=size, up=up, dn=dn, oth=oth).items()}


def feats(D):
    """Second-order competition + anchors + rival features, one row per pruned candidate (built column by column in
    float32: a 6M-row US frame stays well under 2 GB)."""
    q1, qx, s3 = D.q1.to_numpy(np.int64), D.qx.to_numpy(np.int64), D.s3.to_numpy()
    p = D.p.to_numpy(np.float64)
    at = lambda v, ix: np.where(ix >= 0, np.asarray(v, np.float64)[np.maximum(ix, 0)], np.nan)
    F = {}
    put = lambda k, v: F.__setitem__(k, np.asarray(v, np.float32))
    put("lp", logit(p))
    put("s3", s3)
    for r in RAW:
        put(r, D[r])
    R = ranks(q1, p)                                      # within the S1
    put("r1", R["rank"]), put("n1", R["size"]), put("up1", R["up"] - p), put("dn1", p - R["dn"])
    put("top1_oth", np.nan_to_num(at(p, R["oth"]), nan=0.0))
    r1, n1 = R["rank"], R["size"]
    R = ranks(q1 * 2 + s3, p)                             # within the S1 and source
    put("r1s", R["rank"]), put("n1s", R["size"]), put("n1o", n1 - R["size"])
    put("ps1", np.bincount(q1, weights=p)[q1] - p)
    R = ranks(qx, p)                                      # within the SX
    rv = R["oth"]                                         # the rival pair: best other S1 of this SX
    put("rx", R["rank"]), put("nx", R["size"])
    del R
    bx = np.nan_to_num(at(p, rv), nan=0.0)
    ge5 = p >= 0.5
    put("bx", bx), put("mx", logit(p) - logit(np.maximum(bx, 1e-6)))
    put("nx05", np.bincount(qx, weights=ge5)[qx] - ge5), put("psx", np.bincount(qx, weights=p)[qx] - p)
    put("rv_r1", at(r1, rv)), put("rv_n1", at(n1, rv)), put("rv_an", at(D.a_n, rv))
    put("rv_a_name", at(D.a_name, rv)), put("rv_a_addr", at(D.a_addr, rv))
    for r in RAW:
        put(f"d_{r}", D[r].to_numpy(np.float64) - at(D[r], rv))
    for r in ("a_n", "a_name", "a_addr"):
        put(r, D[r])
    put("d_a_name", F["a_name"] - F["rv_a_name"])
    return pd.DataFrame(F)


def owner_rank(q1, qx, s3, s):
    """own = the SX's best-scoring S1; rk = rank of an owned pair among its S1's owned same-source pairs (s desc).
    acc(t) = own & (s >= t) & (rk < cap): caps take the best first, so rk does not depend on t."""
    o = np.lexsort((-s, qx))
    own = np.zeros(len(s), bool)
    own[o[np.r_[True, qx[o][1:] != qx[o][:-1]]]] = True
    i = np.flatnonzero(own)
    rk = np.full(len(s), 1 << 30)
    rk[i] = ranks(q1[i].astype(np.int64) * 2 + s3[i], s[i])["rank"]
    return own & (rk < np.where(s3, CAP3, CAP2))


def s1_pool(c):
    """id1 (q1 order), n_true and fold per q1 over the whole train S1 pool of c."""
    id1, _ = FT.ids("train", c)
    q = pc.drop_null(pc.index_in(pq.read_table(W / "gt_pairs.parquet", columns=["s1"]).column(0), value_set=id1))
    n_true = np.bincount(q.to_numpy(), minlength=len(id1))
    fo = pq.read_table(W / "s1_folds.parquet", columns=["entity_id", "fold"])
    ix = pc.index_in(id1, value_set=fo.column("entity_id").combine_chunks())
    assert ix.null_count == 0
    return id1, n_true, fo.column("fold").to_numpy()[ix.to_numpy()]


def per_s1(q1, y, acc, sel, n_true):
    N = len(n_true)
    return fb(np.bincount(q1[acc & y], minlength=N)[sel], np.bincount(q1[acc], minlength=N)[sel], n_true[sel])


def tune(q1, y, cand, s, sel, n_true, grid=np.round(np.arange(0.30, 0.961, 0.01), 2)):
    f = [per_s1(q1, y, cand & (s >= t), sel, n_true).mean() for t in grid]
    return float(grid[int(np.argmax(f))]), float(max(f))


def baseline(c, id1):
    """Today's decision per fold-0 S1 (P76 frame, sub54 p54 >= .71 out of fold) and those S1's q1."""
    F = pd.read_parquet(W / f"p76/F_{c}.parquet", columns=["g", "y", "p54"])
    S = pd.read_parquet(W / f"p76/S_{c}.parquet", columns=["id", "n_true", "half"])
    g, y, acc = F.g.to_numpy(), F.y.to_numpy(), F.p54.to_numpy() >= T_BASE
    n = len(S)
    f = fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), S.n_true.to_numpy())
    q = pc.index_in(pa.array(S.id.to_numpy(object), pa.large_string()), value_set=id1)
    assert q.null_count == 0
    return f, q.to_numpy(), S.n_true.to_numpy(), S.half.to_numpy()


def cmp(a, b):
    d = a - b
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), up=int((d > 0).sum()), down=int((d < 0).sum()))


def fit_c(X, y, f):
    import lightgbm as lgb
    s, bs = np.full(len(y), np.nan), {}
    va = f == 1
    dv = lambda ref: lgb.Dataset(X[va], y[va].astype(np.float32), reference=ref)
    for name, fit_f, score_f in (("A", (2, 3), (4, 5)), ("B", (4, 5), (2, 3))):
        tr = np.isin(f, fit_f)
        d = lgb.Dataset(X[tr], y[tr].astype(np.float32), free_raw_data=True)
        b = lgb.train(PARAMS, d, 3000, valid_sets=[dv(d)], callbacks=[lgb.early_stopping(50, verbose=False)])
        m = np.isin(f, score_f)
        s[m] = b.predict(X[m], num_iteration=b.best_iteration)
        bs[name] = b
    m = ~np.isin(f, (2, 3, 4, 5))
    s[m] = sum(b.predict(X[m], num_iteration=b.best_iteration) for b in bs.values()) / 2
    return s, bs


def fit(frac, out, ho=False):
    t0 = time.time()
    res, per = {}, {}
    for c in C:
        D = pd.read_parquet(out / f"{'sc_ho' if ho else 'sc'}_{c}.parquet")
        id1, n_true, s1_fold = s1_pool(c)
        if ho:
            s1_fold = np.where(s1_fold == 0, pseudo_fold(np.arange(len(id1))), -1)
        samp = keep(np.arange(len(id1)), s1_fold, ALL, frac)          # smoke: the S1 of the 1% sample
        q1, qx, y, s3, f = (D[k].to_numpy() for k in ("q1", "qx", "y", "s3", "fold"))
        assert (s1_fold[q1] == f).all(), "frame fold = S1 fold"
        assert (np.bincount(q1[y], minlength=len(id1)) <= n_true).all()
        X = feats(D)
        print(f"{c}: {len(D):,} rows x {X.shape[1]} feats | {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        s, bs = fit_c(X, y, f)
        del X
        for k, b in bs.items():                                         # test side: mean of both, like folds 0/1/6-9
            b.save_model(str(out / f"{'sc_ho' if ho else 'sc'}_C{k}_{c}.txt"), num_iteration=b.best_iteration)
        m0 = f == 0                                                     # fold-0 scores for hybrid checks vs p54
        own = _owner(qx, s) == np.arange(len(s))
        pd.DataFrame({"q1": q1[m0], "qx": qx[m0], "p": D.p.to_numpy()[m0], "s": s[m0].astype(np.float32),
                      "own": own[m0]}).to_parquet(
            out / f"{'sc_ho' if ho else 'sc'}_s0_{c}.parquet", index=False)
        base, q_s, nt_s, half = baseline(c, id1)
        assert (n_true[q_s] == nt_s).all(), "gt n_true != P76 S n_true"
        ev = samp[q_s] & (s1_fold[q_s] == 0)                          # fold-0 S1 evaluated (all of S unless smoke / ho)
        sel1 = np.flatnonzero((s1_fold == 1) & samp)
        r = dict(rows=len(D), s1_fold0=int(ev.sum()), trees={k: b.best_iteration for k, b in bs.items()})
        fs = {"base": base[ev]}
        for arm, sc in (("stageB", D.p.to_numpy(np.float64)), ("stageC", s)):
            cand = owner_rank(q1, qx, s3, sc)
            t, f1 = tune(q1, y, cand, sc, sel1, n_true)
            fs[arm] = per_s1(q1, y, cand & (sc >= t), q_s[ev], n_true)
            r[arm] = dict(t=t, f_fold1=f1, f_fold0=float(fs[arm].mean()))
        r["owner_changed_frac"] = float(np.mean(_owner(qx, D.p.to_numpy(np.float64)) != _owner(qx, s)))
        r["f_fold0_base"] = float(fs["base"].mean())
        r["C_vs_base"], r["B_vs_base"], r["C_vs_B"] = cmp(fs["stageC"], fs["base"]), cmp(fs["stageB"], fs["base"]), \
            cmp(fs["stageC"], fs["stageB"])
        r["C_vs_base_half"] = {int(h): cmp(fs["stageC"][half[ev] == h], fs["base"][half[ev] == h]) for h in (0, 1)}
        imp = bs["A"].feature_importance("gain")
        r["gain_top"] = {k: round(float(v / imp.sum()), 4) for k, v in
                         sorted(zip(bs["A"].feature_name(), imp), key=lambda kv: -kv[1])[:15]}
        res[c] = r
        print(c, json.dumps({k: v for k, v in r.items() if k != "gain_top"}), f"| {time.time() - t0:.0f}s", flush=True)
        del D, s
    for k in ("C_vs_base", "B_vs_base", "C_vs_B"):
        res[f"lbeq_{k}"] = dict(mean=sum(LB_W[c] * res[c][k]["mean"] for c in C),
                                se=float(np.sqrt(sum((LB_W[c] * res[c][k]["se"]) ** 2 for c in C))))
    res["sec"] = round(time.time() - t0)
    dest = HERE / ("stagec_ho.json" if ho else "stagec.json") if frac == 1 else out / "stagec_smoke.json"
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("lbeq")}), "->", dest, flush=True)


def apply_test(c, test_p, out):
    """The fold-0 recipe on TEST: pruned test pairs (p = cf boosters' logit mean from score_test.py / p80-sct, files named
    like the X files under test_p), anchors on P3's test texts, mean of sc_CA / sc_CB, owner re-decision, fold-1 t, caps.
    -> <out>/sc_test_<c>.parquet: accepted (s1, sx, s) entity ids."""
    import lightgbm as lgb
    t0 = time.time()
    t = json.loads((HERE / "stagec.json").read_text())[c]["stageC"]["t"]
    parts = []
    for xf in sorted((W / "p4/X").glob(f"test_{c}_sx_*.parquet")):
        tp = sorted(Path(test_p).rglob(f"*{xf.name}"))
        assert len(tp) == 1, (xf.name, tp)
        o = pd.read_parquet(tp[0], columns=["q1", "qx", "p"])
        po = pd.Series(o.p.to_numpy(np.float64), index=keys(o.q1, o.qx))
        d = pq.read_table(xf, columns=["q1", "qx", "is_s3"] + RAW).to_pandas()
        p = po.reindex(keys(d.q1, d.qx)).to_numpy(np.float64)
        assert not np.isnan(p).any(), f"{xf.name}: test pairs without p"
        k = p >= PMIN
        parts.append(pd.DataFrame({"q1": d.q1.to_numpy(np.int32)[k], "qx": d.qx.to_numpy(np.int32)[k], "fold": np.int8(-1),
                                   "y": False, "s3": d.is_s3.to_numpy()[k] > 0.5, "p": p[k].astype(np.float32),
                                   **{r: d[r].to_numpy(np.float32)[k] for r in RAW}}))
        del d, o, po
    D = pd.concat(parts, ignore_index=True)
    anchors(c, D, split="test")
    X = feats(D)
    s = sum(lgb.Booster(model_file=str(out / f"sc_C{k}_{c}.txt")).predict(X) for k in "AB") / 2
    q1, qx, s3 = D.q1.to_numpy(), D.qx.to_numpy(), D.s3.to_numpy()
    acc = owner_rank(q1, qx, s3, s) & (s >= t)
    id1, idx = FT.ids("test", c)
    pd.DataFrame({"s1": id1.take(q1[acc]).to_numpy(zero_copy_only=False), "sx": idx.take(qx[acc]).to_numpy(zero_copy_only=False),
                  "s": s[acc].astype(np.float32)}).to_parquet(out / f"sc_test_{c}.parquet", index=False)
    print(f"{c} test: {len(D):,} pruned pairs, {acc.sum():,} accepted over {len(np.unique(q1[acc])):,} S1 (t {t}) | "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def hybrid(out, ho=False):
    """Today's decision (P76 p54 >= .71 on #8c argmax pairs) edited by stage C, knob tuned on one fold-0 half and reported
    on the other (both directions). veto: drop accepted pairs with s < v. steal: drop accepted pairs whose SX stage C
    gives to another S1, unless s >= w. rescue: add unaccepted argmax pairs that stage C owns with s >= u."""
    grids = {"veto": np.r_[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], "steal": np.r_[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01],
             "rescue": np.r_[0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]}
    res = {}
    for c in C:
        id1, idx = FT.ids("train", c)
        F = pd.read_parquet(W / f"p76/F_{c}.parquet", columns=["g", "y", "p54", "sx_id"])
        S = pd.read_parquet(W / f"p76/S_{c}.parquet", columns=["id", "n_true", "half"])
        q = pc.index_in(pa.array(S.id.to_numpy(object), pa.large_string()), value_set=id1).to_numpy()
        qx = pc.index_in(pa.array(F.sx_id.to_numpy(object), pa.large_string()), value_set=idx)
        assert qx.null_count == 0
        s0 = pd.read_parquet(out / f"{'sc_ho' if ho else 'sc'}_s0_{c}.parquet")
        k0 = pd.Series(np.arange(len(s0)), index=keys(s0.q1, s0.qx))
        j = k0.reindex(keys(q[F.g.to_numpy()], qx.to_numpy())).to_numpy()
        hit = ~np.isnan(j)
        j = np.where(hit, j, 0).astype(np.int64)
        s = np.where(hit, s0.s.to_numpy()[j], np.nan)                   # pruned by stage C's p floor: no opinion
        own = np.where(hit, s0.own.to_numpy()[j], True)
        g, y, n, nt, h = F.g.to_numpy(), F.y.to_numpy(), len(S), S.n_true.to_numpy(), S.half.to_numpy()
        base = F.p54.to_numpy() >= T_BASE
        ff = lambda acc: fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), nt)
        f0 = ff(base)
        arms = {"veto": lambda v: base & ~(s < v), "steal": lambda w: base & (own | (s >= w)),
                "rescue": lambda u: base | (hit & own & (s >= u))}
        r = {"pruned_or_missing_argmax_pairs": float(1 - hit.mean())}
        for arm, fn in arms.items():
            fs = {x: ff(fn(x)) - f0 for x in grids[arm]}
            r[arm] = {}
            for fit_h in (0, 1):
                best = max(grids[arm], key=lambda x: fs[x][h == fit_h].mean())
                d = fs[best][h == 1 - fit_h]
                r[arm][f"h{fit_h}->h{1 - fit_h}"] = dict(knob=float(best), mean=float(d.mean()),
                                                         se=float(d.std(ddof=1) / np.sqrt(len(d))))
            r[arm]["all_knobs_h0h1"] = {float(x): [round(float(fs[x][h == k].mean()), 7) for k in (0, 1)] for x in grids[arm]}
        res[c] = r
        print(c, json.dumps(r), flush=True)
    for arm in ("veto", "steal", "rescue"):
        res[f"lbeq_{arm}"] = {k: sum(LB_W[c] * res[c][arm][k]["mean"] for c in C) for k in ("h0->h1", "h1->h0")}
    (HERE / ("hybrid_ho.json" if ho else "hybrid.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("lbeq")}), flush=True)


def _owner(qx, s):
    """Row index of each row's SX owner (best s), per row."""
    o = np.lexsort((-s, qx))
    first = np.r_[True, qx[o][1:] != qx[o][:-1]]
    return o[np.flatnonzero(first)][np.cumsum(first) - 1][np.argsort(o)]


def selftest():
    # ranks: two groups, ties broken by order; rival = best other row
    key, p = np.array([5, 5, 5, 7]), np.array([0.2, 0.9, 0.5, 0.3])
    R = ranks(key, p)
    assert R["rank"].tolist() == [2, 0, 1, 0] and R["size"].tolist() == [3, 3, 3, 1] and R["oth"].tolist() == [1, 2, 1, -1]
    assert np.allclose(R["up"][[0, 2]], [0.5, 0.9]) and np.isnan(R["up"][1]) and np.allclose(R["dn"][[1, 2]], [0.5, 0.2])
    # anchors: S1 0 has anchors rows 0, 2; row 1 (S1 0) pairs with both, row 0 with 2, S1 1 none
    i, j = anchor_pairs(np.array([0, 0, 0, 1]), np.array([True, False, True, False]))
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 2), (1, 0), (1, 2), (2, 0)]
    # owner + caps: SX 9 claimed by S1 0 (.8) and S1 1 (.95) -> S1 1; S1 0 has 6 S2 above t -> only 5 kept
    q1 = np.array([0, 1] + [0] * 6)
    qx = np.array([9, 9, 1, 2, 3, 4, 5, 6])
    s3 = np.zeros(8, bool)
    s = np.array([0.8, 0.95, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85])
    acc = owner_rank(q1, qx, s3, s)
    assert acc.tolist() == [False, True, True, True, True, True, True, False], acc
    assert _owner(qx, s).tolist() == [1, 1, 2, 3, 4, 5, 6, 7]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("frame", "fit", "test", "hybrid", "selftest"))
    ap.add_argument("country", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ho", action="store_true", help="rehearsal on fold 0 alone (m2_big p, 6 pseudo folds)")
    ap.add_argument("--out", type=Path, default=W / "p80")
    ap.add_argument("--test-p", type=Path, help="test: dir with score_test.py's per-X-file test p")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frac = 0.01 if a.smoke else 1.0
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "hybrid":
        hybrid(a.out, a.ho)
    elif a.cmd == "test":
        apply_test(a.country, a.test_p, a.out)
    elif a.cmd == "frame":
        frame(a.country, frac, a.out, a.ho)
    else:
        fit(frac, a.out, a.ho)
