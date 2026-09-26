"""P27 SageMaker job: the LOCO 'tgtonly' arm applied to France test (an LB-test candidate, NOT gate-validated).

A France-only LightGBM is fit on France's own pseudo-labels (P27 plain_cls0: argmax & p >= .98 & house numbers equal
or missing -> 1, p <= .02 -> 0; p = #9ab's France p), cross-fit on two S1-hash halves (each half is scored by the
model fit on the other), early-stopped on US+India train fold 1 x0.2. Then p' = sigmoid((1-w) logit p + w logit p_fr)
for w in hp['ws']. Per w: France profile + label-free thresholds (#9ab's empty rate at t 0.985 = the LB-best France
operating point, D4's 6.85%, and P6's 5.75% reference).

Channel 'train' = amc26-p16/data/v1 (validation only). Channel 'test' = p27.py prep_fr: keys.parquet (#9ab rows:
q1, qx, p, p_gbdt), X/test_France_sx_*.parquet (IDF-fixed France X), hn_France_{s1,sx}.npy, meta.json, m2_big.txt.
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)
DEFAULTS = {"s3": "", "deadline_s": 5400, "ws": [0.1, 0.2, 0.35, 0.5], "rounds": 1500, "t_ref": 0.985,
            "variants": ["full", "nohn"]}
HNF = {"hn_eq", "hn_both", "hn_diff", "addr_rule", "rule_acc"}      # structured house-number evidence (P27 LOCO _nohn)
TS = np.r_[np.round(np.arange(0.5, 0.99, 0.005), 3), 0.99, 0.992, 0.995, 0.997, 0.998, 0.999]
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg):
    print(f"[p27fr {time.time() - T0:6.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def argmax(qx, p):
    o = np.lexsort((-p, qx))
    first = np.r_[True, qx[o][1:] != qx[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


def profile(q1, qx, p, n_s1):
    """Per t: empty-S1 rate over ALL France S1, matches per S1."""
    am = argmax(qx, p)
    out = []
    for t in TS:
        m = am & (p >= t)
        n = np.bincount(q1[m], minlength=n_s1)
        out.append((float((n == 0).mean()), float(n.sum() / n_s1)))
    return np.array(out)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    import pandas as pd
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    aux, model_dir = Path(test_dir), Path(model_dir)
    P = dict(PARAMS, num_threads=os.cpu_count())
    s3c = None
    if hp["s3"]:
        import boto3
        bucket, prefix = hp["s3"][5:].rstrip("/").split("/", 1)
        s3c = boto3.client("s3")
    put = lambda f, k: s3c and s3c.upload_file(str(f), bucket, f"{prefix}/{k}")
    log(f"lightgbm {lgb.__version__}, hp {hp}")
    # France X (every blocked pair) -> the #9ab rows
    fs = sorted((aux / "X").glob("test_France_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs[0]).names if c not in DROP]
    parts = [pq.read_table(f, columns=["q1", "qx"] + cols) for f in fs]
    xq1 = np.concatenate([t.column("q1").to_numpy() for t in parts]).astype(np.int64)
    xqx = np.concatenate([t.column("qx").to_numpy() for t in parts]).astype(np.int64)
    X = np.empty((len(xq1), len(cols)), np.float32)
    a = 0
    for t in parts:
        for j, c in enumerate(cols):
            X[a:a + len(t), j] = t.column(c).to_numpy().astype(np.float32)
        a += len(t)
    del parts
    k = pd.read_parquet(aux / "keys.parquet")
    q1, qx, p0, pg = (k[c].to_numpy() for c in ("q1", "qx", "p", "p_gbdt"))
    key_x = (xq1 << 32) | xqx
    o = np.argsort(key_x)
    pos = np.searchsorted(key_x[o], (q1 << 32) | qx)
    assert np.all(key_x[o][pos] == (q1 << 32) | qx), "a #9ab row is missing from the France X"
    X = X[o[pos]]
    n_s1 = json.load(open(aux / "meta.json"))["n_s1"]
    log(f"France X {len(key_x):,} pairs -> {len(X):,} #9ab rows, {len(cols)} feats")
    res = {"hp": hp, "ts": TS.tolist()}
    m2 = lgb.Booster(model_file=str(aux / "m2_big.txt"))
    fi = [cols.index(f) for f in m2.feature_name()]
    chk = m2.predict(X[:200_000][:, fi], num_threads=os.cpu_count())
    res["m2_check"] = {"max_abs": float(np.abs(chk - pg[:200_000]).max()), "corr": float(np.corrcoef(chk, pg[:200_000])[0, 1])}
    log(f"m2_big on the IDF-fixed X vs #9ab p_gbdt (first 200k rows): {res['m2_check']}")
    del m2, chk
    # pseudo-labels (P27 plain_cls0) on #9ab's p
    h1, hx = np.load(aux / "hn_France_s1.npy"), np.load(aux / "hn_France_sx.npy")
    a1, bx = h1[q1], hx[qx]
    cls0 = (a1 < 0) | (bx < 0) | (a1 == bx)
    am = argmax(qx, p0)
    pos_ = am & (p0 >= 0.98) & cls0
    neg_ = (p0 <= 0.02) & ~pos_
    res["labels"] = {"n_pos": int(pos_.sum()), "n_neg": int(neg_.sum()), "n_rows": len(p0),
                     "n_conf_not_cls0": int((am & (p0 >= 0.98) & ~cls0).sum())}
    log(f"pseudo-labels {res['labels']}")
    # validation: US+India train fold 1 x0.2 (P6)
    vx, vy = [], []
    for f in sorted(Path(train_dir).glob("p4/X/train_*_sx_*.parquet")):
        t = pq.read_table(f, columns=["q1", "fold", "is_true"] + cols)
        qq = t.column("q1").to_numpy().astype(np.uint64)
        m = (t.column("fold").to_numpy() == 1) & ((qq * np.uint64(2654435761)) % np.uint64(1000) < 200)
        vx.append(np.column_stack([t.column(c).to_numpy().astype(np.float32)[m] for c in cols]))
        vy.append(t.column("is_true").to_numpy()[m])
    vx, vy = np.concatenate(vx), np.concatenate(vy).astype(np.float32)
    log(f"val {len(vy):,} pairs")
    half = ((q1.astype(np.uint64) * np.uint64(2654435761) + np.uint64(11)) % np.uint64(2)).astype(bool)
    base = profile(q1, qx, p0, n_s1)
    i_ref = int(np.argmin(np.abs(TS - hp["t_ref"])))
    e_ref = float(base[i_ref, 0])
    res["base"] = {"empty_at_ref": e_ref, "mps_at_ref": float(base[i_ref, 1])}
    out = pd.DataFrame(index=np.arange(len(p0)))
    acc0 = argmax(qx, p0) & (p0 >= hp["t_ref"])
    for v in hp["variants"]:
        fi = [i for i, c in enumerate(cols) if not (v == "nohn" and c in HNF)]
        pfr = np.zeros(len(p0), np.float32)
        trees = []
        for h in (False, True):
            tr = (half == h) & (pos_ | neg_)
            ds = lgb.Dataset(X[tr][:, fi], pos_[tr].astype(np.float32), feature_name=[cols[i] for i in fi], free_raw_data=True)
            dv = lgb.Dataset(vx[:, fi], vy, reference=ds)
            m = lgb.train(P, ds, hp["rounds"], valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            sc = half != h
            pfr[sc] = m.predict(X[sc][:, fi], num_threads=os.cpu_count())
            trees.append(m.num_trees())
            m.save_model(str(model_dir / f"fr_tgtonly_{v}_half{int(h)}.txt"))
            log(f"{v} half {int(h)}: fit on {int(tr.sum()):,} rows, {m.num_trees()} trees, scored {int(sc.sum()):,}")
        res[f"trees_{v}"] = trees
        out[f"p_fr_{v}"] = pfr
        for w in hp["ws"]:
            pw = (1 / (1 + np.exp(-((1 - w) * lg(p0) + w * lg(pfr))))).astype(np.float32)
            pr = profile(q1, qx, pw, n_s1)
            r = {}
            for nm, target in (("ref", e_ref), ("d4", 0.0685), ("p6", 0.0575)):
                i = int(np.argmin(np.abs(pr[:, 0] - target)))
                acc = argmax(qx, pw) & (pw >= TS[i])
                ch = np.unique(np.r_[q1[acc & ~acc0], q1[acc0 & ~acc]])
                r[nm] = {"t": float(TS[i]), "empty": float(pr[i, 0]), "mps": float(pr[i, 1]),
                         "s1_changed_vs_9ab": int(len(ch)), "added": int((acc & ~acc0).sum()),
                         "removed": int((acc0 & ~acc).sum()),
                         "added_not_cls0": int((acc & ~acc0 & ~cls0).sum()), "removed_not_cls0": int((acc0 & ~acc & ~cls0).sum())}
            res[f"{v}_w{w}"] = r
            out[f"p_{v}_w{w}"] = pw
            log(f"{v} w {w}: {r}")
    out.to_parquet(model_dir / "fr_blend.parquet")
    (model_dir / "result.json").write_text(json.dumps(res))
    put(model_dir / "fr_blend.parquet", "fr_blend.parquet")
    put(model_dir / "result.json", "result.json")
    log("done")
