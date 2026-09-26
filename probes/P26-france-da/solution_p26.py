"""P26 SageMaker job (CPU): France domain adaptation of the m2-style LightGBM matcher. Shipped alone as solution.py.

Channel 'train' = p4/X/train_<c>_sx_NNN.parquet (s3 amc26-p16/data/v1/, P4's full-corpus train X, 65 features).
Channel 'test'  = meta/{train_India,train_US,test_France}.npz (prep.py: house numbers per q1/qx, fold-0 truth),
                  fr/X/test_France_sx_NNN.parquet (P14's IDF-fixed France X), fr/keys_France.parquet (#6-pruner keys,
                  m2_big p), m2_big.txt.
Arms (hp['arms'], each {"name", "w": 0|1, "hn": 0|1, "w_pow": float}):
  w  = importance weights p_t/(1-p_t) from an adversarial LightGBM (source rows vs target rows, on the ADV[arm['adv']]
       feature set, default "sim"), p clipped to [0.02, 0.98], w**w_pow clipped to hp["w_clip"], mean 1 over training rows.
  hn = + 5 generator-aware house-number features on the full candidate list (label-free): hn_cls (P15 class), hn_d
       (signed hx - h1, clip +-100), hn_nsame (other SX of the S1 with this SX's hn), hn_nssame (their best name_cos),
       hn_ns1hn (other SX of the S1 carrying the S1's hn).
  ctx = "pruned": the 8 candidate-list context features (s1/sx _n _gap _margin, block ranks) recomputed over the #6-pruned
       list (pruner q >= 0.03 | rule_acc, blocked pairs), trained on kept pairs only (P20: France's sx_margin is set by a
       competitor the pruner removes 93% of the time). train = "kept": train on kept pairs with the original context.
  Evaluation is always on the pruned list (hp['prune'] = 1, test-faithful: #6 onward score only kept pairs).
mode 'loco': train on hp['src'] fold 2, early-stop on its fold 1 (x0.25); target = hp['tgt'] (unlabelled for training).
  Every arm scores fold 0 of both countries (argmax per SX over all train S1 = full corpus); source t* = best F0.5,
  target t = P6's label-free rule (target predicted-empty rate = the source's at t*, searched on [t*, 0.99]).
mode 'full': m2_big's recipe (folds 2-5, early stop fold 1 x0.25, lr 0.05) + the arm, weights toward France test.
  Scores the holdout (fold 0, both countries; paired vs m2_big on the same rows, t tuned on half 0, reported on half 1)
  and France's pruned keys -> pred/test_France.parquet (q1, qx, p, p_m2).
-> model_dir + hp['s3']: result.json, status.json, pred/*.parquet, <arm>.txt boosters.
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
ID = ["q1", "qx", "is_true", "fold"]
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}      # model.DROP
HN = ["hn_cls", "hn_d", "hn_nsame", "hn_nssame", "hn_ns1hn"]
CTX = ["s1_n", "s1_gap", "s1_margin", "sx_n", "sx_gap", "sx_margin", "block_rank_s1", "block_rank_sx"]
TAU = 0.03                                                               # prune.TAU
SIB = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]                                  # P15 hn_cls sibling offsets
# adversarial feature sets (Phase A): all 65 features separate France / US / India at AUC 0.998 (weights collapse);
# the 29 pair-similarity features at 0.956, names only at 0.82 (phase_a2.json)
SIM = ["name_ratio", "name_partial", "name_tset", "name_tsort", "name_jw", "core_ratio", "core_partial", "core_tset",
       "core_tsort", "core_jw", "core_eq", "core_jacc", "core_first_eq", "name_cos", "addr_cos", "addr_tset",
       "addr_ratio", "addr_jacc", "hn_eq", "hn_diff", "city_eq", "city_diff", "web_eq", "web_ratio", "web_partial",
       "sq_eq", "acro", "block_score", "rule_acc"]
ADV = {"sim": SIM, "names": SIM[:14], "all": None}
BASE = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
        "verbose": -1}                                                   # m2_big (P25 variants.json _base)
DEFAULTS = {"mode": "loco", "src": "US", "tgt": "India", "arms": None, "lr": 0.1, "rounds": 3000, "es": 100,
            "folds": [2], "val_frac": 0.25, "adv_n": 1_000_000, "w_clip": [0.1, 10.0],
            "deadline_s": 9000, "s3": "", "tiny": 0.0, "prune": 1, "booster": "", "tars": {}}
TS = np.round(np.arange(0.02, 0.995, 0.01), 2)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg: str):
    print(f"[p26 {time.time() - T0:7.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


class S3:
    """Best-effort uploads under s3://bucket/prefix/; '' disables; a failed upload is logged, never fatal."""

    def __init__(self, uri: str):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def put(self, path: Path, key: str):
        if self.on:
            try:
                self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")

    def put_json(self, obj: dict, key: str):
        if self.on:
            try:
                self.c.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=json.dumps(obj, default=float).encode())
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")


# ---------------------------------------------------------------- data
class Country:
    """Every pair of one country's X files in memory: X float32 [n, 65], q1, qx, fold, y, + hn features [n, 5]."""

    def __init__(self, files: list, meta: dict, cols: list, tiny: float = 0.0, pruner=None, keys=None):
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        has_y = "is_true" in pq.read_schema(files[0]).names
        self.X = np.empty((n, len(cols)), np.float32)
        self.q1, self.qx = np.empty(n, np.int32), np.empty(n, np.int32)
        self.fold, self.y = np.full(n, -1, np.int8), np.zeros(n, bool)
        a = 0
        for f in files:
            t = pq.read_table(f, columns=(ID if has_y else ["q1", "qx"]) + cols)
            k = len(t)
            for j, c in enumerate(cols):
                self.X[a:a + k, j] = t.column(c).to_numpy().astype(np.float32)          # nulls -> NaN
            self.q1[a:a + k], self.qx[a:a + k] = t.column("q1").to_numpy(), t.column("qx").to_numpy()
            if has_y:
                self.fold[a:a + k], self.y[a:a + k] = t.column("fold").to_numpy(), t.column("is_true").to_numpy()
            a += k
            del t
        assert a == n
        if tiny:                                    # local rehearsal: keep a hash sample of SX (whole candidate lists of SX)
            m = (self.qx.astype(np.uint64) * np.uint64(2654435761) % np.uint64(1000)) < tiny * 1000
            for k in ("X", "q1", "qx", "fold", "y"):
                setattr(self, k, getattr(self, k)[m])
        self.meta, self.cols = meta, cols
        self.H = hn_feats(self.q1, self.qx, self.X[:, cols.index("name_cos")], meta)
        self.keep, self.prune_info = np.ones(len(self.q1), bool), {}
        if pruner is not None:                      # #6's pruner: q >= TAU or rule-accepted
            ix = [cols.index(c) for c in pruner.feature_name()]
            q = np.concatenate([pruner.predict(self.X[a:a + 2_000_000][:, ix], num_threads=os.cpu_count())
                                for a in range(0, len(self.q1), 2_000_000)])
            self.keep = (q >= TAU) | (self.X[:, cols.index("rule_acc")] > 0)
            self.prune_info = {"kept": float(self.keep.mean()), "true_kept": float(self.keep[self.y].mean()) if self.y.any() else None}
        if keys is not None:                        # France: the shipped key set is the truth; report pruner agreement
            kk = (self.q1.astype(np.int64) << 32) | self.qx.astype(np.int64)
            ks = np.isin(kk, (keys.q1.to_numpy(np.int64) << 32) | keys.qx.to_numpy(np.int64))
            self.prune_info |= {"keys": int(ks.sum()), "pruner_vs_keys_agree": float((ks == self.keep).mean())}
            self.keep = ks
        self.CX = pruned_ctx(self.q1, self.qx, self.X[:, cols.index("block_score")],
                             self.X[:, cols.index("blocked")] > 0, self.keep)
        self.ci = [cols.index(c) for c in CTX]

    def mat(self, m, hn: bool, ctxp: bool = False, hni=None):
        X = self.X[m]
        if ctxp:
            X[:, self.ci] = self.CX[m]
        return np.hstack([X, self.H[m] if hni is None else self.H[m][:, hni]]) if hn else X


def pruned_ctx(q1, qx, bs, blocked, keep) -> np.ndarray:
    """[n, 8] = CTX recomputed over the kept blocked pairs (features.context / block ranks); NaN elsewhere."""
    out = np.full((len(q1), len(CTX)), np.nan, np.float32)
    r = np.flatnonzero(keep & blocked)
    s = np.nan_to_num(bs[r])
    for j, key in ((0, q1[r]), (3, qx[r])):
        u, inv, n = np.unique(key, return_inverse=True, return_counts=True)
        best = np.full(len(u), -np.inf, np.float32)
        np.maximum.at(best, inv, s)
        first = np.zeros(len(s), bool)
        o = np.flatnonzero(s == best[inv])
        first[o[np.unique(inv[o], return_index=True)[1]]] = True
        sec = np.full(len(u), -np.inf, np.float32)
        np.maximum.at(sec, inv[~first], s[~first])
        other = np.where(first, sec[inv], best[inv])
        out[r, j], out[r, j + 1] = n[inv], best[inv] - s
        out[r, j + 2] = np.where(np.isfinite(other), s - other, 1.0)
        o = np.lexsort((-s, key))                   # rank by block_score within the group, 0 = best
        ks = key[o]
        st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        rk = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
        out[r[o], 6 if j == 0 else 7] = rk
    return out


def hn_feats(q1, qx, nc, M) -> np.ndarray:
    """[n, 5] float32 label-free house-number features over the full candidate list (see module doc)."""
    a, la, b, lb = M["h1"][q1], M["l1"][q1], M["hx"][qx], M["lx"][qx]
    both = (a >= 0) & (b >= 0)
    dd = b - a
    diff = both & (dd != 0)
    sib = diff & np.isin(dd, SIB)
    ls, ll = np.minimum(la, lb).astype(np.int64), np.maximum(la, lb).astype(np.int64)
    short, lng = np.where(la < lb, a, b), np.where(la < lb, b, a)
    ends = (lng % (10 ** ls)) == short
    starts = (lng // (10 ** np.clip(ll - ls, 0, 18))) == short
    dig = diff & ~sib & (la != lb) & (ends | starts)
    small = diff & ~sib & ~dig & (np.abs(dd) <= 21)
    far = diff & ~sib & ~dig & ~small
    cls = (1 * sib + 2 * dig + 3 * small + 4 * far).astype(np.float32)
    d = np.where(both, np.clip(dd, -100, 100), np.nan).astype(np.float32)
    # support: groups (q1, hx) among SX that carry a house number
    n = len(q1)
    nsame, nssame = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    hx = np.flatnonzero(b >= 0)
    if len(hx):
        key = (q1[hx].astype(np.int64) << 31) | b[hx]
        v = np.nan_to_num(nc[hx], nan=0.0)
        o = np.lexsort((-v, key))
        ks, vs = key[o], v[o]
        start = np.r_[True, ks[1:] != ks[:-1]]
        gid = np.cumsum(start) - 1
        size = np.bincount(gid)
        first = np.flatnonzero(start)
        gmax = vs[first]
        gsec = np.where(size >= 2, vs[np.minimum(first + 1, len(vs) - 1)], np.nan)
        other = np.where(start, gsec[gid], gmax[gid])                  # best name_cos among the OTHER members
        ns = (size[gid] - 1).astype(np.float32)
        nsame[hx[o]] = ns
        nssame[hx[o]] = np.where(ns > 0, other, np.nan)
    eqh = (both & (dd == 0)).astype(np.float64)
    ns1 = (np.bincount(q1, weights=eqh, minlength=int(q1.max()) + 1)[q1] - eqh).astype(np.float32)
    return np.column_stack([cls, d, nsame, nssame, ns1]).astype(np.float32)


def hsh(q, mod=1000):
    return (q.astype(np.uint64) * np.uint64(2654435761) >> np.uint64(7)) % np.uint64(mod)


# ---------------------------------------------------------------- model pieces
def adversarial(src_X, tgt_X, cols, adv, n_max, seed=0):
    """LightGBM P(target | x) on the ADV[adv] columns; returns (booster, kept column idx, held-out AUC, top gain)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    keep = [j for j, c in enumerate(cols) if ADV[adv] is None or c in ADV[adv]]
    s = src_X[rng.choice(len(src_X), min(n_max, len(src_X)), replace=False)][:, keep]
    t = tgt_X[rng.choice(len(tgt_X), min(n_max, len(tgt_X)), replace=False)][:, keep]
    n = min(len(s), len(t))
    X, y = np.vstack([s[:n], t[:n]]), np.r_[np.zeros(n), np.ones(n)]
    del s, t
    te = rng.random(len(y)) < 0.2
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               verbose=-1, num_threads=os.cpu_count())
    m = lgb.train(prm, lgb.Dataset(X[~te], y[~te]), 300)
    auc = float(roc_auc_score(y[te], m.predict(X[te])))
    gain = pd.Series(m.feature_importance("gain"), index=[cols[j] for j in keep])
    return m, keep, auc, (gain / gain.sum()).sort_values(ascending=False).head(10).round(4).to_dict()


def weights(adv, keep, X, w_pow, clip, norm=True):
    p = np.clip(adv.predict(X[:, keep], num_threads=os.cpu_count()), 0.02, 0.98)
    w = np.clip((p / (1 - p)) ** w_pow, *clip)
    return (w / w.mean() if norm else w).astype(np.float32)


def fit(Xtr, ytr, wtr, Xva, yva, wva, names, params, rounds, es, deadline_s):
    import lightgbm as lgb
    d = lgb.Dataset(Xtr, ytr.astype(np.float32), weight=wtr, feature_name=names, free_raw_data=True)
    dv = lgb.Dataset(Xva, yva.astype(np.float32), weight=wva, reference=d)
    best = {"i": -1, "v": np.inf, "res": None}

    def guard(env):
        _, _, v, _ = env.evaluation_result_list[0]
        if v < best["v"]:
            best.update(i=env.iteration, v=v, res=env.evaluation_result_list)
        if (env.iteration + 1) % 250 == 0:
            log(f"  iter {env.iteration + 1} val_loss={v:.6f} best {best['v']:.6f} @ {best['i'] + 1}")
        if time.time() - T0 > deadline_s:
            log(f"  deadline at iter {env.iteration + 1}: keeping {best['i'] + 1}")
            raise lgb.callback.EarlyStopException(best["i"], best["res"])

    guard.order = 40
    m = lgb.train(dict(params, num_threads=os.cpu_count()), d, rounds, valid_sets=[dv],
                  callbacks=[guard, lgb.early_stopping(es, verbose=False)])
    return m, float(best["v"])


# ---------------------------------------------------------------- evaluation
def argmax_sx(s1, sx, p):
    o = np.lexsort((-p, sx))
    first = np.r_[True, sx[o][1:] != sx[o][:-1]]
    m = np.zeros(len(p), bool)
    m[o[first]] = True
    return m


def f05(tp, npred, ntrue):
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    return np.where((npred == 0) & (ntrue == 0), 1.0, np.where(tp == 0, 0.0, f))


class Eval:
    """Fold-0 evaluation of one country: rows = every pair of an SX with a fold-0 candidate (full-corpus argmax)."""

    def __init__(self, C: Country):
        self.rows = np.flatnonzero(C.keep & np.isin(C.qx, np.unique(C.qx[C.keep & (C.fold == 0)])))
        self.C, self.t0, self.nt = C, C.meta["t0_q1"], C.meta["t0_n"].astype(float)
        self.half = hsh(self.t0, 2).astype(np.int8)

    def per_s1(self, p):
        """-> (k, pk, yk): fold-0 argmax pairs as (S1 index into t0, p, is_true)."""
        C, r = self.C, self.rows
        am = argmax_sx(C.q1[r], C.qx[r], p)
        sel = am & (C.fold[r] == 0)
        k = np.searchsorted(self.t0, C.q1[r][sel])
        assert (self.t0[k] == C.q1[r][sel]).all()
        return k, p[sel], C.y[r][sel]

    def at(self, kpy, t):
        k, pk, yk = kpy
        m = pk >= t
        n = len(self.t0)
        npred = np.bincount(k[m], minlength=n).astype(float)
        tp = np.bincount(k[m & yk], minlength=n).astype(float)
        return f05(tp, npred, self.nt), npred

    def curve(self, kpy, mask=None):
        rows = []
        for t in TS:
            f, npred = self.at(kpy, t)
            if mask is not None:
                f, npred = f[mask], npred[mask]
            rows.append((t, f.mean(), float((npred == 0).mean()), npred.mean()))
        return pd.DataFrame(rows, columns=["t", "f05", "emp", "mps"])


def _paired(a, b):
    d = a - b
    return {"delta": float(d.mean()), "se": float(d.std() / np.sqrt(len(d))), "up": int((d > 0).sum()),
            "down": int((d < 0).sum())}


# ---------------------------------------------------------------- modes
def arms_of(hp):
    return hp["arms"] or [{"name": "a0", "w": 0, "hn": 0}, {"name": "a1", "w": 1, "hn": 0},
                          {"name": "a2", "w": 0, "hn": 1}, {"name": "a3", "w": 1, "hn": 1}]


def loco(train_dir, test_dir, model_dir, hp, s3):
    import lightgbm as lgb
    fs = lambda c: sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs("US")[0]).names if c not in DROP]
    src, tgt = hp["src"], hp["tgt"]
    pr = lgb.Booster(model_file=str(Path(test_dir) / "pruner.txt")) if hp["prune"] else None
    D = {c: Country(fs(c), dict(np.load(Path(test_dir) / f"meta/train_{c}.npz")), cols, hp["tiny"], pr) for c in (src, tgt)}
    S, T = D[src], D[tgt]
    log(f"loaded {src} {len(S.q1):,} {S.prune_info} / {tgt} {len(T.q1):,} {T.prune_info}")
    tr = np.isin(S.fold, hp["folds"])
    va = (S.fold == 1) & (hsh(S.q1) < hp["val_frac"] * 1000)
    E = {c: Eval(D[c]) for c in (src, tgt)}
    res = {"src": src, "tgt": tgt, "n_train": int(tr.sum()), "n_val": int(va.sum()), "arms": {},
           "prune": {src: S.prune_info, tgt: T.prune_info}}
    advs, res["adv"] = {}, {}
    for a in {a.get("adv", "sim") for a in arms_of(hp) if a.get("w")}:   # unlabelled pools: fold 3 of each country
        advs[a] = adversarial(S.X[S.fold == 3], T.X[T.fold == 3], cols, a, hp["adv_n"])
        res["adv"][a] = {"auc": advs[a][2], "top_gain": advs[a][3], "feats": len(advs[a][1])}
        log(f"adversarial[{a}] {src} vs {tgt}: {res['adv'][a]}")
    F = {}
    params = dict(BASE, learning_rate=hp["lr"])
    for arm in arms_of(hp):
        name, hn, cp = arm["name"], bool(arm.get("hn")), arm.get("ctx") == "pruned"
        kept = arm.get("train", "kept" if cp else "full") == "kept"     # ctx pruned + train full: non-kept rows get NaN ctx
        hni = [HN.index(c) for c in arm["hn_cols"]] if arm.get("hn_cols") else None
        tr_, va_ = (tr & S.keep, va & S.keep) if kept else (tr, va)
        wtr = wva = None
        info = {"n_train": int(tr_.sum())}
        if arm.get("w"):
            adv, keep = advs[arm.get("adv", "sim")][:2]
            wtr = weights(adv, keep, S.X[tr_], arm.get("w_pow", 1.0), hp["w_clip"])
            wva = weights(adv, keep, S.X[va_], arm.get("w_pow", 1.0), hp["w_clip"])
            info["w_q"] = {q: float(np.quantile(wtr, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
            info["w_ess"] = float(wtr.sum() ** 2 / (wtr ** 2).sum() / len(wtr))
        hnn = [HN[i] for i in hni] if hni else HN
        names = cols + (hnn if hn else [])
        m, vl = fit(S.mat(tr_, hn, cp, hni), S.y[tr_], wtr, S.mat(va_, hn, cp, hni), S.y[va_], wva, names, params, hp["rounds"], hp["es"],
                    hp["deadline_s"])
        m.save_model(str(model_dir / f"{name}.txt"))
        info |= {"trees": m.num_trees(), "val_loss": vl}
        kp = {}
        for c in (src, tgt):
            p = m.predict(D[c].mat(E[c].rows, hn, cp, hni), num_threads=os.cpu_count())
            kp[c] = E[c].per_s1(p)
            del p
        cs, ct = E[src].curve(kp[src]), E[tgt].curve(kp[tgt])
        i = int(cs.f05.idxmax())
        ts, ref = float(cs.t[i]), float(cs.emp[i])
        j_or = int(ct.f05.idxmax())
        cand = ct[ct.t >= ts - 1e-9]
        j = int((cand.emp - ref).abs().idxmin())
        tr_ = float(ct.t[j])
        F[name] = {"src": E[src].at(kp[src], ts)[0], "tgt_rule": E[tgt].at(kp[tgt], tr_)[0],
                   "tgt_oracle": E[tgt].at(kp[tgt], float(ct.t[j_or]))[0]}
        info |= {"src_f05": float(cs.f05[i]), "src_t": ts, "src_empty": ref, "tgt_t_rule": tr_,
                 "tgt_f05_rule": float(ct.f05[j]), "tgt_empty_rule": float(ct.emp[j]),
                 "tgt_f05_oracle": float(ct.f05[j_or]), "tgt_t_oracle": float(ct.t[j_or]),
                 "tgt_f05_at_src_t": float(ct.f05[ct.t.sub(ts).abs().idxmin()]),
                 "tgt_empty_at_src_t": float(ct.emp[ct.t.sub(ts).abs().idxmin()]),
                 "tgt_true_empty": float((E[tgt].nt == 0).mean())}
        if name != "a0" and "a0" in F:
            for k in ("src", "tgt_rule", "tgt_oracle"):
                info[f"vs_a0_{k}"] = _paired(F[name][k], F["a0"][k])
        imp = pd.Series(m.feature_importance("gain"), index=names)
        info["hn_gain_share"] = float(imp[hnn].sum() / imp.sum()) if hn else 0.0
        info["top_gain"] = (imp / imp.sum()).sort_values(ascending=False).head(12).round(4).to_dict()
        res["arms"][name] = info
        pd.concat([cs.assign(c=src), ct.assign(c=tgt)]).to_csv(model_dir / f"curve_{name}.csv", index=False)
        s3.put(model_dir / f"curve_{name}.csv", f"curve_{name}.csv")
        log(f"{name}: {json.dumps(info, default=float)}")
        s3.put_json(res | {"state": f"arm {name} done"}, "status.json")
        del m
    return res


def full(train_dir, test_dir, model_dir, hp, s3):
    """Phase C: m2_big recipe + arm on US+India folds 2-5, weights toward France test."""
    import lightgbm as lgb
    arm = arms_of(hp)[0]
    name, hn, cp = arm["name"], bool(arm.get("hn")), arm.get("ctx") == "pruned"
    kept = arm.get("train", "kept" if cp else "full") == "kept"
    hni = [HN.index(c) for c in arm["hn_cols"]] if arm.get("hn_cols") else None
    hnn = [HN[i] for i in hni] if hni else HN
    fs = lambda c: sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs("US")[0]).names if c not in DROP]
    pr = lgb.Booster(model_file=str(Path(test_dir) / "pruner.txt"))
    D = {c: Country(fs(c), dict(np.load(Path(test_dir) / f"meta/train_{c}.npz")), cols, hp["tiny"], pr)
         for c in ("India", "US")}
    keys = pd.read_parquet(Path(test_dir) / "fr/keys_France.parquet")
    FR = Country(sorted(Path(test_dir).glob("fr/X/test_France_sx_*.parquet")),
                 dict(np.load(Path(test_dir) / "meta/test_France.npz")), cols, 0.0, pr, keys)
    log(f"loaded India {len(D['India'].q1):,} US {len(D['US'].q1):,} France {len(FR.q1):,} {FR.prune_info}")
    tr = {c: np.isin(D[c].fold, hp["folds"]) & (D[c].keep if kept else True) for c in D}
    va = {c: (D[c].fold == 1) & (hsh(D[c].q1) < hp["val_frac"] * 1000) & (D[c].keep if kept else True) for c in D}
    cat = lambda f: np.concatenate([f(c) for c in ("India", "US")])
    res = {"arm": arm, "n_train": int(sum(v.sum() for v in tr.values())), "n_val": int(sum(v.sum() for v in va.values())),
           "prune": {c: D[c].prune_info for c in D} | {"France": FR.prune_info}}
    wtr = wva = None
    if arm.get("w"):
        pool = cat(lambda c: D[c].X[np.isin(D[c].fold, [6, 7, 8, 9])][::4])
        adv, keep, auc, top = adversarial(pool, FR.X, cols, arm.get("adv", "sim"), hp["adv_n"])
        del pool
        wtr = cat(lambda c: weights(adv, keep, D[c].X[tr[c]], arm.get("w_pow", 1.0), hp["w_clip"], False))
        wva = cat(lambda c: weights(adv, keep, D[c].X[va[c]], arm.get("w_pow", 1.0), hp["w_clip"], False))
        wtr, wva = wtr / wtr.mean(), wva / wva.mean()
        res["adv"] = {"auc": auc, "top_gain": top, "feats": len(keep),
                      "w_q": {q: float(np.quantile(wtr, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
                      "w_ess": float(wtr.sum() ** 2 / (wtr ** 2).sum() / len(wtr)),
                      "w_mean_India": float(wtr[:tr["India"].sum()].mean()),
                      "w_mean_US": float(wtr[tr["India"].sum():].mean())}
        log(f"adversarial train vs France: {res['adv']}")
    names = cols + (hnn if hn else [])
    Xtr, ytr = cat(lambda c: D[c].mat(tr[c], hn, cp, hni)), cat(lambda c: D[c].y[tr[c]])
    Xva, yva = cat(lambda c: D[c].mat(va[c], hn, cp, hni)), cat(lambda c: D[c].y[va[c]])
    log(f"train {len(ytr):,} val {len(yva):,} feats {len(names)}")
    s3.put_json(res | {"state": "training"}, "status.json")
    m, vl = fit(Xtr, ytr, wtr, Xva, yva, wva, names, dict(BASE, learning_rate=hp["lr"]), hp["rounds"], hp["es"],
                hp["deadline_s"])
    del Xtr, ytr, Xva, yva
    m.save_model(str(model_dir / f"{name}.txt"))
    s3.put(model_dir / f"{name}.txt", f"{name}.txt")
    res |= {"trees": m.num_trees(), "val_loss": vl}
    imp = pd.Series(m.feature_importance("gain"), index=names)
    res["hn_gain_share"] = float(imp[hnn].sum() / imp.sum()) if hn else 0.0
    res["top_gain"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
    log(f"trained {res}")
    ctl = lgb.Booster(model_file=str(Path(test_dir) / "m2_big.txt"))
    pred = model_dir / "pred"
    pred.mkdir(exist_ok=True)
    res["holdout"] = {}
    for c in ("India", "US"):
        E = Eval(D[c])
        p = m.predict(D[c].mat(E.rows, hn, cp, hni), num_threads=os.cpu_count())
        p0 = ctl.predict(D[c].X[E.rows], num_threads=os.cpu_count())
        out = {}
        for tag, pp in (("arm", p), ("m2", p0)):
            kp = E.per_s1(pp)
            cu = E.curve(kp, E.half == 0)
            t = float(cu.t[cu.f05.idxmax()])
            f = E.at(kp, t)[0]
            out[tag] = {"t_half0": t, "f05_half1": float(f[E.half == 1].mean()), "f": f}
            c71 = E.at(kp, 0.71)[0]
            out[tag]["f05_half1_t71"] = float(c71[E.half == 1].mean())
            out[tag]["f71"] = c71
        h1 = E.half == 1
        res["holdout"][c] = {"arm_t": out["arm"]["t_half0"], "m2_t": out["m2"]["t_half0"],
                             "arm_f05_h1": out["arm"]["f05_half1"], "m2_f05_h1": out["m2"]["f05_half1"],
                             "vs_m2_h1": _paired(out["arm"]["f"][h1], out["m2"]["f"][h1]),
                             "vs_m2_h1_t71": _paired(out["arm"]["f71"][h1], out["m2"]["f71"][h1]),
                             "n_h1": int(h1.sum())}
        np.save(pred / f"holdout_f_{c}.npy", np.vstack([out["arm"]["f"], out["m2"]["f"], E.half]))
        log(f"holdout {c}: {res['holdout'][c]}")
        del p, p0
    # France: pruned keys
    k = keys
    want = pd.Series(np.arange(len(k)), index=(k.q1.to_numpy(np.int64) << 32) | k.qx.to_numpy(np.int64))
    pos = want.reindex((FR.q1.astype(np.int64) << 32) | FR.qx.astype(np.int64)).to_numpy()
    rows = np.flatnonzero(~np.isnan(pos))
    assert len(rows) == len(k), f"France keys found {len(rows)} of {len(k)}"
    p = m.predict(FR.mat(rows, hn, cp, hni), num_threads=os.cpu_count())
    p0 = ctl.predict(FR.X[rows], num_threads=os.cpu_count())
    fr = pd.DataFrame({"q1": FR.q1[rows], "qx": FR.qx[rows], "p": p.astype(np.float32), "p_m2": p0.astype(np.float32)})
    ref = k.set_index((k.q1.to_numpy(np.int64) << 32) | k.qx.to_numpy(np.int64)).p
    res["france"] = {"pairs": len(fr), "m2_repro_maxabs": float(np.abs(
        ref.reindex((fr.q1.to_numpy(np.int64) << 32) | fr.qx.to_numpy(np.int64)).to_numpy() - fr.p_m2).max()),
        "corr_logit": float(np.corrcoef(np.log(p / (1 - p) + 1e-12), np.log(p0 / (1 - p0) + 1e-12))[0, 1])}
    fr.to_parquet(pred / "test_France.parquet", index=False)
    s3.put(pred / "test_France.parquet", "pred/test_France.parquet")
    for c in ("India", "US"):
        s3.put(pred / f"holdout_f_{c}.npy", f"pred/holdout_f_{c}.npy")
    log(f"France {res['france']}")
    return res


def score(train_dir, test_dir, model_dir, hp, s3):
    """Booster hp['booster'] (s3 uri; features per arms[0]) -> pred/holdout_<c>.parquet (q1, qx, p) on P22's holdout keys
    (test channel keys/holdout_<c>.parquet = #6's pruned holdout pairs), the input P25's #8c-yardstick eval reads."""
    import boto3
    import lightgbm as lgb
    arm = arms_of(hp)[0]
    hn, cp = bool(arm.get("hn")), arm.get("ctx") == "pruned"
    hni = [HN.index(c) for c in arm["hn_cols"]] if arm.get("hn_cols") else None
    bk, key = hp["booster"][5:].split("/", 1)
    f = model_dir / Path(key).name
    boto3.client("s3").download_file(bk, key, str(f))
    m = lgb.Booster(model_file=str(f))
    fs = lambda c: sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs("US")[0]).names if c not in DROP]
    pr = lgb.Booster(model_file=str(Path(test_dir) / "pruner.txt"))
    res, pred = {"booster": hp["booster"], "arm": arm}, model_dir / "pred"
    pred.mkdir(exist_ok=True)
    for c in ("India", "US"):
        C = Country(fs(c), dict(np.load(Path(test_dir) / f"meta/train_{c}.npz")), cols, hp["tiny"], pr)
        k = pd.read_parquet(Path(test_dir) / f"keys/holdout_{c}.parquet", columns=["q1", "qx"])
        want = (k.q1.to_numpy(np.int64) << 32) | k.qx.to_numpy(np.int64)
        rows = np.flatnonzero(np.isin((C.q1.astype(np.int64) << 32) | C.qx.astype(np.int64), want))
        assert len(rows) == len(k), f"holdout {c}: {len(rows)} of {len(k)} keys"
        p = m.predict(C.mat(rows, hn, cp, hni), num_threads=os.cpu_count())
        pd.DataFrame({"q1": C.q1[rows], "qx": C.qx[rows], "p": p.astype(np.float32)}).to_parquet(
            pred / f"holdout_{c}.parquet", index=False)
        s3.put(pred / f"holdout_{c}.parquet", f"pred/holdout_{c}.parquet")
        res[f"holdout_{c}"] = len(rows)
        log(f"scored holdout {c}: {len(rows):,}")
        del C
    return res


WOOD, LO, HI = [0.9245, 0.2595, 0.0515], 0.02, 0.98                  # P13's France (out-of-domain) blend


def _lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _blend(p, r, w):
    b = (p > LO) & (p < HI) & ~np.isnan(r)
    out = p.copy()
    out[b] = 1 / (1 + np.exp(-(w[0] * _lg(p[b]) + w[1] * _lg(r[b]) + w[2])))
    return out


def ceproxy(train_dir, test_dir, model_dir, hp, s3):
    """The France procedure with EVERYTHING out-of-domain: GBDT trained on the source only (LOCO a0 / a2 boosters from
    hp['tars'] = {src: model.tar.gz uri}), bge CE trained on the source only (P13 r_bge_loco<src>, holdout band of
    m2_big), source blended in-domain (weights fit on its fold-0 half-0 band pairs), target blended with P13's WOOD.
    t* = best F0.5 on source half 0; target t = P6's rule (empty rate = source's at t*), clipped [t*, t*+0.2] as in
    fr_diag.france_threshold, and unclipped. F0.5 on the target's fold-0 half 1 (and all fold 0), paired a2 vs a0."""
    import boto3, tarfile
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    fs = lambda c: sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs("US")[0]).names if c not in DROP]
    pr = lgb.Booster(model_file=str(Path(test_dir) / "pruner.txt"))
    D = {c: Country(fs(c), dict(np.load(Path(test_dir) / f"meta/train_{c}.npz")), cols, hp["tiny"], pr) for c in ("India", "US")}
    E = {c: Eval(D[c]) for c in D}
    cli = boto3.client("s3")
    M = {}
    for src, uri in hp["tars"].items():
        bk, key = uri[5:].split("/", 1)
        f = model_dir / f"{src}.tar.gz"
        cli.download_file(bk, key, str(f))
        d = model_dir / src
        with tarfile.open(f) as t:
            t.extractall(d)
        M[src] = {a: lgb.Booster(model_file=str(d / f"{a}.txt")) for a in ("a0", "a2")}
    R, Y, H1, F0 = {}, {}, {}, {}
    for c in D:
        C, rows = D[c], E[c].rows
        kk = (C.q1[rows].astype(np.int64) << 32) | C.qx[rows].astype(np.int64)
        for src in ("US", "India"):
            r = pd.read_parquet(Path(test_dir) / f"r/r_bge_loco{src}_{c}.parquet")
            R[c, src] = pd.Series(r.r.to_numpy(np.float64), index=(r.q1.to_numpy(np.int64) << 32) | r.qx.to_numpy(np.int64)
                                  ).reindex(kk).to_numpy()
        Y[c], H1[c], F0[c] = C.y[rows], hsh(C.q1[rows], 2), C.fold[rows] == 0
    res = {"r_coverage": {f"{c}_{s}": float(np.mean(~np.isnan(v))) for (c, s), v in R.items()}}
    for a, b in (("US", "India"), ("India", "US")):
        per = {}
        for arm in ("a0", "a2"):
            hn = arm == "a2"
            pa = M[a][arm].predict(D[a].mat(E[a].rows, hn), num_threads=os.cpu_count())
            pb = M[a][arm].predict(D[b].mat(E[b].rows, hn), num_threads=os.cpu_count())
            for mode in ("base", "blend"):
                info = {}
                if mode == "blend":
                    m = F0[a] & (H1[a] == 0) & (pa > LO) & (pa < HI) & ~np.isnan(R[a, a])
                    lr = LogisticRegression(C=1.0).fit(np.c_[_lg(pa[m]), _lg(R[a, a][m])], Y[a][m])
                    wa = [float(lr.coef_[0, 0]), float(lr.coef_[0, 1]), float(lr.intercept_[0])]
                    xa, xb = _blend(pa, R[a, a], wa), _blend(pb, R[b, a], WOOD)
                    band_b = (pb > LO) & (pb < HI)
                    info |= {"w_src": wa, "tgt_band_r_cov": float(np.mean(~np.isnan(R[b, a][band_b])))}
                else:
                    xa, xb = pa, pb
                ka, kb = E[a].per_s1(xa), E[b].per_s1(xb)
                ca = E[a].curve(ka, E[a].half == 0)
                ts = float(ca.t[ca.f05.idxmax()])
                ref = float(E[a].curve(ka).set_index("t").emp[ts])
                cb = E[b].curve(kb).set_index("t")
                tu = float((cb.emp - ref).abs().idxmin())
                tc = float(min(max(tu, ts), round(ts + 0.2, 2)))
                for tag, t in (("clip", tc), ("unclip", tu)):
                    f = E[b].at(kb, t)[0]
                    per[arm, mode, tag] = f
                    info |= {f"t_{tag}": t, f"f05_h1_{tag}": float(f[E[b].half == 1].mean()), f"f05_all_{tag}": float(f.mean())}
                info |= {"t_star": ts, "ref_empty": ref, "src_f05_h1": float(E[a].at(ka, ts)[0][E[a].half == 1].mean())}
                res[f"{a}>{b}_{arm}_{mode}"] = info
                log(f"{a}>{b} {arm} {mode}: {info}")
        h1 = E[b].half == 1
        for tag in ("clip", "unclip"):
            for x, y_, nm in ((("a2", "blend"), ("a0", "blend"), "a2-a0_blend"), (("a2", "base"), ("a0", "base"), "a2-a0_base"),
                              (("a0", "blend"), ("a0", "base"), "ce_on_a0"), (("a2", "blend"), ("a2", "base"), "ce_on_a2"),
                              (("a2", "blend"), ("a0", "base"), "a2ce-a0base")):
                res[f"{a}>{b}_{nm}_{tag}"] = _paired(per[x + (tag,)][h1], per[y_ + (tag,)][h1])
        s3.put_json(res | {"state": f"{a}>{b} done"}, "status.json")
    return res


def ceproxy2(train_dir, test_dir, model_dir, hp, s3):
    """Out-of-domain blend weights for the hn base (P13's `xw`): per direction a>b and arm, w_tgt = the blend fit on the
    target's fold-0 half-0 band pairs (labels of b: an oracle for that direction); France-style weights for b are then
    the OTHER direction's w_tgt (xfit) or the mean of both (mean, = how P13 got WOOD). Evaluated like ceproxy (P6 rule,
    clipped) on b's half 1, vs WOOD."""
    import boto3, tarfile
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    fs = lambda c: sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(fs("US")[0]).names if c not in DROP]
    pr = lgb.Booster(model_file=str(Path(test_dir) / "pruner.txt"))
    D = {c: Country(fs(c), dict(np.load(Path(test_dir) / f"meta/train_{c}.npz")), cols, hp["tiny"], pr) for c in ("India", "US")}
    E = {c: Eval(D[c]) for c in D}
    cli, M = boto3.client("s3"), {}
    for src, uri in hp["tars"].items():
        bk, key = uri[5:].split("/", 1)
        f = model_dir / f"{src}.tar.gz"
        cli.download_file(bk, key, str(f))
        with tarfile.open(f) as t:
            t.extractall(model_dir / src)
        M[src] = {a: lgb.Booster(model_file=str(model_dir / src / f"{a}.txt")) for a in ("a0", "a2")}
    R, Y, H1, F0 = {}, {}, {}, {}
    for c in D:
        C, rows = D[c], E[c].rows
        kk = (C.q1[rows].astype(np.int64) << 32) | C.qx[rows].astype(np.int64)
        for src in ("US", "India"):
            r = pd.read_parquet(Path(test_dir) / f"r/r_bge_loco{src}_{c}.parquet")
            R[c, src] = pd.Series(r.r.to_numpy(np.float64), index=(r.q1.to_numpy(np.int64) << 32) | r.qx.to_numpy(np.int64)
                                  ).reindex(kk).to_numpy()
        Y[c], H1[c], F0[c] = C.y[rows], hsh(C.q1[rows], 2), C.fold[rows] == 0
    fitw = lambda p, r, y, m: (lambda lr: [float(lr.coef_[0, 0]), float(lr.coef_[0, 1]), float(lr.intercept_[0])])(
        LogisticRegression(C=1.0).fit(np.c_[_lg(p[m]), _lg(r[m])], y[m]))
    P, WT, WS = {}, {}, {}
    dirs = (("US", "India"), ("India", "US"))
    for a, b in dirs:
        for arm in ("a0", "a2"):
            hn = arm == "a2"
            pa = M[a][arm].predict(D[a].mat(E[a].rows, hn), num_threads=os.cpu_count())
            pb = M[a][arm].predict(D[b].mat(E[b].rows, hn), num_threads=os.cpu_count())
            P[a, arm] = (pa, pb)
            band = lambda p: (p > LO) & (p < HI)
            WS[a, arm] = fitw(pa, R[a, a], Y[a], F0[a] & (H1[a] == 0) & band(pa) & ~np.isnan(R[a, a]))
            WT[a, arm] = fitw(pb, R[b, a], Y[b], F0[b] & (H1[b] == 0) & band(pb) & ~np.isnan(R[b, a]))
    res = {"w_tgt": {f"{a}>{'India' if a == 'US' else 'US'}_{arm}": w for (a, arm), w in WT.items()},
           "w_src": {f"{a}_{arm}": w for (a, arm), w in WS.items()}}
    log(f"weights {res}")
    for a, b in dirs:
        other = b                                               # the other direction's source is b
        per = {}
        for arm in ("a0", "a2"):
            pa, pb = P[a, arm]
            xa = _blend(pa, R[a, a], WS[a, arm])
            ka = E[a].per_s1(xa)
            ca = E[a].curve(ka, E[a].half == 0)
            ts = float(ca.t[ca.f05.idxmax()])
            ref = float(E[a].curve(ka).set_index("t").emp[ts])
            wm = list(np.mean([WT["US", arm], WT["India", arm]], axis=0))
            for wn, w in (("wood", WOOD), ("xfit", WT[other, arm]), ("mean", wm), ("oracle", WT[a, arm])):
                kb = E[b].per_s1(_blend(pb, R[b, a], w))
                cb = E[b].curve(kb).set_index("t")
                tu = float((cb.emp - ref).abs().idxmin())
                tc = float(min(max(tu, ts), round(ts + 0.2, 2)))
                f = E[b].at(kb, tc)[0]
                per[arm, wn] = f
                res[f"{a}>{b}_{arm}_{wn}"] = {"w": [round(x, 4) for x in w], "t": tc, "f05_h1": float(f[E[b].half == 1].mean())}
                log(f"{a}>{b} {arm} {wn}: {res[f'{a}>{b}_{arm}_{wn}']}")
        h1 = E[b].half == 1
        for arm in ("a0", "a2"):
            for wn in ("xfit", "mean", "oracle"):
                res[f"{a}>{b}_{arm}_{wn}-wood"] = _paired(per[arm, wn][h1], per[arm, "wood"][h1])
        for wn in ("wood", "xfit", "mean"):
            res[f"{a}>{b}_a2-a0_{wn}"] = _paired(per["a2", wn][h1], per["a0", wn][h1])
    return res


def main(train_dir, test_dir, model_dir, out_dir, params):
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    assert not {0, 1} & set(hp["folds"]), "never train on fold 0 (holdout) / 1 (early stopping)"
    s3, model_dir = S3(hp["s3"]), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"{os.cpu_count()} vCPU, hp {hp}")
    s3.put_json({"state": "started", "hp": hp}, "status.json")
    res = {"loco": loco, "full": full, "score": score, "ceproxy": ceproxy, "ceproxy2": ceproxy2}[hp["mode"]](train_dir, test_dir, model_dir, hp, s3)
    res |= {"hp": hp, "state": "done", "elapsed_s": round(time.time() - T0), "peak_gb": round(peak(), 2)}
    (model_dir / "result.json").write_text(json.dumps(res, default=float, indent=1))
    s3.put(model_dir / "result.json", "result.json")
    s3.put_json(res, "status.json")
    log(f"done {json.dumps(res, default=float)}")


if __name__ == "__main__":                        # self-check of the hn features (no data needed)
    M = {"h1": np.array([10, 10, 5402, 10, -1]), "l1": np.array([2, 2, 4, 2, 0], np.int8),
         "hx": np.array([13, 8, 402, 10, 3, 13]), "lx": np.array([2, 1, 3, 2, 1, 2], np.int8)}
    q1, qx = np.array([0, 1, 2, 3, 4, 0]), np.array([0, 1, 2, 3, 4, 5])
    H = hn_feats(q1, qx, np.array([.9, .8, .7, .6, .5, .4], np.float32), M)
    assert list(H[:5, 0]) == [1, 3, 2, 0, 0], H[:, 0]                   # = P15 hn_check's self-test
    assert H[0, 2] == 1 and H[5, 2] == 1 and np.isclose(H[0, 3], .4) and np.isclose(H[5, 3], .9)   # S1 0: two SX hn 13
    assert np.isnan(H[1, 2]) or H[1, 2] == 0
    assert H[3, 4] == 0 and np.isnan(H[4, 1])
    assert list(f05(np.array([2., 0, 0]), np.array([3., 0, 1]), np.array([2., 0, 0]))) == [1.25 * (2 / 3) / (0.25 * 2 / 3 + 1), 1.0, 0.0]
    # pruned_ctx = features._group_ctx on the kept blocked pairs (its self-test: sx_n, s1_gap, margins)
    q1, qx = np.array([0, 0, 1, 1, 2, 0]), np.array([0, 1, 1, 2, 3, 3])
    bs = np.array([.9, .8, .7, .6, .5, .95], np.float32)
    CX = pruned_ctx(q1, qx, bs, np.array([1, 1, 1, 1, 1, 0], bool), np.array([1, 1, 1, 1, 1, 1], bool))
    assert list(CX[:5, 0]) == [2, 2, 2, 2, 1] and np.isnan(CX[5]).all()                 # s1_n; unblocked -> NaN
    assert np.isclose(CX[0, 2], .1) and np.isclose(CX[1, 1], .1) and CX[4, 2] == 1.0     # s1 margin / gap / lone
    assert list(CX[:5, 3]) == [1, 2, 2, 1, 1] and CX[1, 5] > 0 > CX[2, 5]               # sx_n; sx_margin signs
    assert list(CX[:5, 6]) == [0, 1, 0, 1, 0] and list(CX[:5, 7]) == [0, 0, 1, 0, 0]     # block ranks
    CX = pruned_ctx(q1, qx, bs, np.ones(6, bool), np.array([1, 0, 1, 1, 1, 1], bool))   # pair 1 pruned
    assert np.isnan(CX[1]).all() and CX[0, 0] == 2 and CX[2, 3] == 1 and CX[2, 5] == 1.0
    print("ok")
