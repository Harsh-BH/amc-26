"""P27 SageMaker job (ONE CPU run): LOCO go/no-go for pseudo-label self-training, both directions.

Shipped alone as solution.py next to sm_train.py; sm_train calls main(train_dir, test_dir, model_dir, out_dir, params).
Channel 'train' = P16's copy of data/work/p4/X/train_<c>_sx_NNN.parquet (65 features, q1, qx, is_true, fold).
Channel 'test'  = p27.py prep's aux: tn_<c>.parquet (fold-0 S1 -> true count), hn_<c>_{s1,sx}.npy, model_<c>.txt
(P6's source-only LOCO boosters: fold 2 x0.5, lr 0.1, early-stopped on the source's fold 1 x0.2).

Protocol = P6 loco_dense.py: a source model scores the unseen target; every train S1 competes (argmax per SX over all
S1), macro F0.5 over EVERY fold-0 S1 of a country (true pairs missing from the candidates are recall losses).
Pseudo-label pool = the TARGET's fold 2 x0.5 (P6). Retrained arms = source fold 2 x0.5 (true labels) + the pool's
pseudo-labels, early-stopped on the source's fold 1 x0.2, P6's LightGBM params. Thresholds reported per arm:
oracle (best t on the target's labels), the source's own t*, and the label-free empty-rate match (P6/P13: target t
whose empty-S1 rate equals the source's at its t*).
"""
import json, os, resource, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()
ID = ["q1", "qx", "is_true", "fold"]
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)        # P4 M.PARAMS + P6 lr
RECAL = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, lambda_l2=1.0, verbose=-1,
             seed=0, deterministic=True)                                                            # P17's stacker
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])
HNF = {"hn_eq", "hn_both", "hn_diff", "addr_rule", "rule_acc"}      # structured house-number evidence
TS = np.r_[np.round(np.arange(0.02, 0.99, 0.01), 2), 0.985, 0.99, 0.995, 0.998, 0.999]
P6GRID = TS <= 0.98 + 1e-9
DEFAULTS = {"src": "", "tgt": "", "directions": [["US", "India"], ["India", "US"]], "s3": "", "deadline_s": 9900,
            "rounds": 1500, "frac": 0.5, "tiny": False,
            "arms": ["base", "plain", "plain_cls0", "filt", "filt_w03", "neg_only", "neg_hard", "sibneg", "recal"],
            "hi": 0.995, "lo": 0.005, "margin": 0.5, "w_ps": 0.3, "base_model": ""}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def log(msg: str):
    print(f"[p27 {time.time() - T0:7.0f}s peak {peak():5.1f} GB] {msg}", flush=True)


class S3:
    def __init__(self, uri: str):
        self.on = bool(uri)
        if self.on:
            import boto3
            self.bucket, self.prefix = uri[5:].rstrip("/").split("/", 1)
            self.c = boto3.client("s3")

    def put_json(self, obj, key: str):
        if self.on:
            try:
                self.c.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=json.dumps(obj).encode())
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")

    def put(self, path: Path, key: str):
        if self.on:
            try:
                self.c.upload_file(str(path), self.bucket, f"{self.prefix}/{key}")
            except Exception as e:
                log(f"s3 {key} failed: {e!r}")


def hsel(q1: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """P4 train.load's S1-hash sample."""
    return (q1.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1000) < frac * 1000


def load_country(train_dir: Path, c: str) -> dict:
    fs = sorted(Path(train_dir).glob(f"p4/X/train_{c}_sx_*.parquet"))
    assert fs, f"no train X for {c}"
    cols = [k for k in pq.read_schema(fs[0]).names if k not in DROP]
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in fs)
    X = np.empty((n, len(cols)), np.float32)
    q1, qx = np.empty(n, np.int64), np.empty(n, np.int64)
    fold, y = np.empty(n, np.int8), np.empty(n, bool)
    a = 0
    for p in fs:
        t = pq.read_table(p, columns=ID + cols)
        m = len(t)
        for j, k in enumerate(cols):
            X[a:a + m, j] = t.column(k).to_numpy().astype(np.float32)
        q1[a:a + m], qx[a:a + m] = t.column("q1").to_numpy(), t.column("qx").to_numpy()
        fold[a:a + m], y[a:a + m] = t.column("fold").to_numpy(), t.column("is_true").to_numpy()
        a += m
        del t
    assert a == n
    return dict(X=X, q1=q1, qx=qx, fold=fold, y=y, cols=cols, n=n)


def sx_stats(qx: np.ndarray, p: np.ndarray) -> tuple:
    """(argmax mask per SX over every S1 -- ties -> first, as P4 argmax_sx --, best p of the OTHER candidates)."""
    o = np.lexsort((-p, qx))
    qs = qx[o]
    first = np.r_[True, qs[1:] != qs[:-1]]
    am = np.zeros(len(p), bool)
    am[o[first]] = True
    grp = np.cumsum(first) - 1
    best = p[o[first]]
    sec_mask = np.r_[False, first[:-1]] & ~first
    second = np.zeros(len(best), np.float32)
    second[grp[sec_mask]] = p[o][sec_mask]
    other = np.empty(len(p), np.float32)
    g = np.empty(len(p), np.int64)
    g[o] = grp
    other[:] = best[g]
    other[am] = second[g[am]]
    nc = np.bincount(g)[g]
    return am, other, nc


class Country:
    def __init__(self, name: str, d: dict, aux: Path):
        self.name, self.d = name, d
        tn = __import__("pandas").read_parquet(aux / f"tn_{name}.parquet")
        self.tq, self.ntrue = tn.q1.to_numpy(np.int64), tn.n_true.to_numpy(float)
        assert np.all(np.diff(self.tq) > 0)
        qx, fold = d["qx"], d["fold"]
        self.f0x = np.isin(qx, np.unique(qx[fold == 0]))          # pairs of SX with a fold-0 candidate (P6)
        h1, hx = np.load(aux / f"hn_{name}_s1.npy"), np.load(aux / f"hn_{name}_sx.npy")
        a, b = h1[d["q1"]], hx[qx]
        has = (a >= 0) & (b >= 0)
        self.cls0 = ~has | (a == b)                                  # hn none / equal (P15 class 0)
        self.sib3 = has & np.isin(b - a, SIB3)                        # generator sibling offsets (P15)
        del h1, hx, a, b

    def predict_all(self, m, rows=None) -> np.ndarray:
        rows = np.flatnonzero(self.f0x) if rows is None else rows
        p = np.zeros(self.d["n"], np.float32)
        p[rows] = m.predict(self.d["X"][rows], num_threads=os.cpu_count()).astype(np.float32)
        return p

    def curve(self, p: np.ndarray) -> dict:
        d = self.d
        am = sx_stats(d["qx"], p)[0]
        sel = am & (d["fold"] == 0)
        k = np.searchsorted(self.tq, d["q1"][sel])
        assert np.all(self.tq[k] == d["q1"][sel])
        pk, yk = p[sel], d["y"][sel]
        n = len(self.tq)
        rows = []
        for t in TS:
            m = pk >= t
            npred = np.bincount(k[m], minlength=n).astype(float)
            tp = np.bincount(k[m & yk], minlength=n).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                P, R = tp / npred, tp / self.ntrue
                f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
            f = np.where(self.ntrue == 0, (npred == 0).astype(float), f)
            rows.append((float(f.mean()), float((npred == 0).mean()), float(npred.sum() / n),
                         float(tp.sum() / max(npred.sum(), 1)), float(tp.sum() / self.ntrue.sum())))
        a = np.array(rows)
        return {"f05": a[:, 0], "empty": a[:, 1], "mps": a[:, 2], "prec": a[:, 3], "rec": a[:, 4]}


def summary(cs: dict, ct: dict) -> dict:
    """Source curve cs (in-country), target curve ct (unseen) -> the threshold policies."""
    i_s = int(np.argmax(cs["f05"]))
    i_o, i_o6 = int(np.argmax(ct["f05"])), int(np.argmax(np.where(P6GRID, ct["f05"], -1)))
    i_e = int(np.argmin(np.abs(ct["empty"] - cs["empty"][i_s])))
    i_m = int(np.argmin(np.abs(ct["mps"] - cs["mps"][i_s])))
    r = lambda x: round(float(x), 5)
    return {"src_f05": r(cs["f05"][i_s]), "src_t": r(TS[i_s]), "src_empty": r(cs["empty"][i_s]),
            "tgt_oracle": r(ct["f05"][i_o]), "tgt_oracle_t": r(TS[i_o]), "tgt_oracle_p6grid": r(ct["f05"][i_o6]),
            "tgt_at_src_t": r(ct["f05"][i_s]), "tgt_lf_empty": r(ct["f05"][i_e]), "tgt_lf_empty_t": r(TS[i_e]),
            "tgt_lf_mps": r(ct["f05"][i_m]), "tgt_lf_mps_t": r(TS[i_m]), "tgt_empty_at_lf": r(ct["empty"][i_e]),
            "tgt_prec_at_lf": r(ct["prec"][i_e]), "tgt_rec_at_lf": r(ct["rec"][i_e])}


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    hp = DEFAULTS | {k: v for k, v in params.items() if k in DEFAULTS}
    if hp["src"]:
        hp["directions"] = [[hp["src"], hp["tgt"]]]
    aux, model_dir = Path(test_dir), Path(model_dir)
    s3 = S3(hp["s3"])
    P = dict(PARAMS, num_threads=os.cpu_count())
    log(f"lightgbm {lgb.__version__}, {os.cpu_count()} vCPU, hp {hp}")
    C = {}
    for c in ("India", "US"):
        C[c] = Country(c, load_country(train_dir, c), aux)
        log(f"{c}: {C[c].d['n']:,} pairs, f0x {C[c].f0x.mean():.3f}, fold-0 S1 {len(C[c].tq):,}, "
            f"sib3 {C[c].sib3.mean():.4f}")
    cols = C["India"].d["cols"]
    assert cols == C["US"].d["cols"]
    res = {"hp": hp, "ts": TS.tolist(), "dirs": {}}
    status = {"state": "loaded", "elapsed_s": round(time.time() - T0)}
    s3.put_json(status, "status.json")

    for src, tgt in hp["directions"]:
        key = f"{src}>{tgt}"
        S, T = C[src], C[tgt]
        R = res["dirs"][key] = {"arms": {}, "pool": {}}
        base = lgb.Booster(model_file=str(aux / f"model_{hp['base_model'] or src}.txt"))   # 'pooled': P6's US+India
        assert base.feature_name() == cols, "P6 model feature order differs"
        # base on EVERY target pair (pool labels need all competitors), and on the source's fold-0 SX
        pT = T.predict_all(base, np.arange(T.d["n"]))
        pS = S.predict_all(base)
        cur_s0 = S.curve(pS)
        amT, otherT, ncT = sx_stats(T.d["qx"], pT)
        pool = np.flatnonzero((T.d["fold"] == 2) & hsel(T.d["q1"], hp["frac"]))
        pp, yp, am, oth = pT[pool], T.d["y"][pool], amT[pool], otherT[pool]
        # P6's argmax: within the pool rows only
        am_pool = sx_stats(T.d["qx"][pool], pp)[0]
        margin = np.where(am, pp - oth, oth - pp)       # am: gap to the 2nd candidate; loser: gap to the winner
        cls0, sib3 = T.cls0[pool], T.sib3[pool]
        Xp = T.d["X"][pool]
        ci = {k: i for i, k in enumerate(cols)}
        look = (Xp[:, ci["core_tset"]] >= 80) | (Xp[:, ci["name_cos"]] >= 0.6)
        loser = ~am & (oth >= 0.98) & (pp < 0.5)
        # pool diagnostics with the full candidate lists (precision uses the target's labels: reporting only)
        conf = am & (pp >= 0.98)
        dg = {"n_pool": int(len(pool)), "true_share": float(yp.mean())}
        for lo_, hi_ in ((0, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
            m = conf & (margin >= lo_) & (margin < hi_)
            dg[f"pos98_margin_{lo_}-{hi_}"] = [int(m.sum()), float(yp[m].mean()) if m.any() else None]
        for nm, m in (("pos98", conf), ("pos98_cls0", conf & cls0), ("pos98_sib3", conf & sib3),
                      ("pos995_cls0_m05", am & (pp >= 0.995) & cls0 & (margin >= 0.5)),
                      ("pos999_cls0_m05", am & (pp >= 0.999) & cls0 & (margin >= 0.5)),
                      ("neg_p005", pp <= 0.005), ("neg_p02", pp <= 0.02), ("loser", loser),
                      ("neg_hard", (pp <= 0.005) & look), ("sibneg98", conf & sib3)):
            dg[nm] = [int(m.sum()), float(yp[m].mean()) if m.any() else None]
        R["pool"] = dg
        log(f"{key}: pool {len(pool):,} pairs; diag {dg}")
        src_tr = np.flatnonzero((S.d["fold"] == 2) & hsel(S.d["q1"], hp["frac"]))
        src_va = np.flatnonzero((S.d["fold"] == 1) & hsel(S.d["q1"], 0.2))
        cur_base_t = T.curve(pT)                     # = f0x-only scoring: an SX with a fold-0 pair has all its pairs in f0x

        def labels(arm: str):
            """(pool row mask, pseudo-label, weight) of a retrained arm."""
            hi, lo, mg, w = hp["hi"], hp["lo"], hp["margin"], hp["w_ps"]
            arm = arm.split("_s")[0] if "_s" in arm and arm.rsplit("_s", 1)[1].isdigit() else arm
            pos = am & (pp >= hi) & cls0 & (margin >= mg)
            neg = pp <= lo
            if arm == "plain":
                pos, neg, w = am_pool & (pp >= 0.98), pp <= 0.02, 1.0
            elif arm == "src":                             # source-only refit (seed-variance control)
                pos, neg = np.zeros_like(pos), np.zeros_like(pos)
            elif arm == "soft":                            # every pool pair, sharpened soft label (T = 0.5)
                return np.ones(len(pp), bool), 1 / (1 + np.exp(-lg(pp) / 0.5)), 1.0
            elif arm == "plain_cls0":                      # P6's thresholds, house-number-clean positives only
                pos, neg, w = am_pool & (pp >= 0.98) & cls0, pp <= 0.02, 1.0
            elif arm == "filt":
                w = 1.0
            elif arm == "filt_w03":
                pass
            elif arm == "neg_only":
                pos, w = np.zeros_like(pos), 1.0
            elif arm == "neg_hard":
                pos, neg, w = np.zeros_like(pos), (neg & look) | loser, 1.0
            elif arm == "sibneg":
                neg, w = neg | (conf & sib3), 1.0
            else:
                raise ValueError(arm)
            neg = neg & ~pos
            return pos | neg, pos.astype(np.float32), w

        for arm in hp["arms"]:
            if time.time() - T0 > hp["deadline_s"]:
                log(f"deadline: skipping {key} {arm}")
                R["arms"][arm] = {"skipped": "deadline"}
                continue
            t1 = time.time()
            info = {}
            if arm == "base":
                cs, ct = cur_s0, cur_base_t
            elif arm == "recal":
                # target-only residual GBDT on logit p_base, fit on the pool's pseudo-labels (filt pos + neg + losers)
                keep, pos, _ = labels("filt")
                keep = keep | loser
                fx = lambda rows, p_, oth_, am_, nc_: np.column_stack([
                    lg(p_), np.where(am_, p_ - oth_, oth_ - p_), nc_, T.sib3[rows], T.cls0[rows],
                    T.d["X"][rows][:, [ci[k] for k in ("hn_eq", "hn_both", "chain_1", "chain_x", "sx_margin",
                                                        "s1_margin", "is_s3", "addr_cos", "name_cos")]]]).astype(np.float32)
                Xr = fx(pool[keep], pp[keep], oth[keep], am[keep], ncT[pool][keep])
                yr = pos[keep]
                ds = lgb.Dataset(Xr, yr, init_score=lg(pp[keep]), free_raw_data=True)
                m = lgb.train(dict(RECAL, num_threads=os.cpu_count()), ds, 200)
                rows = np.flatnonzero(T.f0x)
                pb = pT[rows]
                p2 = np.zeros(T.d["n"], np.float32)
                p2[rows] = 1 / (1 + np.exp(-(lg(pb) + m.predict(fx(rows, pb, otherT[rows], amT[rows], ncT[rows]),
                                                                   raw_score=True))))
                cs, ct = cur_s0, T.curve(p2)
                pb_ = pos >= 0.5
                info = {"n_pos": int((keep & pb_).sum()), "n_neg": int((keep & ~pb_).sum()),
                        "prec_pos": float(yp[keep & pb_].mean()), "prec_neg": float(1 - yp[keep & ~pb_].mean())}
            elif arm.startswith("tgtonly"):
                # a target-only model on the pool's pseudo-labels, logit-blended with the base (France-only use).
                # _plain: P6 plain labels (else plain_cls0); _nohn: no structured house-number features
                keep, pos, _ = labels("plain" if "_plain" in arm else "plain_cls0")
                fi = [i for i, k in enumerate(cols) if not ("_nohn" in arm and k in HNF)]
                fc = [cols[i] for i in fi]
                ds = lgb.Dataset(Xp[keep][:, fi], pos[keep], feature_name=fc, free_raw_data=True)
                dv = lgb.Dataset(S.d["X"][src_va][:, fi], S.d["y"][src_va].astype(np.float32), reference=ds)
                m = lgb.train(P, ds, hp["rounds"], valid_sets=[dv],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
                rows = np.flatnonzero(T.f0x)
                pt = m.predict(T.d["X"][rows][:, fi], num_threads=os.cpu_count())
                for w in (0.2, 0.35, 0.5):
                    p2 = np.zeros(T.d["n"], np.float32)
                    p2[rows] = 1 / (1 + np.exp(-((1 - w) * lg(pT[rows]) + w * lg(pt))))
                    sm = summary(cur_s0, T.curve(p2))
                    R["arms"][f"{arm}_w{w}"] = sm | {"trees": m.num_trees(), "w": w, "n_feats": len(fi),
                                                     "curve_tgt": {}, "curve_src": {}}
                    log(f"{key} {arm}_w{w}: {sm}")
                continue
            else:
                keep, pos, w = labels(arm)
                n_s = len(src_tr)
                Xtr = np.concatenate([S.d["X"][src_tr], Xp[keep]])
                ytr = np.r_[S.d["y"][src_tr].astype(np.float32), pos[keep]].astype(np.float32)
                wtr = np.r_[np.ones(n_s, np.float32), np.full(int(keep.sum()), w, np.float32)]
                ds = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=cols, free_raw_data=True)
                dv = lgb.Dataset(S.d["X"][src_va], S.d["y"][src_va].astype(np.float32), reference=ds)
                sd = arm.rsplit("_s", 1)[1] if "_s" in arm and arm.rsplit("_s", 1)[1].isdigit() else None
                Pa = P | ({"seed": int(sd)} if sd else {}) | ({"objective": "cross_entropy"} if arm == "soft" else {})
                m = lgb.train(Pa, ds, hp["rounds"], valid_sets=[dv],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
                del Xtr, ds, dv
                cs, ct = S.curve(S.predict_all(m)), T.curve(T.predict_all(m))
                pb_ = pos >= 0.5
                info = {"n_src": n_s, "n_pos": int((keep & pb_).sum()), "n_neg": int((keep & ~pb_).sum()),
                        "prec_pos": float(yp[keep & pb_].mean()) if (keep & pb_).any() else None,
                        "prec_neg": float(1 - yp[keep & ~pb_].mean()) if (keep & ~pb_).any() else None,
                        "w": w, "trees": m.num_trees(), "seed": sd}
                m.save_model(str(model_dir / f"m_{src}_{tgt}_{arm}.txt"))
            sm = summary(cs, ct)
            R["arms"][arm] = sm | info | {"runtime_s": round(time.time() - t1, 1),
                                          "curve_tgt": {k: np.round(v, 5).tolist() for k, v in ct.items()},
                                          "curve_src": {k: np.round(v, 5).tolist() for k, v in cs.items()}}
            log(f"{key} {arm}: {sm} {info}")
            s3.put_json({"state": f"{key} {arm} done", "elapsed_s": round(time.time() - T0)}, "status.json")
            (model_dir / "result.json").write_text(json.dumps(res))
            s3.put(model_dir / "result.json", "result.json")
        del pT, pS
    res["elapsed_s"], res["peak_gb"] = round(time.time() - T0), round(peak(), 2)
    (model_dir / "result.json").write_text(json.dumps(res))
    s3.put(model_dir / "result.json", "result.json")
    s3.put_json({"state": "done", "elapsed_s": res["elapsed_s"]}, "status.json")
    log("done")
