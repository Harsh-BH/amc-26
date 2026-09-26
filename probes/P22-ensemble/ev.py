"""P22 evaluation on the #8c yardstick (run.py --split holdout --recipe sub8c, CE_MISSING=fallback), numpy-only.

Every variant scores the SAME rows: #6's pruned holdout pairs (the m2_big p9 cache, every competing train S1 included).
Only M2's p changes. Then, as run.model_parts does for #8c:
  band 0.02 < p < 0.98 -> bge r (p9 cache 3bd0b930 + our extra scores) blended with w; band pairs without a bge r
  take #7's e5 blend (cache bb0c29f0 + extra, W7) -- the holdout proxy (P13 scored only the holdout S1's band)
  -> US: P17's residual stacker (shipped booster, or refit on half 0) -> argmax S1 per SX, p >= t.
F0.5 per scored S1 (P16's holdout_frame / decide / per_s1), half 1 = P5's report half, paired vs the control.

    ev.py frame India|US       # build data/work/p22/frame_<c>.parquet (rows, r_bge, r_e5, house numbers, truth)
    ev.py repro                # control = #8c as shipped: must equal the pipe row (0.9899 / 0.9902 / 0.9896)
"""
import gc, json, os, resource, subprocess, sys, tempfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "code/business_entity_resolution/src"
sys.path[:0] = [str(HERE), str(SRC), str(ROOT), str(ROOT / "probes/P16-aws-lgbm")]
WORK = ROOT / "data/work"
OUT = WORK / "p22"
M2 = "d2b6569f_6a23c2ec_0.03"
BGE, E5 = "3bd0b930", "bb0c29f0"                    # M.tag of p13/ckpt/ce_bge_pool.pt, p11/ce_s2P.pt
W8 = [0.5901490903244407, 0.6319357246516834, 0.29732167912424273]   # run.RECIPES sub8c w (P13, full precision)
W7 = [0.617, 0.666, 0.261]                           # run.FALLBACK w (#7's e5 CE)
T8 = 0.71
STACK = WORK / "p17/stack_sup-bge-l21.0_US.txt"
LO, HI = 0.02, 0.98
C = ("India", "US")
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def lookup(files: list, k: np.ndarray) -> np.ndarray:
    """r of pair keys k from (k, r) caches (the first file that holds a key wins); NaN = unscored."""
    r = np.full(len(k), np.nan, np.float32)
    for f in files:
        if not Path(f).exists():
            continue
        t = pd.read_parquet(f)
        s = pd.Series(t.r.to_numpy(np.float32), index=t.k.to_numpy(np.int64))
        s = s[~s.index.duplicated()]
        m = np.isnan(r)
        r[m] = s.reindex(k[m]).to_numpy(np.float32)
        del t, s
    return r


def hn(side: str, c: str, ids: pa.Array, q: np.ndarray) -> np.ndarray:
    """stack.hn_of in arrow memory (the pandas version holds the whole rules.feats id column as Python objects)."""
    t = pq.read_table(WORK / f"rf_train_{side}_{c}.parquet", columns=["id", "hn"])
    pos = pc.index_in(ids, value_set=t.column("id"))
    assert pos.null_count == 0
    h = pc.fill_null(t.column("hn"), "").combine_chunks().take(pos)
    return h.take(pa.array(q)).to_numpy(zero_copy_only=False).astype(object)


def extra(ce: str, c: str) -> Path:
    return OUT / f"r_extra_{ce}_{c}.parquet"


def frame(c: str) -> Path:
    """Rows of the variant-independent frame: q1, qx (m2_big cache order), p_lgb, r_bge, r_e5, hn1, hnx, y, pos."""
    import features as F
    import p16
    t0 = time.time()
    d = pq.read_table(WORK / "p9" / f"m2_holdout_{c}_{M2}.parquet")
    q1, qx = d.column("q1").to_numpy(), d.column("qx").to_numpy()
    k = (q1.astype(np.int64) << 32) | qx.astype(np.int64)
    h = p16.holdout_frame(c)
    id1, idx = F.ids("train", c)
    out = pa.table({"q1": q1, "qx": qx, "p_lgb": d.column("p").cast(pa.float32()),
                    "y": np.isin(k, h["tkeys"]), "pos": h["pos"][q1].astype(np.int32),
                    "hn1": hn("s1", c, id1, q1), "hnx": hn("sx", c, idx, qx)})
    pq.write_table(out, OUT / f"frame_{c}.parquet")
    np.savez(OUT / f"s1_{c}.npz", ntrue=h["ntrue"], half=h["half"])
    print(f"{c}: frame {out.num_rows:,} rows, {int(out.column('y').to_numpy().sum()):,} true, "
          f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    return OUT / f"frame_{c}.parquet"


def load(c: str) -> dict:
    t = pq.read_table(OUT / f"frame_{c}.parquet")
    d = {n: t.column(n).to_numpy(zero_copy_only=False) for n in t.column_names}
    d["k"] = (d["q1"].astype(np.int64) << 32) | d["qx"].astype(np.int64)
    s = np.load(OUT / f"s1_{c}.npz")
    d["h"] = {"pos": None, "ntrue": s["ntrue"], "half": s["half"], "n": len(s["ntrue"])}
    d["r_bge"] = lookup([WORK / "p9" / f"r_holdout_{c}_{BGE}.parquet", extra("bge", c)], d["k"])
    d["r_e5"] = lookup([WORK / "p9" / f"r_holdout_{c}_{E5}.parquet", extra("e5", c)], d["k"])
    return d


def band(p):
    return (p > LO) & (p < HI)


def fit_w(ds: dict, P: dict) -> list:
    """P13's w refit (run.refit_w): logistic stack on the band pairs with a bge r of the scored half-0 S1."""
    import rerank as RR
    xs = []
    for c, d in ds.items():
        p = P[c]
        m = band(p) & ~np.isnan(d["r_bge"]) & (d["pos"] >= 0)
        m[m] &= d["h"]["half"][d["pos"][m]] == 0
        xs.append((p[m], d["r_bge"][m], d["y"][m]))
    return RR.fit_blend(*(np.concatenate(z) for z in zip(*xs)))


def blended(d: dict, p: np.ndarray, w: list) -> np.ndarray:
    """run.rerank with CE_MISSING=fallback: bge blend where the band pair has a bge r, else #7's e5 blend."""
    import rerank as RR
    b = band(p)
    rb = np.where(b, d["r_bge"], np.nan).astype(np.float32)
    out = RR.blend(p, rb, w, LO, HI)
    f = b & np.isnan(d["r_bge"])
    out[f] = RR.blend(p[f], d["r_e5"][f], W7, LO, HI)
    return out


def stacked(d: dict, p: np.ndarray, booster=None) -> np.ndarray:
    import lightgbm as lgb
    import stack as S
    m = booster or lgb.Booster(model_file=str(STACK))
    return S.stacked(m, S.X_of(d["q1"], p, d["hn1"], d["hnx"]))


def refit_stack(d: dict, p: np.ndarray, tag: str):
    """P17's stacker refit on the variant's blended p: US holdout S1 of half 0 (stack.train, same recipe)."""
    import stack as S
    X = S.X_of(d["q1"], p, d["hn1"], d["hnx"])
    tr = d["pos"] >= 0
    tr[tr] &= d["h"]["half"][d["pos"][tr]] == 0
    return S.train(X, d["y"], tr, OUT / f"stack_{tag}_US.txt")


def f05(d: dict, p: np.ndarray, ts) -> np.ndarray | dict:
    """Per scored S1 F0.5 (singletons included) under argmax S1 per SX, p >= t (p16.decide / per_s1). The argmax is
    t-independent, so it is computed once; ts = one t (-> array) or a list (-> {t: array})."""
    import model as M
    import train_matcher as TM
    am = M.argmax_sx(d["qx"], p)
    ps = d["pos"][am]
    m = ps >= 0
    ps, y, pa_ = ps[m], d["y"][am][m], p[am][m]
    n = d["h"]["n"]
    one = lambda t: TM.f05_per_s1(np.bincount(ps[(pa_ >= t) & y], minlength=n), np.bincount(ps[pa_ >= t], minlength=n),
                                  d["h"]["ntrue"])
    return one(ts) if np.isscalar(ts) else {t: one(t) for t in ts}


CKPT = {"bge": WORK / "p13/ckpt/ce_bge_pool.pt", "e5": WORK / "p11/ce_s2P.pt"}


def ce_score(ce: str, c: str, q1: np.ndarray, qx: np.ndarray) -> np.ndarray:
    """The recipe's CE on pairs (q1, qx) of train country c: rerank.py score in its own process under the GPU lock."""
    import rerank as RR
    ta, tb = RR.texts("train", c, q1, qx)
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        pd.DataFrame({"ta": ta, "tb": tb}).to_parquet(f"{d}/in.parquet", index=False)
        del ta, tb
        subprocess.run(["flock", "/tmp/amc-gpu.lock", sys.executable, str(SRC / "rerank.py"), "score", f"{d}/in.parquet",
                        f"{d}/out.parquet", str(CKPT[ce]), ce], check=True)
        return pd.read_parquet(f"{d}/out.parquet").r.to_numpy(np.float32)


def add_extra(ce: str, c: str, k: np.ndarray):
    """Score pair keys k with the CE and append them to our private cache (never to data/work/p9)."""
    k = np.unique(k)
    if not len(k):
        return
    r = ce_score(ce, c, (k >> 32).astype(np.int32), (k & 0xFFFFFFFF).astype(np.int32))
    f = extra(ce, c)
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame({"k": np.zeros(0, np.int64), "r": np.zeros(0, np.float32)})
    pd.concat([old, pd.DataFrame({"k": k, "r": r})], ignore_index=True).to_parquet(f, index=False)
    print(f"{c}: {len(k):,} pairs scored with {ce} -> {f.name}", flush=True)


def ce_check(n: int = 2000):
    """Rescore n cached bge / e5 US pairs: the scorer path must reproduce the cached r (fp16 noise only)."""
    out = {}
    for ce, tag in (("bge", BGE), ("e5", E5)):
        t = pd.read_parquet(WORK / "p9" / f"r_holdout_US_{tag}.parquet").sample(n, random_state=0)
        t0 = time.time()
        r = ce_score(ce, "US", (t.k.to_numpy() >> 32).astype(np.int32), (t.k.to_numpy() & 0xFFFFFFFF).astype(np.int32))
        out[ce] = {"max_abs_diff": float(np.abs(r - t.r.to_numpy()).max()), "pairs_s": n / (time.time() - t0)}
    print(out)
    return out

PRED = {"xgb": OUT / "xgb/pred/holdout_{c}.parquet", "cat": OUT / "cat/pred/holdout_{c}.parquet",
        "p16": WORK / "p9/m2_holdout_{c}_0726cb9a_6a23c2ec_0.03.parquet"}     # p16 = the dry-run stand-in
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
sg = lambda z: (1 / (1 + np.exp(-z))).astype(np.float32)


def preds(d: dict, c: str, models) -> dict:
    """{model: p aligned to the frame rows} (every frame key must be scored)."""
    out = {"lgb": d["p_lgb"]}
    for m in models:
        t = pd.read_parquet(str(PRED[m]).format(c=c))
        s = pd.Series(t.p.to_numpy(np.float32), index=(t.q1.to_numpy(np.int64) << 32) | t.qx.to_numpy(np.int64))
        out[m] = s.reindex(d["k"]).to_numpy(np.float32)
        assert not np.isnan(out[m]).any(), f"{m} {c}: frame keys without a p"
    return out


def half0_rows(d: dict) -> np.ndarray:
    m = d["pos"] >= 0
    m[m] &= d["h"]["half"][d["pos"][m]] == 0
    return m


def variants(ds: dict, P: dict, models) -> dict:
    """M2 p variants per country from P[c][model]: lr (logistic stack in logit space, fit on the half-0 holdout S1's
    kept pairs, both countries pooled), avg (equal-weight logit mean, nothing fit), rank (mean rank mapped onto
    m2_big's p quantiles per country: same marginal as m2_big), alone (the new model by itself)."""
    from sklearn.linear_model import LogisticRegression
    out, info = {}, {}
    sets = [(m,) for m in models] + ([tuple(models)] if len(models) > 1 else [])
    for s in sets:
        ms = ("lgb",) + s
        tag = "+".join(s)
        Z = {c: np.column_stack([lg(P[c][m]) for m in ms]) for c in ds}
        tr = {c: half0_rows(ds[c]) for c in ds}
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate([Z[c][tr[c]] for c in ds]),
                                                          np.concatenate([ds[c]["y"][tr[c]] for c in ds]))
        info[f"lr_{tag}"] = [round(float(x), 4) for x in lr.coef_[0]] + [round(float(lr.intercept_[0]), 4)]
        out[f"lr_{tag}"] = {c: sg(Z[c] @ lr.coef_[0] + lr.intercept_[0]) for c in ds}
        out[f"avg_{tag}"] = {c: sg(Z[c].mean(1)) for c in ds}
        rk = {}
        for c in ds:
            n = len(P[c]["lgb"])
            u = np.mean([pd.Series(P[c][m]).rank(method="average").to_numpy() for m in ms], axis=0)   # 1..n
            srt = np.sort(P[c]["lgb"])
            rk[c] = np.interp(u, np.arange(1, n + 1), srt).astype(np.float32)
        out[f"rank_{tag}"] = rk
    for m in models:
        out[f"alone_{m}"] = {c: P[c][m] for c in ds}
    return out, info


def run_variant(ds: dict, P: dict, tag: str, ts=np.round(np.arange(0.50, 0.951, 0.01), 2), pre=None) -> dict:
    """One M2 p variant through #8c's rest: w refit (half 0), bge/e5 blend, US stacker (shipped | refit on half 0),
    t 0.71 and t swept on half 0. -> per-S1 F0.5 arrays (India then US) per setting + the fitted w / t."""
    import lightgbm as lgb
    w = fit_w(ds, P) if pre is None else [np.nan] * 3
    shipped = lgb.Booster(model_file=str(STACK))
    res = {"w": [round(x, 4) for x in w]}
    fs = {}
    for st in ("ship", "refit"):
        per = {}
        for c, d in ds.items():
            p = blended(d, P[c], w) if pre is None else pre[c]
            if c == "US":
                p = stacked(d, p, shipped if st == "ship" else refit_stack(d, p, tag))
            per[c] = f05(d, p, list(ts))
        hv = np.concatenate([ds[c]["h"]["half"] for c in ds])
        allf = {t: np.concatenate([per[c][t] for c in ds]) for t in ts}
        t0 = max(ts, key=lambda t: allf[t][hv == 0].mean())
        fs[f"{st}_t71"], fs[f"{st}_tsw"] = allf[T8], allf[t0]
        res[f"t_half0_{st}"] = float(t0)
    return res, fs


def band3(ds: dict, P: dict, ms: tuple) -> tuple:
    """The new model(s) inside the CE stack instead of before it: band = m2_big's band; pairs with a bge r get
    sigmoid(a logit p_lgb + b logit p_new.. + c logit r + d), fit on the half-0 holdout S1's band pairs (both countries
    pooled, as fit_w); band pairs without a bge r keep #7's e5 fallback on p_lgb; every other pair keeps p_lgb."""
    from sklearn.linear_model import LogisticRegression
    X, Y = [], []
    for c, d in ds.items():
        m = band(d["p_lgb"]) & ~np.isnan(d["r_bge"]) & half0_rows(d)
        X.append(np.column_stack([lg(P[c][k][m]) for k in ms] + [lg(d["r_bge"][m])]))
        Y.append(d["y"][m])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate(X), np.concatenate(Y))
    out = {}
    for c, d in ds.items():
        p = blended(d, d["p_lgb"], W8)                   # the e5 fallback part; the bge part is replaced below
        m = band(d["p_lgb"]) & ~np.isnan(d["r_bge"])
        z = np.column_stack([lg(P[c][k][m]) for k in ms] + [lg(d["r_bge"][m])]) @ lr.coef_[0] + lr.intercept_[0]
        p[m] = sg(z)
        out[c] = p
    return out, [round(float(x), 4) for x in lr.coef_[0]] + [round(float(lr.intercept_[0]), 4)]


def paired(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict:
    x = (a - b)[m]
    return {"gain": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "up": int((x > 0).sum()),
            "down": int((x < 0).sum())}


def evaluate(models=("xgb", "cat"), name="p22"):
    """Every variant vs two controls on half 1: #8c as shipped (W8, shipped stacker, t 0.71) and the control put
    through the variant's own procedure (w refit, stacker as the setting, t as the setting)."""
    t00 = time.time()
    ds = {c: load(c) for c in C}
    P = {c: preds(ds[c], c, models) for c in C}
    V, info = variants(ds, P, models)
    hv = np.concatenate([ds[c]["h"]["half"] for c in C])
    grp = np.concatenate([[c] * ds[c]["h"]["n"] for c in C])
    h1 = hv == 1
    ship = []
    for c, d in ds.items():                                  # #8c exactly as shipped
        p = blended(d, d["p_lgb"], W8)
        ship.append(f05(d, stacked(d, p) if c == "US" else p, T8))
    ship = np.concatenate(ship)
    ctl_res, ctl = run_variant(ds, {c: P[c]["lgb"] for c in C}, "ctl")
    out = {"control_8c_shipped": {"f05_half1": float(ship[h1].mean()), "f05_all": float(ship.mean())},
           "control_refit": ctl_res | {k: float(v[h1].mean()) for k, v in ctl.items()}, "lr_weights": info}
    for tag, pv in V.items():
        b = {c: band(pv[c]) for c in C}
        new = {c: b[c] & np.isnan(ds[c]["r_bge"]) for c in C}
        r, f = run_variant(ds, pv, tag)
        r |= {"band": int(sum(b[c].sum() for c in C)),
              "band_no_bge_holdoutS1": int(sum((new[c] & (ds[c]["pos"] >= 0)).sum() for c in C)),
              "band_no_ce": int(sum((new[c] & np.isnan(ds[c]["r_e5"])).sum() for c in C))}
        for k, v in f.items():
            r[f"f05_half1_{k}"] = float(v[h1].mean())
            r[f"vs_ctl_{k}"] = paired(v, ctl[k], h1)
            for c in C:
                r[f"vs_ctl_{k}_{c}"] = paired(v, ctl[k], h1 & (grp == c))["gain"]
        r["vs_8c_refit_t71"] = paired(f["refit_t71"], ship, h1)
        r["vs_8c_ship_t71"] = paired(f["ship_t71"], ship, h1)
        out[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k.startswith(("vs_", "band", "w", "t_"))}, default=float),
              f"peak {peak():.2f} GB", flush=True)
        gc.collect()
    sets = [(m,) for m in models] + ([tuple(models)] if len(models) > 1 else [])
    b3c, wc = band3(ds, P, ("lgb",))                     # control of the same form: lgb + r only
    _, fc = run_variant(ds, {c: P[c]["lgb"] for c in C}, "b3ctl", pre=b3c)
    out["b3_ctl"] = {"w3": wc} | {f"f05_half1_{k}": float(v[h1].mean()) for k, v in fc.items()}
    for s in sets:
        tag = "b3_" + "+".join(s)
        pre, w3 = band3(ds, P, ("lgb",) + s)
        r, f = run_variant(ds, {c: P[c]["lgb"] for c in C}, tag, pre=pre)
        r["w3"] = w3
        for k, v in f.items():
            r[f"f05_half1_{k}"] = float(v[h1].mean())
            r[f"vs_ctl_{k}"] = paired(v, fc[k], h1)
            for c in C:
                r[f"vs_ctl_{k}_{c}"] = paired(v, fc[k], h1 & (grp == c))["gain"]
        r["vs_8c_refit_t71"] = paired(f["refit_t71"], ship, h1)
        r["vs_8c_ship_t71"] = paired(f["ship_t71"], ship, h1)
        out[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k.startswith(("vs_", "w"))}, default=float), flush=True)
    out["runtime_s"], out["peak_gb"] = round(time.time() - t00), round(peak(), 2)
    (OUT / f"eval_{name}.json").write_text(json.dumps(out, default=float, indent=1))
    return out

def vp(kind: str, P: dict, ms: tuple, wts=None) -> np.ndarray:
    """One country's variant p from P[model] (same rows): lr (wts = coefs + intercept over ms), avg, rank, alone."""
    if kind == "alone":
        return P[ms[-1]]
    if kind == "rank":
        n = len(P["lgb"])
        u = np.mean([pd.Series(P[m]).rank(method="average").to_numpy() for m in ms], axis=0)
        return np.interp(u, np.arange(1, n + 1), np.sort(P["lgb"])).astype(np.float32)
    Z = np.column_stack([lg(P[m]) for m in ms])
    return sg(Z @ np.array(wts[:-1]) + wts[-1]) if kind == "lr" else sg(Z.mean(1))


def spec(tag: str, ev_json: str = "p22") -> tuple:
    """'lr_xgb+cat' -> ('lr', ('lgb', 'xgb', 'cat'), weights from eval_<ev_json>.json)."""
    kind, rest = tag.split("_", 1)
    ms = (("lgb",) if kind != "alone" else ()) + tuple(rest.split("+"))
    wts = json.loads((OUT / f"eval_{ev_json}.json").read_text())["lr_weights"][tag] if kind == "lr" else None
    return kind, ms, wts


def test_preds(c: str, models) -> tuple:
    """(m2_big test cache table [q1, qx, p], {model: p aligned to its rows})."""
    t = pd.read_parquet(WORK / "p9" / f"m2_test_{c}_{M2}.parquet")
    k = (t.q1.to_numpy(np.int64) << 32) | t.qx.to_numpy(np.int64)
    P = {"lgb": t.p.to_numpy(np.float32)}
    for m in models:
        if m == "lgb":
            continue
        x = pd.read_parquet(OUT / m / "pred" / f"test_{c}.parquet")
        s = pd.Series(x.p.to_numpy(np.float32), index=(x.q1.to_numpy(np.int64) << 32) | x.qx.to_numpy(np.int64))
        P[m] = s.reindex(k).to_numpy(np.float32)
        assert not np.isnan(P[m]).any(), f"{m} test {c}: rows without a p"
    return t, P


def prob_test(tag: str, ev_json: str = "p22"):
    """The adopted variant's M2 p on #6's pruned test rows (m2_big cache order) for every test label ->
    data/work/p22/prob_test_<c>.parquet (s1, sx, q1, qx, p, p_lgb). p is M2's p (before the CE / stacker)."""
    import features as F
    kind, ms, wts = spec(tag, ev_json)
    out = {}
    for c in sorted(pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist()):
        t, P = test_preds(c, ms)
        p = vp(kind, P, ms, wts)
        id1, idx = F.ids("test", c)
        pq.write_table(pa.table({"s1": id1.take(pa.array(t.q1.to_numpy())), "sx": idx.take(pa.array(t.qx.to_numpy())),
                                 "q1": t.q1.to_numpy(), "qx": t.qx.to_numpy(), "p": p, "p_lgb": P["lgb"]}),
                       OUT / f"prob_test_{c}.parquet")
        b = band(p)
        out[c] = {"pairs": len(p), "band": int(b.sum()), "band_lgb": int(band(P["lgb"]).sum()),
                  "corr_logit": float(np.corrcoef(lg(p), lg(P["lgb"]))[0, 1])}
        print(c, out[c], flush=True)
    return out


SM = OUT / "sm_bge"


def bge_build(tag: str, ev_json: str = "p22", n_ctl: int = 1000):
    """P13's bge_ckpt data channel (P19's pattern) for the variant's band pairs without a bge r: holdout S1 band pairs
    (y from the frame) and test India / US band pairs, each + n_ctl control pairs P13 already scored."""
    import features as F
    import rerank as RR
    kind, ms, wts = spec(tag, ev_json)
    SM.mkdir(parents=True, exist_ok=True)
    src = WORK / "p13/sm_data"
    res = {}
    for c in C:
        d = load(c)
        P = preds(d, c, [m for m in ms if m != "lgb"])
        p = vp(kind, P, ms, wts)
        miss = band(p) & np.isnan(d["r_bge"]) & (d["pos"] >= 0)
        id1, idx = F.ids("train", c)
        q1, qx = d["q1"][miss], d["qx"][miss]
        ta, tb = RR.texts("train", c, q1, qx)
        new = pd.DataFrame({"s1": id1.take(pa.array(q1)).to_numpy(zero_copy_only=False),
                            "sx": idx.take(pa.array(qx)).to_numpy(zero_copy_only=False), "ta": ta, "tb": tb,
                            "y": d["y"][miss].astype(np.int8), "p": p[miss]})
        ctl = pd.read_parquet(src / f"holdout_{c}.parquet").sample(n_ctl, random_state=0)
        pd.concat([new, ctl[new.columns]], ignore_index=True).to_parquet(SM / f"holdout_{c}.parquet", index=False)
        del d, P
        gc.collect()
        t, Pt = test_preds(c, ms)
        pt = vp(kind, Pt, ms, wts)
        kt = (t.q1.to_numpy(np.int64) << 32) | t.qx.to_numpy(np.int64)
        rt = lookup([WORK / "p9" / f"r_test_{c}_{BGE}.parquet"], kt)
        mt = band(pt) & np.isnan(rt)
        id1, idx = F.ids("test", c)
        q1, qx = t.q1.to_numpy()[mt], t.qx.to_numpy()[mt]
        ta, tb = RR.texts("test", c, q1, qx)
        newt = pd.DataFrame({"s1": id1.take(pa.array(q1)).to_numpy(zero_copy_only=False),
                             "sx": idx.take(pa.array(qx)).to_numpy(zero_copy_only=False), "ta": ta, "tb": tb,
                             "p": pt[mt]})
        ctlt = pd.read_parquet(src / f"test_{c}.parquet").sample(n_ctl, random_state=0)
        pd.concat([newt, ctlt[newt.columns]], ignore_index=True).to_parquet(SM / f"test_{c}.parquet", index=False)
        pd.read_parquet(src / f"val_{c}.parquet").sample(200, random_state=0).to_parquet(SM / f"val_{c}.parquet", index=False)
        res[c] = {"holdout_new": len(new), "test_new": len(newt), "test_band": int(band(pt).sum())}
        print(c, res[c], flush=True)
    pd.read_parquet(src / "test_France.parquet").sample(200, random_state=0).to_parquet(SM / "test_France.parquet", index=False)
    return res


def repro():
    ds = {c: load(c) for c in C}
    out = {}
    for c, d in ds.items():
        p = blended(d, d["p_lgb"], W8)
        if c == "US":
            p = stacked(d, p)
        f = f05(d, p, T8)
        hv = d["h"]["half"]
        out[c] = {"all": f.mean(), "h0": f[hv == 0].mean(), "h1": f[hv == 1].mean(), "n": len(f)}
        out[c]["_f"], out[c]["_h"] = f, hv
    f = np.concatenate([out[c].pop("_f") for c in C])
    hv = np.concatenate([out[c].pop("_h") for c in C])
    res = {"all": f.mean(), "h0": f[hv == 0].mean(), "h1": f[hv == 1].mean(), **{c: out[c] for c in C},
           "band_no_bge_holdoutS1": {c: int((band(ds[c]["p_lgb"]) & np.isnan(ds[c]["r_bge"]) & (ds[c]["pos"] >= 0)).sum()) for c in C},
           "band_no_ce": {c: int((band(ds[c]["p_lgb"]) & np.isnan(ds[c]["r_bge"]) & np.isnan(ds[c]["r_e5"])).sum()) for c in C},
           "peak_gb": peak()}
    print(json.dumps(res, default=float, indent=1))
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "frame":
        frame(args[0])
    elif cmd == "repro":
        repro()
    elif cmd == "eval":
        evaluate(tuple(args[0].split(",")) if args else ("xgb", "cat"), args[1] if len(args) > 1 else "p22")
    elif cmd == "probtest":
        print(prob_test(*args))
    elif cmd == "bgebuild":
        print(bge_build(*args))
    elif cmd == "cecheck":
        ce_check()
    else:
        sys.exit(__doc__)
