"""P4b-rerank: a multilingual cross-encoder (intfloat/multilingual-e5-small, MIT, 118M) rescoring the GBDT's
uncertain band, blended back into p before the decision layer. Adopted config = CFG below.

    rerank.py band <country>    # train-side band: quick LightGBM on P4's X (fit fold 2, predict folds 3-9)
    flock /tmp/amc-gpu.lock rerank.py train [countries=India,US tag=]   # band pairs of folds >= 4 -> data/work/p4b/ce<tag>.pt
    flock /tmp/amc-gpu.lock rerank.py score <p5|p4> band [tag=]         # r for holdout pairs 0.02<p<0.98 -> r<tag>_holdout
    rerank.py eval <p5|p4>      # blend per band width, D3 re-tuned on holdout half 0, reported on half 1 (ledger)
    rerank.py robust p4         # paired s.e. of the gain + leave-one-country-out (needs ce_loco_<c> scores)
    flock /tmp/amc-gpu.lock rerank.py france    # label-free France check on rules-accepted pairs
    flock /tmp/amc-gpu.lock rerank.py rescore <split> <country> <in.parquet> <out.parquet>   # PIPELINE STEP
    rerank.py blend_file        # P4's prob_holdout with the adopted blend -> data/work/p4b/prob_holdout_blend.parquet
    rerank.py check             # blend-rule self-check

Texts are P3's cleaned (Latin, lower-case) "name | addr" of both records. q1 = row of data/work/p3/txt/<c>_s1,
qx = row of the concatenated <c>_sx_* files (P4's convention). Folds 0 (holdout) and 1 (P4's early stopping)
are never trained on; fold 2 fits the band model, fold 3 is validation.
"""
import gc, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src")]
WORK = ROOT / "data/work"
OUT = WORK / "p4b"
OUT.mkdir(exist_ok=True)
TXT = {"train": WORK / "p3/txt", "test": WORK / "p3/txt_test"}
BASE = "intfloat/multilingual-e5-small"
LO, HI = 0.1, 0.9            # the band
MAXLEN = 64                  # pair tokens p50 ~50 / p99 ~75; longest_first truncation drops the address tail
                             # (city/state) and keeps name + house number + street; GPU-bound, 64 vs 96 = ~1.5x
ID = ["q1", "qx", "is_true", "fold"]
# Adopted config (RESULT.md): band + stack weights fit on holdout half 0 of P4's M2 p (ledger p4-D3-blend-0.02-0.98).
# countries = the CE's training countries. Leave-one-country-out LOSES F0.5 (India->US -0.0016), so a country
# without training labels (France, any unseen label) keeps the GBDT p unchanged -- it is not filtered.
CFG = {"band": (0.02, 0.98), "w": [0.636, 0.757, 0.355], "countries": ("India", "US")}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def band(country: str, seed: int = 0):
    """P4-like LightGBM fitted on fold 2, predicting folds 3-9: keep its band + 5% of the rest (robustness to a
    different band definition at eval time). -> data/work/p4b/trainband_<country>.parquet"""
    import lightgbm as lgb
    t0 = time.time()
    files = sorted((WORK / "p4/X").glob(f"train_{country}_sx_*.parquet"))
    cols = [c for c in pq.read_schema(files[0]).names if c not in ID]
    d = pd.concat([pq.read_table(p, filters=[("fold", "==", 2)]).to_pandas() for p in files], ignore_index=True)
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6),
                  lgb.Dataset(d[cols].to_numpy(np.float32), d.is_true.to_numpy(np.float32), feature_name=cols), 300)
    print(f"{country}: fit on {len(d):,} fold-2 pairs {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    del d
    gc.collect()
    rng, out = np.random.default_rng(seed), []
    for p in files:
        d = pq.read_table(p, filters=[("fold", ">=", 3)]).to_pandas()
        pr = m.predict(d[cols].to_numpy(np.float32), num_threads=6).astype(np.float32)
        inb = (pr > LO) & (pr < HI)
        keep = inb | (rng.random(len(d)) < 0.05)
        out.append(d.loc[keep, ID].assign(p=pr[keep], inband=inb[keep]))
        del d
        gc.collect()
    b = pd.concat(out, ignore_index=True)
    b.to_parquet(OUT / f"trainband_{country}.parquet", index=False)
    ib = b[b.inband]
    print(f"{country}: {len(ib):,} band pairs (true {ib.is_true.mean():.3f}), {len(b) - len(ib):,} out-of-band, "
          f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def _join(t: pa.Table) -> np.ndarray:
    import pyarrow.compute as pc
    return np.array(pc.binary_join_element_wise(t["name"], t["addr"], pa.scalar(" | ", pa.large_string())).to_pylist(), dtype=object)


def texts(split: str, country: str, q1: np.ndarray, qx: np.ndarray):
    """("name | addr" of S1 row q1, same of SX row qx) as object arrays; one SX text file in memory at a time."""
    u1, i1 = np.unique(q1, return_inverse=True)
    a = _join(pq.read_table(TXT[split] / f"{country}_s1.parquet", columns=["name", "addr"]).take(u1))[i1]
    ux, ix = np.unique(qx, return_inverse=True)
    tx, off = np.empty(len(ux), dtype=object), 0
    for p in sorted(TXT[split].glob(f"{country}_sx_*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        lo, hi = np.searchsorted(ux, [off, off + n])
        if hi > lo:
            tx[lo:hi] = _join(pq.read_table(p, columns=["name", "addr"]).take(ux[lo:hi] - off))
        off += n
    return a, tx[ix]


def rows(split: str, country: str, s1: pd.Series, sx: pd.Series):
    """Entity ids -> (q1, qx) rows of P3's text files (for pairs that carry ids, e.g. P5/P4 holdout files)."""
    i1 = pd.Index(pq.read_table(TXT[split] / f"{country}_s1.parquet", columns=["entity_id"]).column(0).to_pandas())
    q1 = i1.get_indexer(s1)
    del i1
    ix = pd.Index(pd.concat([pq.read_table(p, columns=["entity_id"]).column(0).to_pandas()
                             for p in sorted(TXT[split].glob(f"{country}_sx_*.parquet"))], ignore_index=True))
    qx = ix.get_indexer(sx)
    assert (q1 >= 0).all() and (qx >= 0).all(), "every pair id is in P3's texts"
    return q1, qx


def load_model(path=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    if path:
        m.load_state_dict({k: v.float() for k, v in torch.load(path).items()}, strict=False)
    return tok, m.cuda()


def score(ta, tb, tok, m, bs: int = 512) -> np.ndarray:
    """sigmoid(logit) per pair; length-sorted batches so padding stays small."""
    import torch
    o = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)], kind="stable")
    r = np.empty(len(ta), np.float32)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(o), bs):
            j = o[i:i + bs]
            e = tok(list(ta[j]), list(tb[j]), truncation="longest_first", max_length=MAXLEN, padding=True,
                    return_tensors="pt").to("cuda")
            r[j] = torch.sigmoid(m(**e).logits.float().squeeze(-1)).cpu().numpy()
    return r


def train(epochs: int = 1, bs: int = 128, lr: float = 6e-5, oob: float = 0.5, seed: int = 0,
          countries=("India", "US"), tag: str = ""):
    """Band pairs of folds 4-9 (+ oob x as many out-of-band pairs) -> ce<tag>.pt; fold 3 band = validation."""
    import torch
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    rng = np.random.default_rng(seed)
    parts, val = [], []
    for c in countries:
        b = pd.read_parquet(OUT / f"trainband_{c}.parquet")
        v = b[(b.fold == 3) & b.inband]
        b = b[b.fold >= 4]
        ib, ob = b[b.inband], b[~b.inband]
        b = pd.concat([ib, ob.sample(min(len(ob), int(oob * len(ib))), random_state=seed)])
        v = v.sample(min(len(v), 20_000), random_state=seed)
        for d, acc in ((b, parts), (v, val)):
            ta, tb = texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            acc.append((ta, tb, d.is_true.to_numpy(np.float32), d.p.to_numpy(np.float32)))
        del b, v, ib, ob
        gc.collect()
    ta, tb, y, _ = (np.concatenate(x) for x in zip(*parts))
    va, vb, vy, vp = (np.concatenate(x) for x in zip(*val))
    del parts, val
    print(f"{len(y):,} train pairs (true {y.mean():.3f}), {len(vy):,} val, texts {time.time() - t0:.0f}s, "
          f"peak {peak():.2f} GB", flush=True)
    torch.manual_seed(seed)
    tok, m = load_model()
    m.bert.embeddings.word_embeddings.weight.requires_grad_(False)     # 96M of the 118M params; saves VRAM + disk
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * (len(y) // bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * max(0.0, 1 - s / steps))
    lossf, step = torch.nn.BCEWithLogitsLoss(), 0
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(len(y))       # length buckets: sort within 100-batch windows, shuffle the batches
        ln = np.array([len(x) + len(z) for x, z in zip(ta, tb)])
        w = bs * 100
        perm = np.concatenate([c[np.argsort(ln[c], kind="stable")] for c in np.split(perm, range(w, len(perm), w))])
        batches = [perm[i:i + bs] for i in range(0, len(perm) - bs + 1, bs)]
        for k in rng.permutation(len(batches)):
            j = batches[k]
            e = tok(list(ta[j]), list(tb[j]), truncation="longest_first", max_length=MAXLEN, padding=True,
                    return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = m(**e).logits.float().squeeze(-1)
            loss = lossf(logit, torch.from_numpy(y[j]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s "
                      f"{step * bs / (time.time() - t0):.0f} pairs/s gpu {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                      flush=True)
        t1 = time.time()
        vr = score(va, vb, tok, m)
        print(f"ep {ep}: val band AUC gbdt {roc_auc_score(vy, vp):.4f} ce {roc_auc_score(vy, vr):.4f} "
              f"mean {roc_auc_score(vy, vp + vr):.4f}; score {len(vy) / (time.time() - t1):.0f} pairs/s", flush=True)
    torch.save({k: v.half() for k, v in m.state_dict().items() if "word_embeddings" not in k}, OUT / f"ce{tag}.pt")
    np.save(OUT / f"val_scores{tag}.npy", np.stack([vy, vp, vr]))
    print(f"train done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


SRC = {"p5": WORK / "p5/prob_proxy.parquet", "p4": WORK / "p4/prob_holdout.parquet"}
RCACHE = OUT / "r_holdout.parquet"


def score_holdout(src: str, band_only: bool = False, tag: str = ""):
    """r for every (s1, sx) of a holdout probability file not yet in r_holdout.parquet (one cache for all srcs)."""
    import torch
    t0 = time.time()
    d = pd.read_parquet(SRC[src], columns=["s1", "sx", "p"])
    if band_only:                          # a wider band than LO/HI, so band-width variants need no rescoring
        d = d[(d.p > 0.02) & (d.p < 0.98)]
    cache = OUT / f"r{tag}_holdout.parquet"
    if cache.exists():
        have = pd.read_parquet(cache, columns=["s1", "sx"]).assign(h=True)
        d = d.merge(have, on=["s1", "sx"], how="left")
        d = d[d.h.isna()].drop(columns="h")
    cty = pd.read_parquet(WORK / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    d["country"] = d.s1.map(cty.set_index("entity_id").country)
    del cty
    print(f"{src}: {len(d):,} pairs to score", flush=True)
    tok = m = None
    out = [pd.read_parquet(cache)] if cache.exists() else []
    for c, g in d.groupby("country"):
        q1, qx = rows("train", c, g.s1, g.sx)
        ta, tb = texts("train", c, q1, qx)
        gc.collect()
        if m is None:
            tok, m = load_model(OUT / f"ce{tag}.pt")
        t1 = time.time()
        r = score(ta, tb, tok, m)
        print(f"{c}: {len(g):,} pairs scored at {len(g) / (time.time() - t1):.0f} pairs/s, peak {peak():.2f} GB",
              flush=True)
        out.append(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "r": r}))
        del ta, tb, q1, qx
        gc.collect()
    pd.concat(out, ignore_index=True).to_parquet(cache, index=False)
    print(f"score done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def logit(x):
    x = np.clip(x.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def fit_blend(p: np.ndarray, r: np.ndarray, y: np.ndarray) -> list:
    """Logistic stack on the band: p' = sigmoid(a*logit(p) + b*logit(r) + c). Returns [a, b, c]."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0).fit(np.c_[logit(p), logit(r)], y)
    return [float(lr.coef_[0, 0]), float(lr.coef_[0, 1]), float(lr.intercept_[0])]


def blend(p: np.ndarray, r: np.ndarray, w: list, lo: float = LO, hi: float = HI) -> np.ndarray:
    """THE BLEND RULE. Pairs with lo < p < hi and a reranker score get the stacked p'; every other pair keeps p.
    r = NaN means not scored (outside the band at inference) -> p unchanged."""
    b = (p > lo) & (p < hi) & ~np.isnan(r)
    out = p.astype(np.float32).copy()
    z = w[0] * logit(p[b]) + w[1] * logit(r[b]) + w[2]
    out[b] = 1 / (1 + np.exp(-z))
    return out


def apply(split: str, country: str, s1: pd.Series, sx: pd.Series, p: np.ndarray, w: list, band=(LO, HI),
          model=None):
    """Pipeline hook: p' for pairs (entity ids + GBDT p) of one country. Only band pairs reach the GPU; a
    country the CE was not trained on returns p unchanged."""
    r = np.full(len(p), np.nan, np.float32)
    if country not in CFG["countries"]:
        return p.astype(np.float32)
    b = np.flatnonzero((p > band[0]) & (p < band[1]))
    if len(b):
        q1, qx = rows(split, country, s1.iloc[b], sx.iloc[b])
        ta, tb = texts(split, country, q1, qx)
        tok, m = model or load_model(OUT / "ce.pt")
        r[b] = score(ta, tb, tok, m)
    return blend(p, r, w, *band)


def rescore(split: str, country: str, inp: str, outp: str):
    """Own-process pipeline step (torch + CUDA = 1.4 GB RSS, too much inside run.py's cap): parquet[s1, sx, p] of
    one country -> the same rows with p replaced by the blended p' (CFG band + weights)."""
    t0 = time.time()
    d = pd.read_parquet(inp)
    nb = int(((d.p > CFG["band"][0]) & (d.p < CFG["band"][1])).sum()) if country in CFG["countries"] else 0
    d["p"] = apply(split, country, d.s1, d.sx, d.p.to_numpy(np.float32), CFG["w"], band=CFG["band"])
    d.to_parquet(outp, index=False)
    print(f"{country}: {len(d):,} pairs, {nb:,} in band rescored, {time.time() - t0:.0f}s, peak {peak():.2f} GB")


def blend_file(src: str = "p4"):
    """Holdout probabilities with the adopted blend applied from the r cache -> data/work/p4b/prob_holdout_blend.parquet
    (P4's schema, p replaced), so P5's bench / the master can re-tune any decision rule on it."""
    d = pd.read_parquet(SRC[src])
    r = d[["s1", "sx"]].merge(pd.read_parquet(RCACHE), on=["s1", "sx"], how="left").r.to_numpy(np.float32).copy()
    ok = d.country.isin(CFG["countries"]).to_numpy()
    r[~ok] = np.nan
    b = (d.p > CFG["band"][0]) & (d.p < CFG["band"][1])
    assert not np.isnan(r[b.to_numpy() & ok]).any(), "every band pair of a trained country has r"
    d["p_gbdt"], d["p"] = d.p, blend(d.p.to_numpy(np.float32), r, CFG["w"], *CFG["band"])
    d.to_parquet(OUT / "prob_holdout_blend.parquet", index=False)
    print(f"{len(d):,} pairs, {int(b.sum()):,} blended -> {OUT / 'prob_holdout_blend.parquet'}")


def p5_harness():
    for d in (ROOT / "probes/P5-decide", ROOT.parent / "p5-decide/probes/P5-decide"):
        if (d / "bench.py").exists():
            sys.path.insert(0, str(d))
            import bench
            return bench
    raise FileNotFoundError("P5-decide bench.py")


def d3(B, d, S):
    """P5's D3 (each SX to its argmax S1, keep p >= t or top-1, S1 answers only if best p >= t0), tuned on
    half 0. Same logic as decide.select, with the t-independent parts (argmax, top-1, best) computed once."""
    import decide as D
    p = d.p.to_numpy(np.float64)
    g, x = D._codes(d.s1), D._codes(d.sx)
    keep = p >= D._gmax(x, p, x.max() + 1)[x]
    pe = np.where(keep, p, 0.0)
    best, top1 = D._gmax(g, pe, g.max() + 1)[g], D._rank(g, pe) == 0
    s1, y, n = d.s1.to_numpy(), d.is_true.to_numpy(), len(S)
    nt, m0 = S.nt.to_numpy(), S.half.to_numpy() == 0

    def f(t, t0):
        sel = ((pe >= t) | top1) & (best >= t0) & keep
        tp, npred = np.bincount(s1[sel & y], minlength=n), np.bincount(s1[sel], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(nt == 0, npred == 0, 1.25 * tp / (0.25 * nt + npred))[m0].mean()

    T = np.round(np.arange(0.05, 0.96, 0.05), 2)
    t, t0 = max(((a, b) for a in T for b in T), key=lambda c: f(*c))
    R = lambda v: np.clip(np.round(v + np.arange(-0.03, 0.031, 0.01), 2), 0.01, 0.99)
    t, t0 = max(((a, b) for a in R(t) for b in R(t0)), key=lambda c: f(*c))
    return {"excl": True, "t": float(t), "t0": float(t0)}


def evaluate(src: str):
    """Same pairs, same truth, same D3 decision: p vs blend(p, r) on the band. Tune half 0, report half 1."""
    from sklearn.metrics import roc_auc_score
    from ledger import log
    B = p5_harness()
    t0 = time.time()
    d, S = B.load({"p5": "proxy"}.get(src, src))
    ids = pd.read_parquet(SRC[src], columns=["s1", "sx"])
    r = ids.merge(pd.read_parquet(RCACHE), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
    assert len(r) == len(d)
    fe = pd.concat([pd.read_parquet(WORK / f"p5/feat_{c}.parquet", columns=["s1", "sx", "core_eq", "hn_miss"])
                    for c in ("India", "US")], ignore_index=True)
    look = ids.merge(fe, on=["s1", "sx"], how="left").eval("core_eq == 1 and hn_miss == 0").to_numpy(bool)
    del ids, fe
    gc.collect()
    p, y, h = d.p.to_numpy(), d.is_true.to_numpy(), S.half.to_numpy()[d.s1.to_numpy()]
    band = (p > LO) & (p < HI)
    print(f"{src}: {len(d):,} pairs, band {band.sum():,}, scored in band {(~np.isnan(r[band])).mean():.4f}, "
          f"look-alike in band {look[band].mean():.3f}, peak {peak():.2f} GB", flush=True)
    res, ws = {}, {}

    def run(nm, pp, extra=None):
        dd = d.assign(p=pp)
        c = d3(B, dd, S)
        sel = B.select(dd, c)
        m = B.metrics(dd, S, sel, 1)
        m["fp_lookalike"] = int((sel & ~y & look & (h == 1)).sum())
        m["tp_lookalike"] = int((sel & y & look & (h == 1)).sum())
        res[nm] = (c, m | (extra or {}))
        print(f"{nm:18s} t={c['t']:.2f} t0={c['t0']:.2f}  " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)

    run("base", p)
    for lo, hi in ((LO, HI), (0.05, 0.95), (0.02, 0.98), (0.2, 0.8)):
        b = (p > lo) & (p < hi) & ~np.isnan(r)
        w = fit_blend(p[b & (h == 0)], r[b & (h == 0)], y[b & (h == 0)])
        pb = blend(p, r, w, lo, hi)
        rep = b & (h == 1)
        auc = {f"auc_{k}_{nm}": round(roc_auc_score(y[m], v[m]), 4)
               for nm, m in (("band", rep), ("lookalike", rep & look))
               for k, v in (("p", p), ("r", r), ("blend", pb))}
        nm = f"blend-{lo}-{hi}"
        ws[nm] = (w, lo, hi, int(b.sum()), b.mean())
        print(f"{nm}: w = {np.round(w, 3).tolist()} band pairs {b.sum():,} ({b.mean():.4f})  {auc}", flush=True)
        run(nm, pb, auc | {"band_pairs": int(b.sum()), "band_share": round(float(b.mean()), 4)})
    for nm, (c, m) in res.items():
        gain = m["f05"] - res["base"][1]["f05"]
        print(f"{nm:18s} F0.5 {m['f05']:.4f}  gain {gain:+.4f}", flush=True)
        log("exp", probe="P4b-rerank", subproblem="rerank", name=f"{src}-D3-{nm}", world="holdout-dense-half1",
            metrics={k: round(float(v), 4) for k, v in m.items()} | ({"gain_f05": round(gain, 4)} if nm != "base" else {}),
            config={"t": c["t"], "t0": c["t0"], "model": BASE, "maxlen": MAXLEN, "src": src}
            | ({"blend_w": [round(x, 4) for x in ws[nm][0]], "band": list(ws[nm][1:3])} if nm != "base" else {}),
            hypothesis="cross-encoder on name|addr text rescoring the GBDT's uncertain band raises F0.5 >= 0.003"
            if nm != "base" else "same pipeline without the reranker (reference)",
            verdict="keep" if (nm == "base" or gain >= 0.003) else "drop",
            runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print(f"eval done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    return res, ws


def robust(src: str = "p4"):
    """(1) paired standard error of the blend gain over half-1 S1; (2) leave-one-country-out as the France proxy:
    CE trained on country a only (ce_loco_<a>.pt), blend weights fit on a's holdout half 0, applied to country b.
    D3 thresholds are tuned on b's half 0 for base and blend alike (threshold transfer is P6's question)."""
    from ledger import log
    B = p5_harness()
    t0 = time.time()
    d, S = B.load(src)
    ids = pd.read_parquet(SRC[src], columns=["s1", "sx"])
    R = lambda tag: ids.merge(pd.read_parquet(OUT / f"r{tag}_holdout.parquet"), on=["s1", "sx"],
                              how="left").r.to_numpy(np.float32)
    p, y, h, cty = d.p.to_numpy(), d.is_true.to_numpy(), S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    m1 = S.half.to_numpy() == 1

    def f1(pp, rows=None):
        dd = d.assign(p=pp) if rows is None else d.assign(p=pp)[rows]
        c = d3(B, dd, S)
        f, _, _ = B.per_s1(dd, S, B.select(dd, c))
        return f, c

    out = {}
    r = R("")
    fb, _ = f1(p)
    for lo, hi in ((LO, HI), (0.02, 0.98)):
        b = (p > lo) & (p < hi) & ~np.isnan(r)
        fx, _ = f1(blend(p, r, fit_blend(p[b & (h == 0)], r[b & (h == 0)], y[b & (h == 0)]), lo, hi))
        dif = (fx - fb)[m1]
        out[f"gain_{lo}_{hi}"], out[f"se_{lo}_{hi}"] = float(dif.mean()), float(dif.std() / np.sqrt(len(dif)))
        out[f"s1_changed_{lo}_{hi}"] = int((dif != 0).sum())
        print(f"band {lo}-{hi}: gain {dif.mean():+.4f} +- {dif.std() / np.sqrt(len(dif)):.4f} (1 se), "
              f"{(dif > 0).sum():,} S1 up, {(dif < 0).sum():,} down of {len(dif):,}", flush=True)
    for a, bc in (("India", "US"), ("US", "India")):
        rl = R(f"_loco_{a}")
        band = (p > LO) & (p < HI) & ~np.isnan(rl)
        rows = cty == bc
        mc = m1 & (S.country.to_numpy() == bc)
        fb_c, _ = f1(p, rows)
        res = {"base": fb_c[mc].mean()}
        for nm, rr, wm in (("loco", rl, cty == a), ("loco_oracle_w", rl, cty == bc), ("both", r, cty == bc)):
            bb = (p > LO) & (p < HI) & ~np.isnan(rr) & (h == 0) & wm
            w = fit_blend(p[bb], rr[bb], y[bb])
            fx, _ = f1(blend(p, rr, w), rows)
            res[nm] = fx[mc].mean()
            res[f"w_{nm}"] = [round(x, 3) for x in w]
        from sklearn.metrics import roc_auc_score
        tb = band & (h == 1) & rows
        res["auc_band_p"], res["auc_band_r_loco"] = roc_auc_score(y[tb], p[tb]), roc_auc_score(y[tb], rl[tb])
        print(f"train {a} -> test {bc}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in res.items()), flush=True)
        for k, v in res.items():
            out[f"{a}>{bc}_{k}"] = v
    log("exp", probe="P4b-rerank", subproblem="rerank", name=f"{src}-robust-se-loco", world="holdout-dense-half1",
        metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if "_w_" not in k},
        config={"band": [LO, HI], "model": BASE, "loco": "ce trained on one country's train band (folds 4-9)",
                "w": {k: v for k, v in out.items() if "_w_" in k}},
        hypothesis="the gain is beyond noise and survives a country the reranker never saw (France proxy)",
        verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print(f"robust done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def france_check(n: int = 20_000, seed: int = 0):
    """Label-free France check: r on rules-accepted pairs (submission #2) for France test vs US/India holdout, where
    the truth tells what a given 'share with r > 0.5' means. Same CE (ce.pt), same texts pipeline."""
    from ledger import log
    t0 = time.time()
    f = next(q for q in (ROOT / "output/sub2", ROOT.parents[1] / "amc-26/output/sub2") if q.exists())
    sub = pd.read_csv(f / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    fr = set(pq.read_table(TXT["test"] / "France_s1.parquet", columns=["entity_id"]).column(0).to_pylist())
    sub = sub[sub.source1_entity_id.isin(fr) & (sub.matched_entity_ids != "")]
    acc = sub.assign(sx=sub.matched_entity_ids.str.split(",")).explode("sx")[["source1_entity_id", "sx"]]
    acc = acc.rename(columns={"source1_entity_id": "s1"}).sample(min(n, len(acc)), random_state=seed)
    del sub, fr
    hp = pd.read_parquet(WORK / "p5/prob_proxy.parquet", columns=["s1", "sx", "is_true", "country", "rule_acc"])
    hp = hp[hp.rule_acc == 1].sample(n, random_state=seed)
    out = {}
    for tag in ("", "_loco_India", "_loco_US"):            # the LOCO models show what 'unseen country' looks like
        tok, m = load_model(OUT / f"ce{tag}.pt")
        if tag == "":
            q1, qx = rows("test", "France", acc.s1, acc.sx)
            rf = score(*texts("test", "France", q1, qx), tok, m)
            out["France_share_r05"], out["France_median_r"] = float((rf > 0.5).mean()), float(np.median(rf))
        for c, g in hp.groupby("country"):
            q1, qx = rows("train", c, g.s1, g.sx)
            rh = score(*texts("train", c, q1, qx), tok, m)
            t = g.is_true.to_numpy()
            k = f"{c}{tag}"
            out |= {f"{k}_share_r05": float((rh > 0.5).mean()), f"{k}_median_r": float(np.median(rh)),
                    f"{k}_share_r05_true": float((rh[t] > 0.5).mean()),
                    f"{k}_share_r05_false": float((rh[~t] > 0.5).mean())}
            if tag == "":
                out[f"{c}_rule_precision"] = float(t.mean())
        del tok, m
        gc.collect()
    print({k: round(v, 4) for k, v in out.items()}, f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    log("exp", probe="P4b-rerank", subproblem="rerank-france", name="france-labelfree-rule-agreement",
        world="test-France+holdout", metrics={k: round(v, 4) for k, v in out.items()},
        config={"pairs": "rules-accepted (sub2 for France, rule_acc for holdout)", "n": n, "model": "ce.pt"},
        hypothesis="if the CE transfers to French text, it confirms rules-accepted France pairs at the US/India rate",
        verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def _check():
    p = np.array([0.05, 0.5, 0.5, 0.5, 0.95], np.float32)
    r = np.array([0.9, np.nan, 0.1, 0.9, 0.1], np.float32)
    q = blend(p, r, [1.0, 1.0, 0.0])
    assert q[0] == p[0] and q[4] == p[4] and q[1] == p[1], "outside the band / unscored pairs keep p"
    assert q[2] < 0.5 < q[3], "r moves band pairs both ways"
    assert np.allclose(blend(p, np.full(5, 0.5, np.float32), [1.0, 0.0, 0.0]), p, atol=1e-6), "identity weights"
    print("rerank self-check ok")


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)       # tag=_loco_India countries=India
    kw = {k: tuple(v.split(",")) if k == "countries" else v for k, v in kw.items()}
    {"band": lambda: band(sys.argv[2]), "train": lambda: train(**kw),
     "score": lambda: score_holdout(sys.argv[2], "band" in sys.argv[3:], kw.get("tag", "")), "eval": lambda: evaluate(sys.argv[2]), "robust": lambda: robust(sys.argv[2]), "france": france_check, "check": _check,
     "rescore": lambda: rescore(*sys.argv[2:6]), "blend_file": blend_file}[cmd]()
