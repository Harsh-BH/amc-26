"""P11-ce-big: a better-fed / bigger cross-encoder than P4b's e5-small, on India + US (France is never touched).

    ce.py band <country>              # train band: m2fast_fold2 p on folds 3-5, m2_big p on folds 6-9 (both unseen)
    flock /tmp/amc-gpu.lock ce.py train tag=<t> [base=small|base epochs=1 src=all|big oob=0.1 bs=128 acc=1 lr=]
    flock /tmp/amc-gpu.lock ce.py score tag=<t>      # r for prob_holdout_big's band pairs -> p11/r_<t>_holdout
    ce.py eval tag=<t>                # P4's yardstick: blend w + D3 fit on half 0, report half 1, paired se vs #5's CE
    flock /tmp/amc-gpu.lock ce.py test tag=<t> <country>   # (winner) m2_big test p blended -> prob_test_<c>_blend

Texts, rows and the blend rule are P4b's (probes/P4b-rerank/rerank.py, imported read-only). Train texts are
tokenised once into flat int32 arrays (p11/tok_<country>_<n>.npz) so the GPU loop does no string work and the
RSS stays small. Folds 0 (holdout) and 1 (m2_big's early stopping) are never trained on.
"""
import gc, json, os, resource, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P4b-rerank")]
import rerank as rr                                    # P4b: texts, rows, blend, fit_blend, d3, logit

WORK = ROOT / "data/work"
OUT = WORK / "p11"
OUT.mkdir(exist_ok=True)
P4 = WORK / "p4"
LO, HI = 0.02, 0.98                    # = the inference band (rr.CFG["band"])
BASES = {"small": "intfloat/multilingual-e5-small", "base": "intfloat/multilingual-e5-base"}
MAXLEN = 64
ID = ["q1", "qx", "is_true", "fold"]
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


PRE = (5e-4, 1 - 5e-4)       # m2fast p outside this never reaches m2_big's band (checked on a sample: `miss` below)


def band(country: str, seed: int = 0, n_check: int = 20_000):
    """Out-of-sample GBDT p for train folds 3-9: m2fast_fold2 (fit on fold 2) on all of them; m2_big (folds 2-5,
    early-stopped on fold 1) on folds 6-9, only where m2fast is not sure (PRE) -- the CPU is shared and m2_big
    costs ~5x m2fast. p = m2fast on folds 3-5, m2_big on folds 6-9 (m2fast where m2_big was skipped).
    Keep the 0.02-0.98 band of p + 5% of the rest -> p11/trainband_<c>.parquet."""
    import lightgbm as lgb
    t0, rng, out, miss = time.time(), np.random.default_rng(seed), [], []
    fast = lgb.Booster(model_file=str(P4 / "m2fast_fold2.txt"))
    big = lgb.Booster(model_file=str(P4 / "m2_big.txt"))
    for f in sorted((P4 / "X").glob(f"train_{country}_sx_*.parquet")):
        d = pq.read_table(f, filters=[("fold", ">=", 3)]).to_pandas()
        pf = fast.predict(d[fast.feature_name()], num_threads=8).astype(np.float32)
        late = d.fold.to_numpy() >= 6
        unsure = late & (pf > PRE[0]) & (pf < PRE[1])
        pb = np.full(len(d), np.nan, np.float32)
        pb[unsure] = big.predict(d.loc[unsure, big.feature_name()], num_threads=8)
        skip = np.flatnonzero(late & ~unsure)            # coverage check: m2_big on a sample of the skipped pairs
        chk = rng.choice(skip, min(len(skip), n_check // 8), replace=False)
        miss.append(((q := big.predict(d.iloc[chk][big.feature_name()], num_threads=8)) > LO) & (q < HI))
        p = np.where(late & unsure, pb, pf)
        inb = (p > LO) & (p < HI)
        keep = inb | (rng.random(len(d)) < 0.05)
        out.append(d.loc[keep, ID].assign(p=p[keep], p_fast=pf[keep], inband=inb[keep],
                                          src=np.where(late[keep], "big", "fast2")))
        print(f"{f.name}: {len(d):,} pairs, m2_big on {unsure.sum():,} of {late.sum():,}, band {inb.sum():,} "
              f"({inb.mean():.3f}), skipped-in-band {miss[-1].mean():.4f} {time.time() - t0:.0f}s peak {peak():.2f} GB",
              flush=True)
        del d, pf, pb, p
        gc.collect()
    b = pd.concat(out, ignore_index=True)
    b.to_parquet(OUT / f"trainband_{country}.parquet", index=False)
    for nm, g in b.groupby("src"):
        ib = g[g.inband]
        print(f"{country} {nm}: {len(ib):,} band pairs (true {ib.is_true.mean():.3f}), {len(g) - len(ib):,} oob sample",
              flush=True)
    print(f"{country}: share of skipped pairs that m2_big puts in the band {np.concatenate(miss).mean():.5f}; "
          f"band done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def tokenize(tok, ta, tb, chunk: int = 50_000):
    """Pair texts -> (flat int32 ids, int64 offsets); P4b's truncation (longest_first, MAXLEN). Chunked so the
    Python-int lists never exceed one chunk."""
    ids, off = [], [np.zeros(1, np.int64)]
    for i in range(0, len(ta), chunk):
        e = tok(list(ta[i:i + chunk]), list(tb[i:i + chunk]), truncation="longest_first", max_length=MAXLEN,
                return_attention_mask=False)["input_ids"]
        ids.append(np.fromiter((t for x in e for t in x), np.int32))
        off.append(np.cumsum([len(x) for x in e]) + off[-1][-1])
        del e
    return np.concatenate(ids), np.concatenate(off)


def batch(ids, off, j, pad: int, dev: str = "cuda"):
    """Rows j of the flat token store -> padded (input_ids, attention_mask) tensors on dev."""
    import torch
    ln = (off[j + 1] - off[j]).astype(np.int64)
    L = int(ln.max())
    x = np.full((len(j), L), pad, np.int64)
    msk = np.arange(L)[None, :] < ln[:, None]
    x[msk] = ids[np.concatenate([np.arange(off[k], off[k + 1]) for k in j])]
    return torch.from_numpy(x).to(dev), torch.from_numpy(msk.astype(np.int64)).to(dev)


def data(oob: float = 0.1, nval: int = 10_000, seed: int = 0):
    """Train/val token stores from p11/trainband_<c>: band pairs of folds 3-9 (+ oob x as many out-of-band pairs);
    nval band pairs per country held out. tr_src = 1 for m2_big's folds 6-9. Cached in p11/tok_<key>.npz; e5-small
    and e5-base share the tokenizer (same 250,002-token vocab, identical ids), so one store serves both."""
    key = f"oob{oob}_s{seed}"
    f = OUT / f"tok_{key}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASES["small"])
    t0, parts = time.time(), {"tr": [], "va": []}
    for c in ("India", "US"):
        b = pd.read_parquet(OUT / f"trainband_{c}.parquet")
        ib = b[b.inband]
        va = ib.sample(nval, random_state=seed)
        ib = ib.drop(va.index)
        ob = b[~b.inband]
        tr = pd.concat([ib, ob.sample(min(len(ob), int(oob * len(ib))), random_state=seed)])
        for nm, d in (("tr", tr), ("va", va)):
            ta, tb = rr.texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            parts[nm].append((ta, tb, d.is_true.to_numpy(np.float32), d.p.to_numpy(np.float32),
                              (d.src == "big").to_numpy(np.int8)))
        del b, ib, ob, tr, va
        gc.collect()
    out = {}
    for nm, ps in parts.items():
        ta, tb, y, p, sb = (np.concatenate(x) for x in zip(*ps))
        out[f"{nm}_ids"], out[f"{nm}_off"] = tokenize(tok, ta, tb)
        out[f"{nm}_y"], out[f"{nm}_p"], out[f"{nm}_src"] = y, p, sb
        del ta, tb
        gc.collect()
    np.savez(f, **out)
    print(f"data {key}: {len(out['tr_y']):,} train (true {out['tr_y'].mean():.3f}), {len(out['va_y']):,} val, "
          f"{len(out['tr_ids']) / len(out['tr_y']):.1f} tokens/pair, {time.time() - t0:.0f}s, peak {peak():.2f} GB",
          flush=True)
    return out


def data_p4b(seed: int = 0):
    """#5's CE training set rebuilt exactly (P4b rerank.train: weak fold-2 LightGBM's 0.1<p<0.9 band of folds 4-9
    + 0.5x as many out-of-band pairs; val = 20k fold-3 band pairs), tokenised -> p11/tok_p4b.npz."""
    f = OUT / "tok_p4b.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASES["small"])
    t0, parts = time.time(), {"tr": [], "va": []}
    for c in ("India", "US"):
        b = pd.read_parquet(WORK / "p4b" / f"trainband_{c}.parquet")
        v = b[(b.fold == 3) & b.inband]
        b = b[b.fold >= 4]
        ib, ob = b[b.inband], b[~b.inband]
        b = pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=seed)])
        v = v.sample(min(len(v), 20_000), random_state=seed)
        for nm, d in (("tr", b), ("va", v)):
            ta, tb = rr.texts("train", c, d.q1.to_numpy(), d.qx.to_numpy())
            parts[nm].append((ta, tb, d.is_true.to_numpy(np.float32), d.p.to_numpy(np.float32)))
        del b, v, ib, ob
        gc.collect()
    out = {}
    for nm, ps in parts.items():
        ta, tb, y, p = (np.concatenate(x) for x in zip(*ps))
        out[f"{nm}_ids"], out[f"{nm}_off"] = tokenize(tok, ta, tb)
        out[f"{nm}_y"], out[f"{nm}_p"], out[f"{nm}_src"] = y, p, np.zeros(len(y), np.int8)
    np.savez(f, **out)
    print(f"data p4b: {len(out['tr_y']):,} train (true {out['tr_y'].mean():.3f}), {len(out['va_y']):,} val, "
          f"{time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    return out


def load_model(base: str, path=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASES[base])
    m = AutoModelForSequenceClassification.from_pretrained(BASES[base], num_labels=1)
    if path:
        m.load_state_dict({k: v.float() for k, v in torch.load(path).items()}, strict=False)
    return tok, m.cuda()


def score_tok(ids, off, m, pad: int, bs: int = 512) -> np.ndarray:
    """sigmoid(logit) per pre-tokenised pair, length-sorted batches."""
    import torch
    o = np.argsort(np.diff(off), kind="stable")
    r = np.empty(len(off) - 1, np.float32)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(o), bs):
            j = o[i:i + bs]
            x, msk = batch(ids, off, j, pad)
            r[j] = torch.sigmoid(m(input_ids=x, attention_mask=msk).logits.float().squeeze(-1)).cpu().numpy()
    return r


def train(tag: str, base: str = "small", epochs=1, src: str = "all", oob=0.1, n=0, bs=128, acc=1, lr=None,
          seed=0):
    """Fine-tune e5-<base> on the band (P4b's recipe: BCE, AdamW, warmup 500 + linear decay, frozen word
    embeddings, bf16 autocast, length buckets). bs = pairs per optimizer step, split into acc micro-batches.
    Saves p11/ce_<tag>.pt after the last epoch (and ce_<tag>_ep<k>.pt after earlier ones)."""
    import torch
    from sklearn.metrics import roc_auc_score
    epochs, oob, n, bs, acc, seed = int(epochs), float(oob), int(n), int(bs), int(acc), int(seed)
    lr = float(lr) if lr else {"small": 6e-5, "base": 3e-5}[base]
    t0 = time.time()
    D = data_p4b() if src == "p4b" else data(oob)       # src: all | big (m2_big's folds 6-9) | p4b (#5's set)
    ids, off, y = D["tr_ids"], D["tr_off"], D["tr_y"]
    ln = np.diff(off)
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(D["tr_src"] == 1) if src == "big" else np.arange(len(y))
    if n and n < len(rows):
        rows = np.sort(rng.choice(rows, n, replace=False))
    print(f"{tag}: {len(rows):,} train pairs (true {y[rows].mean():.3f}), {epochs} ep, bs {bs} (acc {acc}), lr {lr}, data {time.time() - t0:.0f}s, "
          f"peak {peak():.2f} GB", flush=True)
    torch.manual_seed(seed)
    tok, m = load_model(base)
    pad = tok.pad_token_id
    m.get_input_embeddings().weight.requires_grad_(False)     # 96M (small) / 192M (base) params: VRAM + disk
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * (len(rows) // bs)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * max(0.0, 1 - s / steps))
    lossf, step, mb, t1 = torch.nn.BCEWithLogitsLoss(), 0, bs // acc, time.time()
    log = []
    for ep in range(epochs):
        m.train()
        perm = rows[rng.permutation(len(rows))]  # length buckets: sort within 100-batch windows, shuffle the batches
        w = bs * 100
        perm = np.concatenate([c[np.argsort(ln[c], kind="stable")] for c in np.split(perm, range(w, len(perm), w))])
        batches = [perm[i:i + bs] for i in range(0, len(perm) - bs + 1, bs)]
        for k in rng.permutation(len(batches)):
            j = batches[k]
            for a in range(acc):
                jj = j[a * mb:(a + 1) * mb]
                x, msk = batch(ids, off, jj, pad)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = m(input_ids=x, attention_mask=msk).logits.float().squeeze(-1)
                loss = lossf(logit, torch.from_numpy(y[jj]).cuda()) / acc
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                el = time.time() - t1
                print(f"ep {ep} step {step}/{steps} loss {loss.item() * acc:.4f} {el:.0f}s {step * bs / el:.0f} pairs/s "
                      f"eta {(steps - step) * el / step / 60:.0f} min gpu {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                      flush=True)
        t2 = time.time()
        vr = score_tok(D["va_ids"], D["va_off"], m, pad)
        vy, vp = D["va_y"], D["va_p"]
        log.append({"ep": ep, "auc_gbdt": roc_auc_score(vy, vp), "auc_ce": roc_auc_score(vy, vr),
                    "auc_sum": roc_auc_score(vy, rr.logit(vp) + rr.logit(vr)),
                    "score_pairs_s": len(vy) / (time.time() - t2), "train_pairs_s": step * bs / (t2 - t1)})
        print(f"ep {ep}: " + " ".join(f"{k} {v:.4f}" for k, v in log[-1].items()), flush=True)
        name = f"ce_{tag}.pt" if ep == epochs - 1 else f"ce_{tag}_ep{ep}.pt"
        torch.save({k: v.half() for k, v in m.state_dict().items() if "word_embeddings" not in k}, OUT / name)
    json.dump({"tag": tag, "base": base, "epochs": epochs, "src": src, "oob": oob, "n_train": int(len(rows)), "bs": bs,
               "acc": acc, "lr": lr, "log": log, "train_s": round(time.time() - t0),
               "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "peak_gb": round(peak(), 2)},
              open(OUT / f"train_{tag}.json", "w"), indent=1)
    print(f"train {tag} done {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


PROB = P4 / "prob_holdout_big.parquet"


def holdout_band():
    d = pd.read_parquet(PROB, columns=["s1", "sx", "p", "country"])
    return d[(d.p > LO) & (d.p < HI) & d.country.isin(rr.CFG["countries"])].reset_index(drop=True)


def score(tag: str, ckpt: str = ""):
    """r of CE <tag> for every m2_big holdout band pair (193,751) -> p11/r_<tag>_holdout.parquet."""
    cfg = json.load(open(OUT / f"train_{tag}.json"))
    t0, out = time.time(), []
    d = holdout_band()
    tok, m = load_model(cfg["base"], OUT / (ckpt or f"ce_{tag}.pt"))
    for c, g in d.groupby("country"):
        q1, qx = rr.rows("train", c, g.s1, g.sx)
        ids, off = tokenize(tok, *rr.texts("train", c, q1, qx))
        t1 = time.time()
        r = score_tok(ids, off, m, tok.pad_token_id)
        print(f"{c}: {len(g):,} pairs scored at {len(g) / (time.time() - t1):.0f} pairs/s (GPU part), "
              f"peak {peak():.2f} GB", flush=True)
        out.append(pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "r": r}))
    nm = Path(ckpt).stem.removeprefix("ce_") if ckpt else tag
    pd.concat(out, ignore_index=True).to_parquet(OUT / f"r_{nm}_holdout.parquet", index=False)
    print(f"score {nm} done {time.time() - t0:.0f}s ({len(d) / (time.time() - t0):.0f} pairs/s end to end)", flush=True)


def _big():
    sys.path.insert(0, str(ROOT / "probes/P4-match"))
    import big
    return big


def eval_(tag: str):
    """P4's #5 yardstick (big._blend_eval on prob_holdout_big: blend w + D3 t/t0 fit on half 0, report half 1) for
    #5's CE (P4b cache) and CE <tag>; paired se of the per-S1 F0.5 difference on half 1."""
    from ledger import log
    big, t0 = _big(), time.time()
    ctl = big._blend_eval(big.PROB)
    rc = pd.read_parquet(OUT / f"r_{tag}_holdout.parquet")
    orig = big._r
    big._r = lambda ids: ids.merge(rc, on=["s1", "sx"], how="left").r.to_numpy(np.float32)
    try:
        new = big._blend_eval(big.PROB)
    finally:
        big._r = orig
    S = new[4]
    h1 = S.half.to_numpy() == 1
    dif = (new[3] - ctl[3])[h1]
    gain, se = float(dif.mean()), float(dif.std(ddof=1) / np.sqrt(len(dif)))
    cfg = json.load(open(OUT / f"train_{tag}.json")) if (OUT / f"train_{tag}.json").exists() else {}
    for nm, (m, c, w, _, _) in (("control", ctl), (tag, new)):
        print(f"{nm}: t={c['t']:.2f} t0={c['t0']:.2f} w={np.round(w, 3).tolist()}",
              {k: round(float(v), 5) for k, v in m.items()}, flush=True)
    print(f"{tag} vs #5 CE on half 1: {gain:+.5f} +- {se:.5f} (1 se), {(dif > 0).sum():,} S1 up, "
          f"{(dif < 0).sum():,} down", flush=True)
    m, c, w = new[0], new[1], new[2]
    res = {k: round(float(v), 5) for k, v in m.items()} | {"f05_control": round(float(ctl[0]["f05"]), 5),
                                                           "gain_vs_ce5": round(gain, 5), "gain_se": round(se, 5),
                                                           "s1_up": int((dif > 0).sum()), "s1_down": int((dif < 0).sum())}
    json.dump(res | {"t": c["t"], "t0": c["t0"], "w": w}, open(OUT / f"eval_{tag}.json", "w"), indent=1)
    log("exp", probe="P11-ce-big", subproblem="rerank", name=f"ce-{tag}", world="holdout-dense-half1",
        metrics=res | {f"val_{k}": round(v, 4) for k, v in (cfg.get("log") or [{}])[-1].items()},
        config={"t": c["t"], "t0": c["t0"], "blend_w": [round(x, 4) for x in w], "band": [LO, HI],
                "gbdt": "m2_big (prob_holdout_big)"} | {k: v for k, v in cfg.items() if k not in ("log", "tag")},
        hypothesis="a better-fed / bigger CE beats #5's e5-small CE by >= +0.0005 on the report half",
        verdict="keep" if gain >= 0.0005 and gain > 3 * se else "drop",
        runtime_s=round(cfg.get("train_s", 0) + time.time() - t0), peak_gb=round(peak(), 2))
    return res


W_BIG = [0.678, 0.715, 0.338]           # #5's blend weights (P4 big.py): inverts prob_test_<c>_big_blend -> #5's r


def test_r(tag: str, country: str, d: pd.DataFrame, b: np.ndarray, n: int = 0) -> np.ndarray:
    """r of CE <tag> for test rows b of prob_test_<c>_big (d). 'ce5' = #5's CE, recovered exactly by inverting #5's
    blend (p' = blend(p, r, W_BIG)); a trained tag is scored on the GPU and cached in p11/r_test_<tag>_<c>.parquet."""
    if tag == "ce5":
        b5 = pd.read_parquet(P4 / f"prob_test_{country}_big_blend.parquet", columns=["s1", "sx", "p", "p_gbdt"])
        assert (b5.s1.to_numpy() == d.s1.to_numpy()).all() and (b5.sx.to_numpy() == d.sx.to_numpy()).all(), "row order"
        assert np.allclose(b5.p_gbdt.to_numpy(), d.p.to_numpy()), "#5's GBDT p is m2_big's"
        pb, pg = (b5[k].to_numpy(np.float64)[b] for k in ("p", "p_gbdt"))
        del b5
        a, w, c = W_BIG
        r = (1 / (1 + np.exp(-(rr.logit(pb) - a * rr.logit(pg) - c) / w))).astype(np.float32)
        assert np.abs(rr.blend(pg.astype(np.float32), r, W_BIG, LO, HI) - pb).max() < 1e-4, "inversion reproduces p'"
        return r
    cache = OUT / f"r_test_{tag}_{country}.parquet"
    if cache.exists() and not n:
        rc = pd.read_parquet(cache)
        assert len(rc) == len(b) and (rc.row.to_numpy() == b).all(), "cache is for the same band rows"
        return rc.r.to_numpy(np.float32)
    cfg = json.load(open(OUT / f"train_{tag}.json"))
    t0 = time.time()
    q1, qx = rr.rows("test", country, d.s1.iloc[b], d.sx.iloc[b])
    tok, m = load_model(cfg["base"], OUT / f"ce_{tag}.pt")
    t1 = time.time()
    ids, off = tokenize(tok, *rr.texts("test", country, q1, qx))
    t2 = time.time()
    r = score_tok(ids, off, m, tok.pad_token_id)
    print(f"{country} {tag}: {len(b):,} band pairs; rows {t1 - t0:.0f}s, texts+tokens {t2 - t1:.0f}s, GPU {time.time() - t2:.0f}s "
          f"({len(b) / (time.time() - t2):.0f} pairs/s), total {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    if not n:
        pd.DataFrame({"row": b, "r": r}).to_parquet(cache, index=False)
    del m
    gc.collect()
    return r


def test(tag: str, country: str, n: int = 0):
    """Own process, GPU (flock): m2_big test p (p4/prob_test_<c>_big) with the band 0.02<p<0.98 rescored by CE <tag>
    (a single CE, or an 'ensemble' tag = logit mean of its parts, as in ens()) and blended with the weights fit on
    holdout half 0 (eval_<tag>.json) -> p11/prob_test_<c>_blend_<tag>.parquet [s1, sx, p = blended, p_gbdt].
    n > 0: time n band pairs of the first trained part only, write nothing (scoring-cost check)."""
    assert country in rr.CFG["countries"], "a country without CE training data keeps m2_big p (LOCO -0.0016)"
    cfg, n = json.load(open(OUT / f"train_{tag}.json")), int(n)
    parts = cfg["ensemble"].split(",") if "ensemble" in cfg else [tag]
    t0 = time.time()
    d = pd.read_parquet(P4 / f"prob_test_{country}_big.parquet", columns=["s1", "sx", "p"])
    p = d.p.to_numpy(np.float32)
    b = np.flatnonzero((p > LO) & (p < HI))
    if n:
        b = np.sort(np.random.default_rng(0).choice(b, min(n, len(b)), replace=False))
        test_r(next(t for t in parts if t != "ce5"), country, d, b, n)
        return
    L = np.mean([rr.logit(test_r(t, country, d, b)) for t in parts], 0)
    ev = json.load(open(OUT / f"eval_{tag}.json"))
    r = np.full(len(p), np.nan, np.float32)
    r[b] = 1 / (1 + np.exp(-L))
    d["p_gbdt"], d["p"] = p, rr.blend(p, r, ev["w"], LO, HI)
    d.to_parquet(OUT / f"prob_test_{country}_blend_{tag}.parquet", index=False)
    print(f"{country}: {len(d):,} pairs, {len(b):,} in band -> {OUT / f'prob_test_{country}_blend_{tag}.parquet'} "
          f"(tag {tag}, w {np.round(ev['w'], 3).tolist()}), {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def _check():
    """The flat token store + batch() rebuild exactly the tokenizer's own padded pair batch."""
    from transformers import AutoTokenizer
    ta = np.array(["abc stores | 12 main st", "x", "सरस्वती भवन | pune", "café du nord | 3 rue de la paix paris"], object)
    tb = np.array(["abc store | 12 main street springfield", "y z", "saraswati bhavan | pune", "cafe nord | paris"], object)
    for base in BASES:
        tok = AutoTokenizer.from_pretrained(BASES[base])
        ids, off = tokenize(tok, ta, tb, chunk=3)
        e = tok(list(ta), list(tb), truncation="longest_first", max_length=MAXLEN, padding=True, return_tensors="pt")
        x, msk = batch(ids, off, np.arange(len(ta)), tok.pad_token_id, dev="cpu")
        assert (x == e["input_ids"]).all() and (msk == e["attention_mask"]).all(), base
        x2, _ = batch(ids, off, np.array([2, 0]), tok.pad_token_id, dev="cpu")
        assert (x2[0, :int(msk[2].sum())] == e["input_ids"][2, :int(msk[2].sum())]).all(), "row subset"
    print("ce self-check ok")


def ens(tag: str, parts: str):
    """Logit-average ensemble of several CEs' holdout r ('ce5' = #5's CE) -> r_<tag>_holdout.parquet, so eval
    scores it like any single CE (same 2-weight blend)."""
    d = holdout_band()[["s1", "sx"]]
    L = []
    for t in parts.split(","):
        r = _big()._r(d) if t == "ce5" else \
            d.merge(pd.read_parquet(OUT / f"r_{t}_holdout.parquet"), on=["s1", "sx"], how="left").r.to_numpy(np.float32)
        assert not np.isnan(r).any(), t
        L.append(rr.logit(r))
    pd.DataFrame({"s1": d.s1, "sx": d.sx, "r": (1 / (1 + np.exp(-np.mean(L, 0)))).astype(np.float32)}).to_parquet(
        OUT / f"r_{tag}_holdout.parquet", index=False)
    json.dump({"tag": tag, "ensemble": parts}, open(OUT / f"train_{tag}.json", "w"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    pos = [a for a in sys.argv[2:] if "=" not in a]
    {"band": lambda: band(*pos), "data": lambda: data_p4b() if kw.get("src") == "p4b" else data(float(kw.get("oob", 0.1))), "train": lambda: train(**kw), "score": lambda: score(**kw),
     "eval": lambda: eval_(**kw), "test": lambda: test(country=pos[0], **kw),
     "check": _check, "ens": lambda: ens(**kw)}[cmd]()
