"""Cross-encoder reranker (P4b-rerank): intfloat/multilingual-e5-small (MIT, 118M params) fine-tuned on
"name | addr" of both records, rescoring M2's uncertain band; blended back into p before the decision.

    python src/rerank.py band <country>                                   # train-side band (LightGBM, fold 2 fit)
    flock /tmp/amc-gpu.lock python src/rerank.py train [countries=India,US tag= n=]  # -> data/work/p4b/ce<tag>.pt, 25 min
    flock /tmp/amc-gpu.lock python src/rerank.py score in.parquet out.parquet      # PIPELINE STEP: [ta, tb] -> [r]
    python src/rerank.py check                                            # blend-rule self-check

The blend (CFG): for pairs with 0.02 < p < 0.98 of a country the CE was trained on,
p' = sigmoid(0.636 logit p + 0.757 logit r + 0.355) (a logistic stack fitted on holdout half-0 band pairs;
run.py --split holdout refits and prints it). Every other pair keeps p: leave-one-country-out LOSES F0.5
(India -> US -0.0016), so France / any unseen label is not reranked -- it is not filtered either.
`score` runs in its own process (torch + CUDA = 1.4 GB RSS) under the GPU lock; run.py passes it the texts.
Texts are P3's cleaned (Latin, lower-case) name / addr (block.texts). q1 = row of data/work/p3/txt/<c>_s1, qx = row
of the concatenated <c>_sx_* files. Folds 0 (holdout) and 1 (M2's early stopping) are never trained on; fold 2
fits the band model, fold 3 is validation. The base weights come from the local Hugging Face cache (download once:
`HF_HUB_OFFLINE=0 python -c "from transformers import AutoModel; AutoModel.from_pretrained('intfloat/multilingual-e5-small')"`).
"""
import gc
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # pretrained weights from the local cache only: no network

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from block import TXT
from worlds import WORK

OUT = WORK / "p4b"
CE = OUT / "ce.pt"
BASE = "intfloat/multilingual-e5-small"
LO, HI = 0.1, 0.9            # the band the train pairs are drawn from
MAXLEN = 64                  # pair tokens p50 ~50 / p99 ~75; longest_first truncation drops the address tail
                             # (city/state) and keeps name + house number + street; GPU-bound, 64 vs 96 = ~1.5x
ID = ["q1", "qx", "is_true", "fold"]
CFG = {"band": (0.02, 0.98), "w": [0.636, 0.757, 0.355], "countries": ("India", "US")}
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def band(country: str, seed: int = 0):
    """P4-like LightGBM fitted on fold 2, predicting folds 3-9: keep its band + 5% of the rest (robustness to a
    different band definition at eval time). -> data/work/p4b/trainband_<country>.parquet"""
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
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


def score_file(inp: str, outp: str, model: str = str(CE)):
    """THE PIPELINE STEP (own process): parquet [ta, tb] -> parquet [r], same rows."""
    t0 = time.time()
    d = pd.read_parquet(inp)
    tok, m = load_model(model)
    r = score(d.ta.to_numpy(object), d.tb.to_numpy(object), tok, m) if len(d) else np.zeros(0, np.float32)
    pd.DataFrame({"r": r}).to_parquet(outp, index=False)
    print(f"rerank: {len(d):,} pairs scored in {time.time() - t0:.0f}s ({len(d) / max(time.time() - t0, 1e-9):.0f}/s), "
          f"peak {peak():.2f} GB", flush=True)


def train(epochs: int = 1, bs: int = 128, lr: float = 6e-5, oob: float = 0.5, seed: int = 0,
          countries=("India", "US"), tag: str = "", n: int = 0):
    """Band pairs of folds 4-9 (+ oob x as many out-of-band pairs) -> ce<tag>.pt; fold 3 band = validation.
    n > 0: at most n train pairs per country (a smoke test of this entry point, not a model)."""
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
        b = b.sample(min(len(b), int(n)), random_state=seed) if int(n) else b
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
    steps = max(epochs * (len(y) // bs), 1)
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
    {"band": lambda: band(sys.argv[2]), "train": lambda: train(**kw), "score": lambda: score_file(*sys.argv[2:5]),
     "check": _check}[cmd]()
