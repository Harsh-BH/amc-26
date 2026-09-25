"""Synthetic world for a target country: S2/S3 records generated from the target's TRAIN S1 (folds >= 2) with the
SOURCE country's noise rates, then run through P3 blocking + P4 rules/features exactly as P4's featrun.py does for
real pairs (same vectorizers, same CFG, same rule stage, same context computation, same rounding).

    world.py gen   India US      # synth.generate -> data/work/p12/India_from_US/{sx_raw,truth,meta}
    world.py rf    India US      # rules._feats + P3 texts of the synthetic SX
    world.py block India US      # P3 country_pairs over the pool, rule-accepted pairs, S1-side context
    world.py feat  India US      # -> data/work/p12/X/India_from_US_NNN.parquet (P4 X format: q1, qx, is_true, fold, 65)
    world.py all   India US      # the four steps, each in its own process (RAM)

Pool = every target S1 of folds >= 2 (blocking index, chain counts, rule stage): the eval folds 0/1 are never read.
Sources = the pool S1 of a seeded random subset of state groups covering FRAC of the pool; only they get synthetic
copies, and training keeps only pairs whose S1 is a source (their S1-side context is complete: chain siblings share
a state, and the blocker searches within the state group).
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P4 = ROOT.parent / "p4-match/probes/P4-match"
sys.path[:0] = [str(HERE), str(P4), str(ROOT / "probes/P3-block"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B
import features as F
import rules
import synth

WORK = ROOT / "data/work"
OUT = WORK / "p12"
RF = ["id", "country", "core", "web", "is_web", "state", "city", "hn", "places"]
FRAC = float(os.environ.get("P12_FRAC", 0.35))
CHUNK = 300_000                                   # SX rows per feature file
CFG = {**B.CFG, "shard": 150_000, "threads": 8}   # shard / threads are RAM / CPU knobs only (same pairs)


def peak_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def trim():
    import ctypes
    import pyarrow as pa
    pa.default_memory_pool().release_unused()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def d(tgt, src):
    p = OUT / f"{tgt}_from_{src}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pool(tgt) -> pd.DataFrame:
    """Pool S1 texts (P3 txt order, folds >= 2) + fold."""
    t = pd.read_parquet(WORK / f"p3/txt/{tgt}_s1.parquet")
    fo = pd.read_parquet(WORK / "s1_folds.parquet").set_index("entity_id").fold
    t["fold"] = fo.reindex(t.entity_id).to_numpy().astype(np.int8)
    return t[t.fold >= 2].reset_index(drop=True)


def gen(tgt, src, seed=0):
    t0 = time.time()
    P = pool(tgt)
    rng = np.random.default_rng(seed)
    st = P.st.astype(str)
    share = st.value_counts(normalize=True)
    groups = [g for g in rng.permutation(share.index.to_numpy()) if g != ""]
    chosen, acc = [], 0.0
    for g in groups:
        if acc >= FRAC:
            break
        chosen.append(g)
        acc += share[g]
    src_mask = st.isin(chosen).to_numpy() | ((st == "").to_numpy() & (rng.random(len(P)) < FRAC))
    ids = P.entity_id[src_mask]
    raw = pd.read_parquet(WORK / "train_source1.parquet", filters=[("country", "==", tgt)])
    raw = raw.set_index("entity_id").loc[ids].reset_index()
    rates = json.loads((HERE / f"rates_{src}.json").read_text())
    sx, truth = synth.generate(raw, rates, seed=seed, prefix=f"p12{tgt[:2].lower()}")
    o = d(tgt, src)
    sx.to_parquet(o / "sx_raw.parquet", index=False)
    truth.to_parquet(o / "truth.parquet", index=False)
    pd.DataFrame({"entity_id": ids}).to_parquet(o / "sources.parquet", index=False)
    meta = {"tgt": tgt, "src": src, "pool": len(P), "sources": int(src_mask.sum()), "states": chosen,
            "state_share": round(acc, 4), "sx": len(sx), "true_pairs": len(truth),
            "orphan_share": round(1 - truth.sx.nunique() / len(sx), 4), "sx_per_source": round(len(sx) / src_mask.sum(), 3),
            "seed": seed, "runtime_s": round(time.time() - t0, 1)}
    (o / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def rf(tgt, src):
    o = d(tgt, src)
    t0 = time.time()
    sx = pd.read_parquet(o / "sx_raw.parquet")
    rules.freq()
    parts, txt = [], []
    for i in range(0, len(sx), 100_000):
        c = sx.iloc[i:i + 100_000]
        parts.append(rules._feats(c))
        txt.append(B.texts(c, B.CFG))
    pd.concat(parts, ignore_index=True).to_parquet(o / "rf.parquet", index=False)
    pd.concat(txt, ignore_index=True).to_parquet(o / "txt.parquet", index=False)
    print(f"rf {tgt}<-{src}: {len(sx):,} SX in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)




def pool_rf(tgt, P) -> pd.DataFrame:
    r = pd.read_parquet(WORK / "rules_s1feat_train.parquet", filters=[("country", "==", tgt)])
    r = r.set_index("id").reindex(P.entity_id.rename("id")).reset_index()
    assert r.country.notna().all(), "rule features for every pool S1"
    return r[RF]


def block(tgt, src):
    o = d(tgt, src)
    t0 = time.time()
    P = pool(tgt)
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    tx = pd.read_parquet(o / "txt.parquet")
    chunks = ((off, tx.iloc[off:off + CFG["shard"]].reset_index(drop=True)) for off in range(0, len(tx), CFG["shard"]))
    g = B.country_pairs(P[["name", "addr", "st"]], chunks, vecs, CFG)
    del P
    trim()
    print(f"block {tgt}<-{src}: {len(g):,} pairs in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
    s1c = F._group_ctx(g.q1.to_numpy(), np.nan_to_num(g.block_score.to_numpy(np.float32)), "s1")
    for k, v in s1c.items():
        g[k] = v
    g["via"] = g.via.astype(str)
    g.to_parquet(o / "pairs.parquet", index=False)
    del g
    trim()
    P = pool(tgt)
    r1 = pool_rf(tgt, P)
    rx = pd.read_parquet(o / "rf.parquet")[RF]
    acc, _ = rules.match(r1, rx, chunk=100_000, by_state=True, keep_cands=False)
    q1 = pd.Index(P.entity_id).get_indexer(acc.s1)
    qx = pd.Index(rx.id).get_indexer(acc.sx)
    assert (q1 >= 0).all() and (qx >= 0).all()
    pd.DataFrame({"q1": q1.astype(np.int32), "qx": qx.astype(np.int32)}).to_parquet(o / "acc.parquet", index=False)
    print(f"rules {tgt}<-{src}: {len(acc):,} accepted in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


def _key(a, b):
    return (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)


def feat(tgt, src):
    import pyarrow as pa
    import pyarrow.parquet as pq
    o = d(tgt, src)
    xd = OUT / "X"
    xd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = pool(tgt)
    fold1 = P.fold.to_numpy()
    sxraw = pd.read_parquet(o / "sx_raw.parquet", columns=["entity_id", "business_name"])
    truth = pd.read_parquet(o / "truth.parquet")
    tk = np.sort(_key(pd.Index(P.entity_id).get_indexer(truth.s1), pd.Index(sxraw.entity_id).get_indexer(truth.sx)))
    assert (tk >> 32 >= 0).all()
    r1 = F.assemble(pool_rf(tgt, P), P[["name", "addr", "st"]], pd.Series("", index=P.index))
    del P
    trim()
    m1, counts = F.mats(r1), F.pool_counts(r1)
    rfx, txx = pd.read_parquet(o / "rf.parquet"), pd.read_parquet(o / "txt.parquet")
    pairs = pd.read_parquet(o / "pairs.parquet").sort_values(["qx", "q1"], kind="stable").reset_index(drop=True)
    ak = np.sort(_key(*pd.read_parquet(o / "acc.parquet").to_numpy().T))
    print(f"feat {tgt}<-{src}: pool {len(r1):,}, {len(pairs):,} pairs ready in {time.time() - t0:.0f}s, "
          f"peak {peak_gb():.2f} GB", flush=True)
    cut = np.searchsorted(pairs.qx.to_numpy(), np.arange(0, len(rfx) + CHUNK, CHUNK))
    for k, lo in enumerate(range(0, len(rfx), CHUNK)):
        p = xd / f"{tgt}_from_{src}_{k:03d}.parquet"
        if p.exists():
            continue
        t1 = time.time()
        hi = min(lo + CHUNK, len(rfx))
        rx = F.assemble(rfx.iloc[lo:hi], txx.iloc[lo:hi], sxraw.business_name.iloc[lo:hi])
        mx = F.mats(rx)
        X = pairs.iloc[cut[k]:cut[k + 1]].reset_index(drop=True)
        b = F.block_cols(X)
        X = pd.concat([X[["q1", "qx", "s1_n", "s1_gap", "s1_margin"]], pd.DataFrame(
            b | F._group_ctx(X.qx.to_numpy(), np.nan_to_num(b["block_score"]), "sx"))], axis=1)
        ak_c = ak[((ak & 0xFFFFFFFF) >= lo) & ((ak & 0xFFFFFFFF) < hi)]
        extra = np.setdiff1d(ak_c, _key(X.q1, X.qx))
        if len(extra):
            X = pd.concat([X, pd.DataFrame({"q1": (extra >> 32).astype(np.int32),
                                            "qx": (extra & 0xFFFFFFFF).astype(np.int32), "blocked": np.float32(0)})],
                          ignore_index=True)
        kk = _key(X.q1, X.qx)
        X.insert(2, "is_true", tk[np.minimum(np.searchsorted(tk, kk), len(tk) - 1)] == kk)
        X.insert(3, "fold", fold1[X.q1.to_numpy()])
        acc_pair = np.isin(kk, ak_c)
        i, j = X.q1.to_numpy(), X.qx.to_numpy() - lo
        ids_ = ["q1", "qx", "is_true", "fold"]
        fc = [c for c in X.columns if c not in ids_]
        w = None
        for a in range(0, len(X), 250_000):
            S = F.strings(r1, rx, i[a:a + 250_000], j[a:a + 250_000], counts, m1, mx)
            S["rule_acc"] = acc_pair[a:a + 250_000].astype(np.float32)
            part = X.iloc[a:a + 250_000].reset_index(drop=True)
            out = pd.concat([part[ids_], part[fc].astype(np.float32).round(3), S.round(3)], axis=1)
            t = pa.Table.from_pandas(out, preserve_index=False)
            w = w or pq.ParquetWriter(p.with_suffix(".tmp"), t.schema, compression="zstd")
            w.write_table(t)
            del S, part, out, t
        w.close()
        p.with_suffix(".tmp").rename(p)
        print(f"  {p.stem}: {hi - lo:,} SX, {len(X):,} pairs (+{len(extra):,} rule-only), true {X.is_true.mean():.3f} "
              f"in {time.time() - t1:.0f}s, peak {peak_gb():.2f} GB", flush=True)
        del X, rx, mx
        trim()


def all_(tgt, src):
    for step in ("gen", "rf", "block", "feat"):
        r = subprocess.run([sys.executable, __file__, step, tgt, src])
        if r.returncode:
            raise SystemExit(f"{step} failed ({r.returncode})")


if __name__ == "__main__":
    {"gen": gen, "rf": rf, "block": block, "feat": feat, "all": all_}[sys.argv[1]](*sys.argv[2:])
