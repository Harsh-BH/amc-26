"""Full TRAIN candidates for the matcher: P3's recommended rule (block.CFG) for EVERY train S1, exactly as
full.py test builds the test candidates. So block_rank_s1 / block_rank_sx / via mean the same thing as on test,
and an SX's argmax can compete over all its S1 (the holdout export saw holdout S1 only).
Holdout S1 (fold 0) get exactly the pairs of data/work/p3/cands_holdout_full_* (same rule, same vectorizers).

    .venv/bin/python probes/P4-match/cands.py run India      # per chunk: reverse picks + folded per-S1 pool (resumable)
    .venv/bin/python probes/P4-match/cands.py export India   # -> data/work/p4/cands_train_full_<country>.parquet
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P3-block"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B
from full import TXT, WORK, peak_gb

OUT = WORK / "p4"
PICKS = OUT / "picks"
CFG = B.CFG


def run(country):
    PICKS.mkdir(parents=True, exist_ok=True)
    vecs = pickle.loads((WORK / "p3/vecs_hash.pkl").read_bytes())
    files = sorted(TXT.glob(f"{country}_sx_*.parquet"))
    offs = np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in files])
    pool_p, done_p = PICKS / f"{country}_pool.parquet", PICKS / f"{country}_pool.json"
    done = json.loads(done_p.read_text()) if done_p.exists() else []
    pool = None
    if done:
        d = pd.read_parquet(pool_p)
        pool = (d.q1.to_numpy(), d.qx.to_numpy(), d.sc.to_numpy())
    todo = [(o, p) for o, p in zip(offs, files) if p.stem not in done]
    if not todo:
        return
    t0 = time.time()
    ix = B.index(pd.read_parquet(TXT / f"{country}_s1.parquet", columns=["name", "addr", "st"]), vecs, CFG)
    print(f"{country}: index {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)
    kr0, kna = CFG["keep_r"], max(CFG["keep_r"], CFG["k_noaddr"])
    for off, p in todo:
        t0 = time.time()
        tb = pd.read_parquet(p, columns=["name", "addr", "st"])
        noaddr = (tb.addr == "").to_numpy(bool)
        for i in range(0, len(tb), CFG["shard"]):
            h = B.search(ix, tb.iloc[i:i + CFG["shard"]].reset_index(drop=True), vecs, CFG)
            q1, qxl = h.q1.to_numpy(np.int32), h.qx.to_numpy(np.int64) + i
            sc, rk = h.score.to_numpy(np.float32), h["rank"].to_numpy()
            del h
            m = rk < np.where(noaddr[qxl], kna, kr0)
            pd.DataFrame({"q1": q1[m], "qx": qxl[m] + off, "sc": sc[m]}).to_parquet(
                PICKS / f"{p.stem}_{i:07d}.parquet", index=False)
            new = (q1, qxl + off, sc)
            pool = B._best(*map(np.concatenate, zip(pool, new)), CFG["keep_f"]) if pool else B._best(*new, CFG["keep_f"])
            del q1, qxl, sc, rk, new
        pd.DataFrame({"q1": pool[0], "qx": pool[1], "sc": pool[2]}).to_parquet(pool_p.with_suffix(".tmp"), index=False)
        pool_p.with_suffix(".tmp").rename(pool_p)
        done.append(p.stem)
        done_p.write_text(json.dumps(done))
        print(f"  {p.stem}: {len(tb):,} SX in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


def export(country, parts=4):
    """Union + ranks (block.finalize) -> q1 / qx (row in txt/<country>_s1 / concatenated txt/<country>_sx_*),
    block features, via, fold, is_true. Ids stay in the text files (mapped in arrow, never as Python strings)."""
    import pyarrow.compute as pc
    files = sorted(TXT.glob(f"{country}_sx_*.parquet"))
    r = pd.concat([pd.read_parquet(p) for f in files for p in sorted(PICKS.glob(f"{f.stem}_*.parquet"))],
                  ignore_index=True)
    d = pd.read_parquet(PICKS / f"{country}_pool.parquet")
    # finalize per SX range (RAM): union, via, block_score and the SX-side rank only involve one SX's pairs;
    # the S1-side rank is recomputed over all parts afterwards.
    cut = np.quantile(r.qx.to_numpy(), np.linspace(0, 1, parts + 1)[1:-1]).astype(np.int64) if parts > 1 else []
    edges = [-1, *cut, np.iinfo(np.int64).max]
    gs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mr, md = (r.qx.to_numpy() > lo) & (r.qx.to_numpy() <= hi), (d.qx.to_numpy() > lo) & (d.qx.to_numpy() <= hi)
        gs.append(B.finalize([(r.q1.to_numpy(np.int64)[mr] << 32) | r.qx.to_numpy(np.int64)[mr]], [r.sc.to_numpy()[mr]],
                             [(d.q1.to_numpy()[md], d.qx.to_numpy()[md], d.sc.to_numpy()[md])]))
    del r, d
    g = pd.concat(gs, ignore_index=True)
    del gs
    o = np.lexsort((g.qx.to_numpy(), g.q1.to_numpy()))           # finalize's order: by key = (q1, qx)
    g = g.iloc[o].reset_index(drop=True)
    o = np.lexsort((-g.block_score.to_numpy(), g.q1.to_numpy()))
    rr = np.empty(len(g), np.int32)
    rr[o] = B.ranks(g.q1.to_numpy()[o])
    g["block_rank_s1"] = rr
    del o, rr
    ids1 = pq.read_table(TXT / f"{country}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    idx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files])
    fo = pq.read_table(WORK / "s1_folds.parquet", columns=["entity_id", "fold"])
    fold = fo.column("fold").to_numpy()[pc.index_in(ids1, value_set=fo.column("entity_id").combine_chunks()).to_numpy()]
    gt = pq.read_table(WORK / "gt_pairs.parquet")
    a = pc.index_in(gt.column("s1"), value_set=ids1).to_numpy(zero_copy_only=False)
    b = pc.index_in(gt.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
    del gt, fo, idx
    ok = ~np.isnan(a)
    assert not np.isnan(b[ok]).any(), "every true SX of a country S1 is in the country corpus"
    tk = (a[ok].astype(np.int64) << 32) | b[ok].astype(np.int64)
    tf = fold[a[ok].astype(np.int64)]
    q1, qx = g.q1.to_numpy(np.int64), g.qx.to_numpy(np.int64)
    pk = (q1 << 32) | qx                                          # sorted: g is in key order
    assert (np.diff(pk) > 0).all()
    tks = np.sort(tk)
    pos = np.minimum(np.searchsorted(tks, pk), len(tks) - 1)
    true = tks[pos] == pk                                         # np.isin sorts both sides: +0.6 GB here
    del tks, pos
    t = pa.table({"q1": q1.astype(np.int32), "qx": qx.astype(np.int32),
                  "block_score": g.block_score.to_numpy(), "block_rank_s1": g.block_rank_s1.to_numpy(),
                  "block_rank_sx": g.block_rank_sx.to_numpy(), "via": pa.array(g.via.astype(str).to_numpy()),
                  "fold": fold[q1], "is_true": true})
    pq.write_table(t, OUT / f"cands_train_full_{country}.parquet")
    meta = {"country": country, "pairs": len(g), "n_s1": len(ids1), "cands_per_s1": len(g) / len(ids1),
            "pair_recall": float(true.sum() / len(tk)), "true_share": float(true.mean()),
            "pair_recall_fold0": float((pk[np.minimum(np.searchsorted(pk, tk[tf == 0]), len(pk) - 1)] == tk[tf == 0]).mean()),
            "peak_gb": peak_gb()}
    (OUT / f"cands_train_full_{country}.json").write_text(json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    {"run": run, "export": export}[sys.argv[1]](sys.argv[2])
