"""Does #9ab's France survive a from-raw rebuild under P21's learned région map (normalize.fr_departements)?
A fresh idf144 world (data/work/p9/fr_idf144_fresh): only the raw sources, train-side fits and models are linked.
France's texts, rules.feats, département map, IDF, blocking, X and M2 are all recomputed with the current src. Then
#9ab's France decision (bge blend w_france on 0.02 < p < 0.98, argmax, t 0.985) is compared with output/sub9ab.

    .venv/bin/python probes/P9-integrate/p21check.py
"""
import csv
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = ["code/business_entity_resolution/src", "."]
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import block as B
import features as F
import model as M
import prune
import rerank as RR
import run
from worlds import WORK

SRC = Path("code/business_entity_resolution/src")
REF = Path("/home/harsh1/github-repos/amc-26/output/sub9ab")


def sub(v: Path, *args):
    subprocess.run([sys.executable, *args], env={**os.environ, "ER_WORK": str(v)}, check=True)


if __name__ == "__main__":
    t0 = time.time()
    run.use("sub9ab")
    v = M.CACHE / "fr_idf144_fresh"
    for rel in ["test_source1.parquet", "test_source2.parquet", "test_source3.parquet", "train_source1.parquet",
                "place_freq_train.json", "p2_xlit_table.parquet", "gt_pairs.parquet", "s1_folds.parquet",
                "rules_s1feat_train.parquet", str(M.BOOSTER.relative_to(WORK)), str(prune.MODEL.relative_to(WORK))]:
        d = v / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            d.symlink_to((WORK / rel).resolve())
    # 1) texts + rules.feats + the learned département map, in the fresh world
    sub(v, "-c", "import sys; sys.path.insert(0, 'code/business_entity_resolution/src'); import run\n"
                 "if __name__ == '__main__':\n    run.prepare('test', 'France')")
    print(f"texts + rules.feats rebuilt, {time.time() - t0:.0f}s", flush=True)
    # 2) the idf144 vectorizers from the fresh texts
    vp = v / "p3/vecs_hash.pkl"
    if not vp.exists():
        vec = B.vecs()                                   # the train fit (this world's parent)
        tx = pd.concat([pq.read_table(f, columns=["name", "addr"]).to_pandas()
                        for f in sorted((v / "p3/txt_test").glob("France_sx_*.parquet"))], ignore_index=True)
        vp.write_bytes(pickle.dumps(B.idf_country(vec, tx, 0.144)))
    # 3) blocking -> X -> prune -> M2
    sub(v, str(SRC / "run.py"), "--split", "test", "--probs-only", "France", "--recipe", "sub9ab")
    print(f"France world rebuilt, {time.time() - t0:.0f}s", flush=True)
    # 4) compare with the cached world, by entity ids
    def ids_of(w: Path):
        s1 = pq.read_table(w / "p3/txt_test/France_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
        sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks()
                               for f in sorted((w / "p3/txt_test").glob("France_sx_*.parquet"))])
        return s1, sx
    old_w = M.CACHE / "fr_idf144"
    name = M.probs_path("test", "France").name
    res = {}
    for tag, w in (("old", old_w), ("fresh", v)):
        d = pd.read_parquet(w / "p9" / name)
        s1, sx = ids_of(w)
        d["s1"] = s1.take(pa.array(d.q1.to_numpy())).to_numpy(zero_copy_only=False)
        d["sx"] = sx.take(pa.array(d.qx.to_numpy())).to_numpy(zero_copy_only=False)
        res[tag] = d
    a, b = res["old"].set_index(["s1", "sx"]).p, res["fresh"].set_index(["s1", "sx"]).p
    both = a.index.intersection(b.index)
    dp = (a.loc[both] - b.loc[both]).abs()
    lo, hi = RR.CFG["band"]
    print({"kept_old": len(a), "kept_fresh": len(b), "only_old": len(a.index.difference(b.index)),
           "only_fresh": len(b.index.difference(a.index)), "p_moved": int((dp > 0).sum()), "max_dp": float(dp.max()),
           "band_old": int(((a > lo) & (a < hi)).sum()), "band_fresh": int(((b > lo) & (b < hi)).sum())}, flush=True)
    # 5) #9ab's France decision on the fresh world (r from the France bge cache, keyed by the main world's rows)
    f = res["fresh"]
    id1, idx = F.ids("test", "France")
    q1 = pd.Index(id1.to_pandas()).get_indexer(f.s1)
    qx = pd.Index(idx.to_pandas()).get_indexer(f.sx)
    assert (q1 >= 0).all() and (qx >= 0).all()
    cache = run.r_cache("test", "France", id1, idx)
    r_all = pd.read_parquet(cache)
    rr = pd.Series(r_all.r.to_numpy(np.float32), index=r_all.k.to_numpy(np.int64))
    pm = f.p.to_numpy(np.float32)
    band = (pm > lo) & (pm < hi)
    r = np.full(len(pm), np.nan, np.float32)
    r[band] = rr.reindex((q1[band].astype(np.int64) << 32) | qx[band]).to_numpy(np.float32)
    miss = int(np.isnan(r[band]).sum())
    p = RR.blend(pm, r, run.W_FRANCE, lo, hi)
    acc = M.argmax_sx(qx, p) & (p >= run.T_FRANCE_OVERRIDE)
    new = f[acc].groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
    cand = f.groupby("s1").sx.agg(lambda s: ",".join(sorted(s))).to_dict()
    fr = set(id1.to_pylist())
    ref = pd.read_csv(REF / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    ref = dict(zip(ref.source1_entity_id, ref.matched_entity_ids))
    refc = pd.read_csv(REF / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    refc = dict(zip(refc.source1_entity_id, refc.candidate_entity_ids))
    dm = sum(new.get(s, "") != ref[s] for s in fr)
    dc = sum(cand.get(s, "") != refc[s] for s in fr)
    print(f"band pairs without a cached bge r: {miss:,} (kept p_gbdt)", flush=True)
    print(f"France S1 whose matching line differs from output/sub9ab: {dm:,} of {len(fr):,}; candidate lines: {dc:,}",
          flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
