"""Full-scale P4 featurization, per split (train | test) and country, resumable per P3 SX text file.
Every step is its own process; the S1 pool side stays in RAM, the SX side streams one text file (~0.5M rows).

    featrun.py rf   train India     # rules._feats for every train SX file -> data/work/p4/rf/train_India_sx_NNN.parquet
    featrun.py ctx  train India     # candidate-list context (needs ALL pairs) -> data/work/p4/ctx/train_India_sx_NNN.parquet
    featrun.py feat train India     # all features -> data/work/p4/X/train_India_sx_NNN.parquet (q1, qx, [is_true, fold], X)

Candidates: train = data/work/p4/cands_train_full_<c> (cands.py), test = data/work/p3/cands_test_<c> (P3), both the
same P3 rule over the whole S1 pool, plus the rule-accepted pairs P3 missed (rules.match by_state, the validated
stage 1). q1 = row in the P3 S1 text file, qx = row in the concatenated P3 SX text files of the country.
SX rule features come from the shared rules.feats caches data/work/rf_{train,test}_sx_<c>.parquet (same row order
as the P3 texts, checked); test S1 from rf_test_s1_<c>, train S1 from rules_s1feat_train. `rf` recomputes them.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")   # plain malloc, so trim() can hand RAM back

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P3-block"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import features as F
import rules
from full import peak_gb

WORK = ROOT / "data/work"
OUT = WORK / "p4"
TXT = {"train": WORK / "p3/txt", "test": WORK / "p3/txt_test"}
COLS = ["entity_id", "business_name", "business_address", "country"]
RF = ["id", "country", "core", "web", "is_web", "state", "city", "hn", "places"]


def sx_files(split, c):
    files = sorted(TXT[split].glob(f"{c}_sx_*.parquet"))
    return files, np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in files])


def raw_sx(split, c):
    """(txt file, raw rows in its order): P3 prep streamed source2 then source3 in 250k batches, country-filtered,
    and cut a file at whole batches -- the same stream re-cut at the file sizes gives the same rows."""
    files, offs = sx_files(split, c)
    sizes, buf, k = np.diff(offs), [], 0
    for src in ("source2", "source3"):
        for rb in pq.ParquetFile(WORK / f"{split}_{src}.parquet").iter_batches(250_000, columns=COLS):
            d = rb.to_pandas()
            buf.append(d[d.country == c])
            while k < len(files) and sum(map(len, buf)) >= sizes[k]:
                a = pd.concat(buf, ignore_index=True)
                ids = pq.read_table(files[k], columns=["entity_id"]).column(0).to_pandas()
                assert (a.entity_id.iloc[:sizes[k]].to_numpy() == ids.to_numpy()).all(), files[k]
                yield files[k], a.iloc[:sizes[k]].reset_index(drop=True)
                buf, k = [a.iloc[sizes[k]:]], k + 1
    assert k == len(files)


def pieces(path, sizes):
    """Consecutive row ranges of a parquet file as DataFrames (streamed; never the whole file in RAM)."""
    import pyarrow as pa
    it, buf = pq.ParquetFile(path).iter_batches(100_000), None
    for n in sizes:
        while buf is None or buf.num_rows < n:
            b = pa.Table.from_batches([next(it)])
            buf = b if buf is None else pa.concat_tables([buf, b])
        yield buf.slice(0, n).to_pandas()
        buf = buf.slice(n)


def rss_gb():
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1e6


def trim():
    """Hand freed heap back to the OS (RSS stays at the transient peak otherwise)."""
    import ctypes
    import pyarrow as pa
    pa.default_memory_pool().release_unused()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def s1_table(split, c):
    """Record table of the whole S1 pool of a country (P3 text order) + its rule features."""
    t = pd.read_parquet(TXT[split] / f"{c}_s1.parquet")
    src = WORK / ("rules_s1feat_train.parquet" if split == "train" else f"rf_test_s1_{c}.parquet")
    rf = pd.read_parquet(src, filters=[("country", "==", c)])
    rf = rf.set_index("id").reindex(t.entity_id.rename("id")).reset_index()
    assert rf.country.notna().all(), "rule features for every S1"
    return F.assemble(rf[RF], t, pd.Series("", index=t.index))    # S1 is always Latin: indic only matters on SX


def rf_step(split, c, procs=0):
    """procs=0: in-process, one chunk at a time (~0.7 GB); procs=n: rules.feats' forkserver pool (~0.5 GB/worker)."""
    (OUT / "rf").mkdir(parents=True, exist_ok=True)
    procs = int(procs)
    for f, raw in raw_sx(split, c):
        p = OUT / "rf" / f"{split}_{f.stem}.parquet"
        if not p.exists():
            t0 = time.time()
            rf = (pd.concat([rules._feats(raw.iloc[i:i + 100_000]) for i in range(0, len(raw), 100_000)],
                            ignore_index=True) if procs == 0 else rules.feats(raw, procs=procs))
            rf.to_parquet(p, index=False)
            print(f"  rf {f.stem}: {len(raw):,} in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


BLOCK = ["block_score", "block_rank_s1", "block_rank_sx", "via"]


def cands(split, c) -> pd.DataFrame:
    """Candidate pairs as int rows (q1, qx) + block columns (+ is_true, fold on train). Ids are mapped in arrow."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if split == "train":
        return pd.read_parquet(OUT / f"cands_train_full_{c}.parquet", columns=["q1", "qx", *BLOCK, "is_true", "fold"])
    t = pq.read_table(WORK / f"p3/cands_test_{c}.parquet")
    ids1 = pq.read_table(TXT[split] / f"{c}_s1.parquet", columns=["entity_id"]).column(0)
    files, _ = sx_files(split, c)
    idx = pa.chunked_array([pq.read_table(p, columns=["entity_id"]).column(0).combine_chunks() for p in files])
    q1 = pc.index_in(t.column("s1"), value_set=ids1.combine_chunks()).to_numpy()
    qx = pc.index_in(t.column("sx"), value_set=idx.combine_chunks()).to_numpy()
    assert not (np.isnan(q1.astype(float)).any() or np.isnan(qx.astype(float)).any()), "every candidate id is a record"
    d = t.select(BLOCK).to_pandas()
    return d.assign(q1=q1.astype(np.int32), qx=qx.astype(np.int32))


def ctx_step(split, c):
    """S1-side context over the whole candidate table (an S1's pairs span every SX file), rows re-cut by SX file.
    The SX side is computed per file in feat_step: a file holds every pair of its SX."""
    (OUT / "ctx").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = cands(split, c)
    d["via"] = d.via.astype("category")
    s1c = F._group_ctx(d.q1.to_numpy(), np.nan_to_num(d.block_score.to_numpy(np.float32)), "s1")
    o = np.argsort(d.qx.to_numpy(), kind="stable")
    d = d.iloc[o].reset_index(drop=True)
    for k, v in s1c.items():
        d[k] = v[o]
    del s1c, o
    files, offs = sx_files(split, c)
    cut = np.searchsorted(d.qx.to_numpy(), offs)
    for k, f in enumerate(files):
        d.iloc[cut[k]:cut[k + 1]].to_parquet(OUT / "ctx" / f"{split}_{f.stem}.parquet", index=False)
    print(f"ctx {split} {c}: {len(d):,} pairs in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB", flush=True)


def _key(a, b):
    return (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)


def feat_step(split, c, chunk=250_000):
    (OUT / "X").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fold1 = tk = None
    if split == "train":      # truth as sorted int keys (q1 << 32 | qx), ids mapped in arrow; first, while RAM is empty
        import pyarrow as pa
        import pyarrow.compute as pc
        ids1 = pq.read_table(TXT[split] / f"{c}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
        files, _ = sx_files(split, c)
        idx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files])
        gt = pq.read_table(WORK / "gt_pairs.parquet")
        a = pc.index_in(gt.column("s1"), value_set=ids1).to_numpy(zero_copy_only=False)
        b = pc.index_in(gt.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
        ok = ~np.isnan(a)
        tk = np.sort(_key(a[ok].astype(np.int64), b[ok].astype(np.int64)))
        fo = pq.read_table(WORK / "s1_folds.parquet")
        fold1 = fo.column("fold").to_numpy()[pc.index_in(ids1, value_set=fo.column("entity_id").combine_chunks())
                                             .to_numpy()].astype(np.int8)
        del gt, fo, idx, a, b, ids1
        trim()
    r1 = s1_table(split, c)
    trim()
    m1, counts = F.mats(r1), F.pool_counts(r1)
    ix1 = pd.Index(r1.id)
    trim()
    print(f"feat {split} {c}: S1 pool {len(r1):,} ready in {time.time() - t0:.0f}s, peak {peak_gb():.2f} GB, "
          f"rss {rss_gb():.2f} GB", flush=True)
    files, offs = sx_files(split, c)
    rfs = pieces(WORK / f"rf_{split}_sx_{c}.parquet", np.diff(offs))   # rules.feats caches, in P3 text order
    for k, (f, raw) in enumerate(raw_sx(split, c)):
        rf = next(rfs)
        p = OUT / "X" / f"{split}_{f.stem}.parquet"
        if p.exists():
            continue
        t0 = time.time()
        off = offs[k]
        assert (rf.id.to_numpy() == raw.entity_id.to_numpy()).all()
        rx = F.assemble(rf[RF], pd.read_parquet(f), raw.business_name)
        mx = F.mats(rx)
        acc, _ = rules.match(r1[RF], rf[RF], chunk=100_000, by_state=True, keep_cands=False)   # chunk: France chains
        ak = _key(ix1.get_indexer(acc.s1), pd.Index(rx.id).get_indexer(acc.sx) + off)
        X = pd.read_parquet(OUT / "ctx" / f"{split}_{f.stem}.parquet")
        b = F.block_cols(X)
        X = pd.concat([X.drop(columns=[k for k in b if k in X] + ["via"]), pd.DataFrame(
            b | F._group_ctx(X.qx.to_numpy(), np.nan_to_num(b["block_score"]), "sx"))], axis=1)
        extra = np.setdiff1d(ak, _key(X.q1, X.qx))
        if len(extra):
            e = pd.DataFrame({"q1": (extra >> 32).astype(np.int32), "qx": (extra & 0xFFFFFFFF).astype(np.int32),
                              "blocked": np.float32(0)})
            if split == "train":
                e["fold"] = fold1[e.q1.to_numpy()]
            X = pd.concat([X, e], ignore_index=True)
        if split == "train":
            k = _key(X.q1, X.qx)
            X["is_true"] = tk[np.minimum(np.searchsorted(tk, k), len(tk) - 1)] == k
        import pyarrow as pa
        i, j = X.q1.to_numpy(), X.qx.to_numpy() - off
        acc_pair = np.isin(_key(X.q1, X.qx), ak)
        ids_ = [col for col in ("q1", "qx", "is_true", "fold") if col in X]
        fc = [col for col in X.columns if col not in ids_]
        w = None
        for a in range(0, len(X), chunk):              # one chunk at a time straight to the writer (RAM)
            S = F.strings(r1, rx, i[a:a + chunk], j[a:a + chunk], counts, m1, mx)
            S["rule_acc"] = acc_pair[a:a + chunk].astype(np.float32)
            part = X.iloc[a:a + chunk].reset_index(drop=True)
            out = pd.concat([part[ids_], part[fc].astype(np.float32).round(3), S.round(3)], axis=1)
            t = pa.Table.from_pandas(out, preserve_index=False)
            w = w or pq.ParquetWriter(p.with_suffix(".tmp"), t.schema, compression="zstd")
            w.write_table(t)
            del S, part, out, t
        w.close()
        p.with_suffix(".tmp").rename(p)
        print(f"  {f.stem}: {len(raw):,} SX, {len(X):,} pairs (+{len(extra):,} rule-only) in {time.time() - t0:.0f}s, "
              f"peak {peak_gb():.2f} GB, rss {rss_gb():.2f} GB", flush=True)
        del X, rx, mx, rf, raw
        trim()


if __name__ == "__main__":
    {"rf": rf_step, "ctx": ctx_step, "feat": feat_step}[sys.argv[1]](*sys.argv[2:])
