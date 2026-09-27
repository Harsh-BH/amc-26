"""P36 step 2: address-only recovery of synthetic-brand ("gib") S2/S3 records that the decision left unassigned.

For every gib SX (gib.is_gib) that no S1 accepted, search the S1 pool of its country by address alone (P3's hashed
word TF-IDF address view, same state group), keep the top-K S1, featurize the pairs from the addresses and house
numbers, score them with a small LightGBM (grecover booster), and give the SX to its best S1 when p_g >= t_g.
The same code builds the holdout frames (split "train": the holdout world's pool = every train S1) and test's.
    probes/P36-iu-residual/py.sh probes/P36-iu-residual/recover.py cands <split> <c>   # -> data/work/p36/gc_<split>_<c>.parquet
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import A, OUT, W  # noqa: E402  (base puts src/ on the path)
import block as B  # noqa: E402
import gib as G  # noqa: E402

K = 10
FEATS = ["cos", "rank", "gap", "cos1", "rel", "n_close", "hn_eq", "hn_x", "hn_a", "hn_d", "city_eq", "tset", "part",
         "nx_tok", "na_tok"]


def gib_ids(split: str, c: str, M) -> pa.Array:
    out = []
    for i in (2, 3):
        f = pq.ParquetFile(W / f"{split}_source{i}.parquet")
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg, columns=["entity_id", "business_name", "country"])
            t = t.filter(pc.equal(t["country"], c))
            g = G.is_gib(t.column("business_name").to_pylist(), M)
            out.append(t.column("entity_id").filter(pa.array(g)).combine_chunks())
    return pa.concat_arrays(out)


def txt_of(split: str, c: str, ids: pa.Array) -> pd.DataFrame:
    parts = []
    for p in sorted(B.TXT[split].glob(f"{c}_sx_*.parquet")):
        t = pq.read_table(p)
        parts.append(t.filter(pc.is_in(t["entity_id"], value_set=ids)).to_pandas())
    return pd.concat(parts, ignore_index=True)


def search(qa: pd.DataFrame, s1: pd.DataFrame, vecs: dict) -> pd.DataFrame:
    """Top-K S1 rows per query by address cosine, within the query's state group (+ S1 without a state); a query
    without a state (or with a group no S1 has) searches the whole pool. -> qi, ci, cos."""
    cfg = dict(B.CFG, threads=8)
    Aq = B.mats(qa, {"addr"}, vecs, cfg)["addr"]
    As = B.mats(s1, {"addr"}, vecs, cfg)["addr"]
    gs = B._groups(s1.st)
    loose = gs.pop("", np.zeros(0, np.int32))
    gq = B._groups(qa.st)
    res = []
    for g, qi in gq.items():
        cols = np.concatenate([gs[g], loose]) if g in gs else np.arange(len(s1), dtype=np.int32)
        a, b, sc = B.topk(Aq[qi], [(cols, As[cols].T.tocsr())], K, cfg)
        res.append(pd.DataFrame({"qi": qi[a], "ci": b, "cos": sc}))
    return pd.concat(res, ignore_index=True)


def hn_rel(h1: np.ndarray, hx: np.ndarray) -> tuple:
    both = (h1 != "") & (hx != "")
    eq = both & (h1 == hx)
    d = np.full(len(h1), -1.0)
    ok = both & np.char.isdigit(h1.astype(str)) & np.char.isdigit(hx.astype(str))
    d[ok] = np.abs(h1[ok].astype(np.int64) - hx[ok].astype(np.int64))
    return eq.astype(np.int8), (hx != "").astype(np.int8), (h1 != "").astype(np.int8), np.log1p(np.maximum(d, 0)) * np.where(d >= 0, 1, np.nan)


def cands(split: str, c: str, skip: pa.Array | None = None) -> pd.DataFrame:
    """Candidate (sx, s1) pairs of the unassigned gib SX of one split / country, with FEATS."""
    from rapidfuzz import fuzz
    t0 = time.time()
    M = G.learn(split)
    ids = gib_ids(split, c, M)
    n_gib = len(ids)
    if skip is not None:
        ids = ids.filter(pc.invert(pc.is_in(ids, value_set=skip)))
    q = txt_of(split, c, ids)
    q = q[q.addr.fillna("").str.len() > 0].reset_index(drop=True)
    print(f"{c}: {n_gib:,} gib SX, {len(q):,} queries with an address, {time.time() - t0:.0f}s", flush=True)
    s1 = pq.read_table(B.TXT[split] / f"{c}_s1.parquet").to_pandas()
    vecs = B.vecs()
    r = search(q, s1, vecs)
    del vecs
    gc.collect()
    r = r.sort_values(["qi", "cos"], ascending=[True, False]).reset_index(drop=True)
    r["rank"] = B.ranks(r.qi.to_numpy())
    cos1 = r.groupby("qi").cos.transform("max")
    nxt = r.groupby("qi").cos.shift(-1)
    r["gap"] = (r.cos - nxt.fillna(0)).astype(np.float32)
    r["cos1"], r["rel"] = cos1.astype(np.float32), (r.cos / cos1).astype(np.float32)
    r["n_close"] = r.assign(c=(r.cos >= cos1 - 0.05)).groupby("qi").c.transform("sum").astype(np.int16)
    r["sx"] = q.entity_id.to_numpy()[r.qi]
    r["s1"] = s1.entity_id.to_numpy()[r.ci]
    ax, a1 = q.addr.to_numpy(object)[r.qi], s1.addr.to_numpy(object)[r.ci]
    r["tset"] = np.fromiter((fuzz.token_set_ratio(a, b) for a, b in zip(ax, a1)), np.float32, len(r))
    r["part"] = np.fromiter((fuzz.partial_ratio(a, b) for a, b in zip(ax, a1)), np.float32, len(r))
    r["nx_tok"] = np.fromiter((len(a.split()) for a in ax), np.int16, len(r))
    r["na_tok"] = np.fromiter((len(a.split()) for a in a1), np.int16, len(r))
    del ax, a1, s1
    rf1 = A.grab(W / f"rf_{split}_s1_{c}.parquet", r.s1.unique(), ["id", "city", "hn"])
    rfx = A.grab(W / f"rf_{split}_sx_{c}.parquet", r.sx.unique(), ["id", "city", "hn"])
    c1, cx = rf1.city.reindex(r.s1).fillna("").to_numpy(object), rfx.city.reindex(r.sx).fillna("").to_numpy(object)
    h1, hx = rf1.hn.reindex(r.s1).fillna("").to_numpy(object), rfx.hn.reindex(r.sx).fillna("").to_numpy(object)
    r["city_eq"] = ((c1 == cx) & (cx != "")).astype(np.int8)
    r["hn_eq"], r["hn_x"], r["hn_a"], r["hn_d"] = hn_rel(h1, hx)
    print(f"{c}: {len(r):,} candidate pairs over {r.sx.nunique():,} SX, {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB",
          flush=True)
    return r.drop(columns=["qi", "ci"])


if __name__ == "__main__":
    cmd, split, c = sys.argv[1:4]
    if cmd == "cands":
        skip = None
        if split == "train":        # holdout world: skip the SX the #34 decision accepted (holdout S1: a10 p'; train S1: #8c)
            from base import state
            s = state(c)
            full = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["sx", "holdout", "p", "am"])
            acc = pd.concat([s["pairs"].sx[s["pairs"].acc], full.sx[~full.holdout & full.am & (full.p >= A.T)]])
            skip = pa.array(acc.unique(), pa.large_string())
        else:
            skip = pa.array(pd.read_parquet(OUT / f"acc_test_{c}.parquet").sx.unique(), pa.large_string())
        cands(split, c, skip).to_parquet(OUT / f"gc_{split}_{c}.parquet", index=False)
