"""P53: S2<->S3 "triangle" features for every candidate row (a, y) of a world, from SX-SX similarity inside S1 blocks.

For each S1 block (its candidate rows with #8c p), and every other SX x of the block:
  sn = token_set_ratio(name x, name y) / 100, sa = the same on the address, s = (sn + sa) / 2   (P3 normalized text)
Row features for (a, y):
  t_max / t_max_n / t_max_a  max_x min(p(x,a), s | sn | sa)                  (NaN: y is alone in the block)
  t_ncs                      #x with p(x,a) >= 0.71 and s(x,y) >= 0.8
  t_best                     s(y, x*), x* = the block's highest-p other SX
  t_hq                       hn agreement with x*: 1 equal, 0 differ, NaN when either has no hn
  c_max                      max over y's other candidate S1 b of t_max(b, y)   (NaN: y has one candidate S1)
  c_pmax                     max over those b of min(p(y,b), t_max(b,y))
  c_ncs                      sum over those b of t_ncs(b, y)
    py.sh tri.py <split> <c>      -> data/work/p53/tri_<split>_<c>.parquet (s1, sx, features), in the pairs file's order
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
W = ROOT / "data/work"
OUT = W / "p53"
T = 0.71
FEATS = ["t_max", "t_max_n", "t_max_a", "t_ncs", "t_best", "t_hq", "c_max", "c_pmax", "c_ncs"]


def pairs_file(split, c):
    return W / (f"p23/pairs_{c}.parquet" if split == "train" else f"p34/te_pairs_{c}.parquet")


def blocks(k1: np.ndarray):
    """Ordered (i, j), i != j, row pairs inside each k1 block."""
    o = np.argsort(k1, kind="stable")
    ks = k1[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    n = np.diff(np.r_[st, len(ks)])
    st, n = st[n > 1], n[n > 1]
    n2 = n * n
    b = np.repeat(np.arange(len(st)), n2)
    off = np.arange(n2.sum()) - np.repeat(np.cumsum(n2) - n2, n2)
    i, j = st[b] + off // n[b], st[b] + off % n[b]
    m = i != j
    return o[i[m]], o[j[m]]


def top2(key: np.ndarray, v: np.ndarray):
    """Per row: the best v among the OTHER rows of its key (-inf when none), and that row's index (-1)."""
    o = np.lexsort((-v, key))
    ks, vs = key[o], v[o]
    first = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(first) - 1
    fi = np.flatnonzero(first)
    t1v, t1r = vs[fi], o[fi]
    sec = np.r_[fi[1:], len(ks)] - fi > 1
    t2v = np.full(len(fi), -np.inf)
    t2r = np.full(len(fi), -1)
    t2v[sec], t2r[sec] = vs[fi[sec] + 1], o[fi[sec] + 1]
    g = np.empty(len(v), np.int64)
    g[o] = gid
    me_top = t1r[g] == np.arange(len(v))
    return np.where(me_top, t2v[g], t1v[g]), np.where(me_top, t2r[g], t1r[g])


def texts(split, c, qx: np.ndarray):
    import features as F
    files, _ = F.sx_files(split, c)
    tb = pa.concat_tables([pq.read_table(f, columns=["name", "addr"]) for f in files]).combine_chunks()
    t = tb.take(pa.array(qx))
    del tb
    hn = pq.read_table(W / f"p44ms/hn/{split}_{c}_sx.parquet").column("hn").combine_chunks().take(pa.array(qx))
    return (t.column("name").to_pylist(), t.column("addr").to_pylist(),
            np.asarray(hn.to_pylist(), dtype=object))


def feats(split, c, k1: np.ndarray, qx: np.ndarray, p: np.ndarray) -> dict:
    """Triangle features for rows (S1 key k1, SX code qx, p) of one world."""
    from rapidfuzz import fuzz, process
    n = len(p)
    ri, rj = blocks(k1)
    # unique unordered SX pairs -> sims
    a, b = np.minimum(qx[ri], qx[rj]), np.maximum(qx[ri], qx[rj])
    key = (a << 32) | b
    uk, inv = np.unique(key, return_inverse=True)
    ua, ub = (uk >> 32).astype(np.int64), (uk & 0xFFFFFFFF).astype(np.int64)
    u = np.unique(np.r_[ua, ub])
    nm, ad, hn = texts(split, c, u)
    pa_, pb_ = np.searchsorted(u, ua), np.searchsorted(u, ub)
    print(c, split, f"rows {n:,}, block pairs {len(ri):,}, unique SX pairs {len(uk):,}, SX {len(u):,}", flush=True)
    sn = process.cpdist([nm[i] for i in pa_], [nm[i] for i in pb_], scorer=fuzz.token_set_ratio, workers=4) / 100.0
    sa = process.cpdist([ad[i] for i in pa_], [ad[i] for i in pb_], scorer=fuzz.token_set_ratio, workers=4) / 100.0
    ha, hb = hn[pa_], hn[pb_]
    hq = np.where((ha == "") | (hb == ""), np.nan, (ha == hb).astype(float))
    del nm, ad, hn
    sn, sa, hq = sn[inv].astype(np.float64), sa[inv].astype(np.float64), hq[inv]
    s = (sn + sa) / 2
    px = p[rj]
    f = {}
    for nme, v in (("t_max", s), ("t_max_n", sn), ("t_max_a", sa)):
        m = np.full(n, -1.0)
        np.maximum.at(m, ri, np.minimum(px, v))
        f[nme] = np.where(m < 0, np.nan, m)
    f["t_ncs"] = np.bincount(ri, ((px >= T) & (s >= 0.8)).astype(float), minlength=n)
    _, best = top2(k1, p)
    hit = rj == best[ri]
    f["t_best"] = np.full(n, np.nan)
    f["t_best"][ri[hit]] = s[hit]
    f["t_hq"] = np.full(n, np.nan)
    f["t_hq"][ri[hit]] = hq[hit]
    # competitors: y's other candidate S1
    tm = np.nan_to_num(f["t_max"], nan=0.0)
    cm, cr = top2(qx, tm)
    f["c_max"] = np.where(cr < 0, np.nan, cm)
    cp, cr2 = top2(qx, np.minimum(p, tm))
    f["c_pmax"] = np.where(cr2 < 0, np.nan, cp)
    tot = np.bincount(qx, f["t_ncs"], minlength=qx.max() + 1)[qx]
    f["c_ncs"] = tot - f["t_ncs"]
    return f


def build(split, c):
    import features as F
    t0 = time.time()
    d = pq.read_table(pairs_file(split, c), columns=["s1", "sx", "p"])
    _, idx = F.ids(split, c)
    qx = pc.index_in(d.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
    del idx
    assert not np.isnan(qx.astype(float)).any()
    qx = qx.astype(np.int64)
    k1 = pd.factorize(d.column("s1").to_numpy(zero_copy_only=False))[0]
    f = feats(split, c, k1, qx, d.column("p").to_numpy().astype(np.float64))
    out = pd.DataFrame({"s1": d.column("s1").to_numpy(zero_copy_only=False),
                        "sx": d.column("sx").to_numpy(zero_copy_only=False),
                        **{k: f[k].astype(np.float32) for k in FEATS}})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"tri_{split}_{c}.parquet", index=False)
    print(c, split, f"{time.time() - t0:.0f}s", out[FEATS].describe().loc[["count", "mean"]].round(3).to_dict(), flush=True)


def selftest():
    k = np.array([0, 0, 1, 0, 1, 2])
    v = np.array([.5, .9, .3, .7, .8, .1])
    best, row = top2(k, v)
    assert np.allclose(best[[0, 1, 3]], [.9, .7, .9]) and list(row[[0, 1, 3]]) == [1, 3, 1]
    assert list(row[[2, 4]]) == [4, 2] and row[5] == -1 and best[5] == -np.inf
    ri, rj = blocks(k)
    assert sorted(zip(ri, rj)) == sorted([(0, 1), (1, 0), (0, 3), (3, 0), (1, 3), (3, 1), (2, 4), (4, 2)])
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        build(sys.argv[1], sys.argv[2])
