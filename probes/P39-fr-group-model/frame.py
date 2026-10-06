"""P39 frames: one row per (S1, its SX-argmax candidate) with label-free group-context columns, for
  LOCO   P6's source-only M2 (data/work/p27/p6/S_holdout_<src>) on the other country's fold 0, band-blended with the
         source-only bge (data/work/p30/r, P13's out-of-domain France weights) + P30 veto tau 0.1 = the France chain
         analogue (P30 eval.py 'tau:0.1'), t = P30's fixed label-free t (US>India 0.915, India>US 0.92)
  France the France chain file (data/work/p31/fr31v0.1_t9868 = #34v_r90's France before the rescue), t 0.9868
The argmax is fixed (LOCO: P6's am_all over every train S1; France: argmax of the chain p), so every SX sits in at
most one S1 and only accept / reject is learned (re-assignment cannot be scored in LOCO: competing S1 are off-frame).

    .venv/bin/python probes/P39-fr-group-model/frame.py loco US India | loco India US | france | selftest
    -> data/work/p39/{loco_US_India, loco_India_US, france}.parquet
"""
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio, token_set_ratio

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
from patterns import name_rel  # noqa: E402

W = ROOT / "data/work"
OUT = W / "p39"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
W_FR, LO, HI, TAU = [0.9245, 0.2595, 0.0515], 0.02, 0.98, 0.1       # P13 France weights, P30 band / veto
T_LOCO = {("US", "India"): 0.915, ("India", "US"): 0.92}            # P30 loco_US_India.json t_base (fixed)
T_FR = 0.9868
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])
KINDS = ["eq", "add1", "drop1", "typo", "sub1", "addN", "dropN", "subN", "disjoint", "web"]
HNREL = ["miss", "eq", "sib3", "d12", "neg12", "other"]
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def blend_veto(g, r):
    """P30 eval.apply_rule(g, r, 'tau:0.1'): band 0.02<g<0.98 blended with the CE, GBDT-confident pairs vetoed at r<0.1."""
    rb = np.where(np.isnan(r), 0.5, r)
    b = (g > LO) & (g < HI)
    p = g.astype(np.float64).copy()
    p[b] = 1 / (1 + np.exp(-(W_FR[0] * logit(g[b]) + W_FR[1] * logit(rb[b]) + W_FR[2])))
    return np.where((g >= HI) & (r < TAU), 0.0, p)


def argmax_sx(qx, p):
    o = np.lexsort((-p, qx))
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, qx[o][1:] != qx[o][:-1]]] if len(o) else o] = True
    return m


def _rf(path, ids):
    t = pq.read_table(path, columns=["id", "core", "is_web", "state", "hn", "places"])
    t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
    return t.to_pandas().set_index("id").reindex(ids)


def hn_rel(h1, hx):
    ok = (h1 != "") & (hx != "")
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[ok]), errors="coerce").to_numpy(float)
    out = np.where(~ok | np.isnan(d), 0, np.where(d == 0, 1, np.where(np.isin(d, SIB3), 2,
                   np.where(np.isin(d, [1, 2]), 3, np.where(np.isin(d, [-1, -2]), 4, 5)))))
    return out.astype(np.int8)


def rank_desc(key, p):
    """1-based rank of p within key (descending)."""
    o = np.lexsort((-p, key))
    k = key[o]
    i = np.arange(len(o))
    start = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    r = np.empty(len(p), np.int32)
    r[o] = i - start + 1
    return r


def group_feats(d: pd.DataFrame, t: float) -> pd.DataFrame:
    """d: argmax rows with s1 (int code), is2, p. Adds the label-free context columns (all per S1 / source)."""
    s1, is2, p = d.s1.to_numpy(), d.is2.to_numpy(), d.p.to_numpy(np.float64)
    ns = int(s1.max()) + 1
    acc = p >= t
    live = p >= LO
    lt = logit(p) - logit(t)
    d["lt"] = lt.astype(np.float32)
    ok = live
    q = np.full(len(p), 0.0)
    q[ok] = pd.Series(p[ok]).rank(pct=True).to_numpy()                 # quantile among the country's live argmax p
    d["q"] = q.astype(np.float32)
    d["acc"] = acc
    key = s1 * 2 + is2
    d["rank_src"] = rank_desc(key, p)
    n2 = np.bincount(s1[acc & is2], minlength=ns)
    n3 = np.bincount(s1[acc & ~is2], minlength=ns)
    na2 = np.bincount(s1[live & is2], minlength=ns)
    na3 = np.bincount(s1[live & ~is2], minlength=ns)
    d["n_acc_src"] = (np.where(is2, n2[s1], n3[s1]) - acc).astype(np.int16)       # excluding self
    d["n_acc_oth"] = np.where(is2, n3[s1], n2[s1]).astype(np.int16)
    d["n_live_src"] = np.where(is2, na2[s1], na3[s1]).astype(np.int16)
    d["n_live_oth"] = np.where(is2, na3[s1], na2[s1]).astype(np.int16)
    b2 = np.full(ns, -12.0)
    b3 = np.full(ns, -12.0)
    np.maximum.at(b2, s1[is2], lt[is2])
    np.maximum.at(b3, s1[~is2], lt[~is2])
    d["best_oth_lt"] = np.where(is2, b3[s1], b2[s1]).astype(np.float32)
    # best in the same source excluding self: the second best when self is the best
    r = d.rank_src.to_numpy()
    sec = np.full(2 * ns, -12.0)
    m2 = r == 2
    sec[key[m2]] = lt[m2]
    best = np.where(is2, b2[s1], b3[s1])
    d["best_src_ex_lt"] = np.where(r == 1, sec[key], best).astype(np.float32)
    return d


def text_feats(d: pd.DataFrame, e1: np.ndarray, ex: np.ndarray, xs: str, c: str) -> pd.DataFrame:
    """Record-level columns from the rule-feature caches (rf_*): hn relation, name relation / similarity, place overlap,
    and group coherence (similarity of this SX to the S1's accepted members)."""
    u1, ux = pd.unique(e1), pd.unique(ex)
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", u1)
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", ux)
    c1 = r1.core.reindex(e1).fillna("").to_numpy()
    cx = rx.core.reindex(ex).fillna("").to_numpy()
    h1 = r1.hn.reindex(e1).fillna("").to_numpy()
    hx = rx.hn.reindex(ex).fillna("").to_numpy()
    wx = rx.is_web.reindex(ex).fillna(False).to_numpy(bool)
    p1 = r1.places.reindex(e1).fillna("").to_numpy()
    px = rx.places.reindex(ex).fillna("").to_numpy()
    st1 = r1.state.reindex(e1).fillna("").to_numpy()
    stx = rx.state.reindex(ex).fillna("").to_numpy()
    del r1, rx
    d["hn_rel"] = hn_rel(h1, hx)
    kind = np.array([name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)], object)
    d["kind"] = pd.Categorical(kind, categories=KINDS).codes.astype(np.int8)
    d["nm_ratio"] = np.array([ratio(a, b) for a, b in zip(c1, cx)], np.float32)
    d["nm_tset"] = np.array([token_set_ratio(a, b) for a, b in zip(c1, cx)], np.float32)
    d["city"] = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)])
    d["state_eq"] = (st1 != "") & (st1 == stx)
    d["is_web"] = wx
    d["addr_eq"] = (d.hn_rel.to_numpy() == 1) & d.city.to_numpy() & np.isin(kind, ["eq", "add1", "drop1"])
    # group coherence: best similarity of this SX to another accepted member of the same S1 (either source)
    acc = d.acc.to_numpy()
    s1 = d.s1.to_numpy()
    o = np.flatnonzero(acc)
    mem = pd.DataFrame({"s1": s1[o], "m": o})
    j = pd.DataFrame({"s1": s1, "i": np.arange(len(d))}).merge(mem, on="s1")
    j = j[j.i.to_numpy() != j.m.to_numpy()]
    ii, mm = j.i.to_numpy(), j.m.to_numpy()
    sim = np.array([ratio(a, b) for a, b in zip(cx[ii], cx[mm])], np.float32)
    hq = (hx[ii] != "") & (hx[ii] == hx[mm])
    g = pd.DataFrame({"i": ii, "sim": sim, "hq": hq}).groupby("i").agg(sim=("sim", "max"), hq=("hq", "max"))
    d["grp_sim"] = g.sim.reindex(np.arange(len(d))).fillna(-1).to_numpy(np.float32)
    d["grp_hn"] = g.hq.reindex(np.arange(len(d))).fillna(False).to_numpy(bool)
    return d


def loco(src: str, tgt: str):
    t0 = time.time()
    import features as F
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country.to_numpy() == ci].drop(columns="country").reset_index(drop=True)
    rf = pd.read_parquet(W / f"p30/r/r_loco{src}_tgt_{tgt}.parquet")
    S = S.merge(rf, on=["s1", "sx"], how="left")
    del rf
    g, r = S.p.to_numpy(np.float64), S.r.to_numpy(np.float64)
    assert not np.isnan(r[g > LO]).any()
    S["pc"] = blend_veto(g, r)
    S = S[S.am_all.to_numpy()].reset_index(drop=True)                  # the decision set: SX-argmax pairs
    id1, idx = F.ids("train", tgt)
    e1 = id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False)
    ex = idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False)
    del id1, idx
    fo = pd.read_parquet(W / "s1_folds.parquet")
    fo = fo[(fo.fold == 0) & (fo.country == tgt)].entity_id.to_numpy()
    code = pd.Index(fo).get_indexer(e1)
    assert (code >= 0).all()
    d = pd.DataFrame({"s1": code.astype(np.int32), "is2": pd.Series(ex).str.startswith("S2-").to_numpy(),
                      "p": S.pc.to_numpy(np.float64), "g": S.p.to_numpy(np.float32),
                      "y": S.is_true.to_numpy(bool)})
    del S
    d = group_feats(d, T_LOCO[(src, tgt)])
    d = text_feats(d, e1, ex, "train", tgt)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"loco_{src}_{tgt}.parquet", index=False)
    gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    pd.DataFrame({"id": fo, "ntrue": gt.reindex(fo).fillna(0).to_numpy(np.int16)}).to_parquet(
        OUT / f"s1_{src}_{tgt}.parquet", index=False)
    print(f"loco {src}>{tgt}: {len(d):,} argmax rows, {len(fo):,} S1, y {d.y.mean():.3f}, acc {d.acc.mean():.3f}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def france():
    t0 = time.time()
    f = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet")
    qx = pd.factorize(f.sx)[0]
    am = argmax_sx(qx, f.p.to_numpy(np.float64))
    f = f[am].reset_index(drop=True)
    s1ids = pq.read_table(W / "rf_test_s1_France.parquet", columns=["id"]).column(0).to_numpy()
    code = pd.Index(s1ids).get_indexer(f.s1)
    assert (code >= 0).all()
    d = pd.DataFrame({"s1": code.astype(np.int32), "is2": f.sx.str.startswith("S2-").to_numpy(),
                      "p": f.p.to_numpy(np.float64)})
    d = group_feats(d, T_FR)
    d = text_feats(d, f.s1.to_numpy(), f.sx.to_numpy(), "test", "France")
    d["sx"] = f.sx.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "france.parquet", index=False)
    pd.DataFrame({"id": s1ids}).to_parquet(OUT / "s1_france.parquet", index=False)
    print(f"france: {len(d):,} argmax rows, {len(s1ids):,} S1, acc {d.acc.mean():.3f}; {time.time() - t0:.0f}s "
          f"peak {peak():.2f} GB", flush=True)


def selftest():
    d = pd.DataFrame({"s1": [0, 0, 0, 1, 1], "is2": [True, True, False, True, False], "p": [0.99, 0.5, 0.97, 0.3, 0.995]})
    d = group_feats(d, 0.95)
    assert d.rank_src.tolist() == [1, 2, 1, 1, 1], d.rank_src.tolist()
    assert d.n_acc_src.tolist() == [0, 1, 0, 0, 0] and d.n_acc_oth.tolist() == [1, 1, 1, 1, 0], d
    assert np.isclose(d.best_src_ex_lt[0], logit(0.5) - logit(0.95)) and d.best_src_ex_lt[2] == -12
    assert np.isclose(d.best_oth_lt[3], logit(0.995) - logit(0.95))
    assert hn_rel(np.array(["12", "12", "", "7"]), np.array(["12", "17", "3", "8"])).tolist() == [1, 2, 0, 3]
    assert np.allclose(blend_veto(np.array([0.99, 0.99, 0.5, 0.01]), np.array([0.05, 0.5, np.nan, np.nan])),
                       [0.0, 0.99, 1 / (1 + np.exp(-W_FR[2])), 0.01])
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"loco": lambda: loco(a[1], a[2]), "france": france, "selftest": selftest}[a[0]]()
