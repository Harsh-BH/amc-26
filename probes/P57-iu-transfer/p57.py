"""P57: P44's holdout frame rebuilt on a sub-world whose competitor rows come only from chosen train folds.

The #8c holdout world = every candidate row of the fold-0 (scored) S1 + every other train S1's row on the same SX.
m2_big was fit on folds 2-5, so those competitors carry in-sample p. A world keeps the scored S1 plus the competitor
S1 of `folds`; argmax, a10's columns, P44's CTX2 / CTX3 and #34's p' are all re-derived inside it (a10.features and
ctx.ctx2 / add3, the same builders the test path uses). Per-pair columns (hn, pruner q, m2hn / hn21 / r_all / zoo p)
are per-row lookups, so they are taken from the full-world tables.
    .venv/bin/python probes/P57-iu-transfer/p57.py build <c> [rows|cluster|swap]   # -> data/work/p57/A_{world}_{c}.parquet
"""
import os
import sys
import time

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import ev as E  # noqa: E402
import a10  # noqa: E402
import ctx as CX  # noqa: E402
import zoo50 as Z  # noqa: E402

OUT = E.W / "p57"
WORLDS = {"full": tuple(range(10)), "oos6": (0, 1, 6, 7, 8, 9), "oos4": (0, 1, 6, 7, 8), "ins4": (0, 2, 3, 4, 5),
          "oos2": (0, 1)}
# cluster worlds: a dropped S1 takes the SX it owns (train truth) with it, so no SX is left without its owner -- a smaller
# universe of businesses, as on test (the S1-only worlds above leave those SX free-looking: an artifact test never has)
# swap worlds: every S1 and row kept; competitor rows of `folds` get P45's fsub (fit on folds 6-9) in place of m2_big (2-5):
# p_m2 := fsub, logit p := logit p + logit fsub - logit p_m2. sw = folds 2-5 swapped (every competitor p out of sample);
# swi = folds 6-9 swapped (the mirror: now 2-9 are all in-sample) -- the control.
SWORLDS = {"sw": (2, 3, 4, 5), "swi": (6, 7, 8, 9)}
CWORLDS = {"c6": (0, 1, 6, 7, 8, 9), "c4o": (0, 1, 6, 7, 8), "c4i": (0, 2, 3, 4, 5), "c2": (0, 1), "c5": (0, 1, 7, 8, 9)}
HN = ["hn_d", "hn_absd", "hn_cls", "hn_sib3", "hn_lev", "hn_lendiff", "hn_len1"]
ZC = [f"p_{m}" for m in Z.MEM + Z.RANK]


def rows(c: str) -> pd.DataFrame:
    """Every row of the full holdout world with its fold and per-row lookups (same order as p23 pairs)."""
    import features as F
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "r", "p", "am"])
    fo = pd.read_parquet(E.W / "s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
    d["fold"] = np.where(d.holdout, 0, fo.reindex(d.s1).to_numpy()).astype(np.int8)
    assert (fo.reindex(d.s1[d.holdout]).to_numpy() == 0).all()
    id1, idx = F.ids("train", c)
    d["q1"] = pc.index_in(pa.array(d.s1.to_numpy(object), pa.string()), value_set=id1).to_numpy(zero_copy_only=False)
    d["qx"] = pc.index_in(pa.array(d.sx.to_numpy(object), pa.string()), value_set=idx).to_numpy(zero_copy_only=False)
    del id1, idx
    d["q1"], d["qx"] = d.q1.astype(np.int64), d.qx.astype(np.int64)
    z = pd.read_parquet(E.W / f"p45/oof/holdout_{c}.parquet", columns=["s1", "sx", "p_r_all"] + ZC)
    assert (z.s1.to_numpy() == d.s1.to_numpy()).all() and (z.sx.to_numpy() == d.sx.to_numpy()).all()
    for k in ["p_r_all"] + ZC:
        d[k] = z[k].to_numpy(np.float32)
    del z
    for m in ("m2hn", "hn21"):
        t = pq.read_table(E.OUT / f"pred_{m}/holdout_{c}.parquet")
        assert (t.column("q1").to_numpy() == d.q1.to_numpy()).all() and (t.column("qx").to_numpy() == d.qx.to_numpy()).all()
        d[f"p_{m}"] = t.column("p").to_numpy().astype(np.float32)
    ux = pa.array(d.sx.unique(), pa.large_string())
    gt = pq.read_table(E.W / "gt_pairs.parquet", columns=["s1", "sx"])
    gt = gt.filter(pc.is_in(gt.column("sx"), value_set=ux)).to_pandas().drop_duplicates("sx")
    of = pd.Series(fo.reindex(gt.s1).to_numpy(), index=gt.sx.to_numpy())
    d["ofold"] = of.reindex(d.sx).fillna(-1).to_numpy().astype(np.int8)          # fold of the SX's true S1, -1 = none
    del gt, of, ux
    hk = pq.read_table(E.W / f"p10/hold_{c}.parquet", columns=["q1", "qx"])
    q = pd.Series(pq.read_table(E.W / f"p10/hold_q_{c}.parquet").column("q").to_numpy(),
                  index=E.keys(hk.column("q1").to_numpy(), hk.column("qx").to_numpy()))
    d["q"] = q[~q.index.duplicated()].reindex(E.keys(d.q1, d.qx)).to_numpy(np.float32)
    return d


def pa34(A: pd.DataFrame) -> np.ndarray:
    """#34's p' out of fold: S1 of half h take P34's booster fit on half 1-h (guarded)."""
    import lightgbm as lgb
    out = np.empty(len(A))
    for fh in (0, 1):
        m = A.half.to_numpy() == 1 - fh
        out[m] = a10.stacked(lgb.Booster(model_file=str(E.W / f"p34/a10_h{fh}.txt")), A[m])
    return out


def world(d: pd.DataFrame, s1t, sxt, k1, kx, gi, S: pd.DataFrame, hn, c: str, folds, cluster=False, swap=None) -> pd.DataFrame:
    from solution_p44 import HnTab  # noqa: F401  (hn is a HnTab)
    import model as M
    m = np.isin(d.fold.to_numpy(), folds)
    if cluster:
        m &= (d.ofold.to_numpy() < 0) | np.isin(d.ofold.to_numpy(), folds)
    ix = np.flatnonzero(m)
    p, pm2 = d.p.to_numpy(np.float64)[ix], d.p_m2.to_numpy(np.float64)[ix]
    if swap:
        sw = np.isin(d.fold.to_numpy()[ix], swap) & ~d.holdout.to_numpy()[ix]
        fs = d.p_fsub.to_numpy(np.float64)[ix][sw]
        p[sw] = 1 / (1 + np.exp(-(E.lg(p[sw]) + E.lg(fs) - E.lg(pm2[sw]))))
        pm2[sw] = fs
    am = M.argmax_sx(kx[ix], p)
    a = a10.features(k1[ix], kx[ix], p, pm2, d.r.to_numpy()[ix], am, d.holdout.to_numpy()[ix], s1t, sxt)
    j = ix[a.row.to_numpy()]                              # row of the full world
    a = a.drop(columns=["q1", "qx"])
    a["j"] = j
    a["am_full"] = d.am.to_numpy()[j]
    for k in ("y", "q1", "qx", "q", "p_r_all", "p_m2hn", "p_hn21", *ZC):
        a[k] = d[k].to_numpy()[j]
    a["s1"], a["sx"] = k1[j], kx[j]                        # integer codes stand in for ids (ctx only groups / merges)
    a["g"] = gi[j]
    a["half"] = S.half.to_numpy()[a.g.to_numpy()]
    for k, v in hn.feats(c, a.q1.to_numpy(), a.qx.to_numpy()).items():
        a[k] = v
    assert (a.hn_cls.to_numpy() == a.hc.to_numpy()).all(), "derived hn class != a10's hc"
    dw = pd.DataFrame({"s1": k1[ix], "sx": kx[ix], "p": p, "am": am})
    hx = hn.t[c]["sx"].take(pa.array(a.qx.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    h1 = hn.t[c]["s1"].take(pa.array(a.q1.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    a = CX.ctx2(a, dw, h1, hx)
    a = CX.add3(a, dw.drop(columns="am"), s3=d.sx.str.startswith("S3-").to_numpy()[a.j.to_numpy()])
    a["pa"] = pa34(a)
    a["n_comp_rows"] = np.bincount(kx[ix], minlength=kx.max() + 1)[a.sx.to_numpy()] - 1   # other S1 rows on the SX
    return a


def build(c: str, which: str = "rows"):
    from solution_p44 import HnTab
    t0 = time.time()
    d = rows(c)
    k1, u1 = pd.factorize(d.s1)
    kx, ux = pd.factorize(d.sx)
    ids = pd.read_parquet(E.W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    gi = pd.Series(np.arange(len(ids)), index=ids).reindex(d.s1).to_numpy()   # NaN on competitor rows
    gi = np.where(np.isnan(gi), -1, gi).astype(np.int64)
    s1t, sxt = a10.texts("train", c, np.asarray(u1, object), np.asarray(ux, object))
    d = d.drop(columns=["s1"])
    A0, S = E.load(c)
    hn = HnTab(E.W / "p44ms/hn", "train")
    print(c, f"rows {len(d):,}, texts in {time.time() - t0:.0f}s", flush=True)
    worlds = {"rows": WORLDS, "cluster": CWORLDS, "swap": SWORLDS}[which]
    for w, folds in worlds.items():
        if which == "swap":
            a = world(d, s1t, sxt, k1, kx, gi, S, hn, c, tuple(range(10)), swap=folds)
        else:
            a = world(d, s1t, sxt, k1, kx, gi, S, hn, c, folds, cluster=which == "cluster")
        if w == "full":                                   # parity: the full world must reproduce P44's A frame
            assert len(a) == len(A0) and (a.am_full.all())
            x = a.sort_values("j").reset_index(drop=True)
            key0 = pd.DataFrame({"s1": A0.s1, "sx": A0.sx}).merge(
                pd.DataFrame({"s1": np.asarray(u1, object)[k1[x.j]], "sx": np.asarray(ux, object)[kx[x.j]], "i": np.arange(len(x))}),
                on=["s1", "sx"], how="left").i.to_numpy()
            x = x.iloc[key0].reset_index(drop=True)
            bad = {}
            for k in a10.COLS + CX.CTX2 + CX.CTX3 + ["pa", "q", "p_r_all", "p_m2hn", "y", "g", "half"] + HN:
                u, v = x[k].to_numpy(np.float64), A0[k].to_numpy(np.float64)
                n = int((~np.isclose(u, v, rtol=1e-5, atol=1e-6, equal_nan=True)).sum())
                if n:
                    bad[k] = n
            print(c, "full-world parity mismatches:", bad or "none", flush=True)
            assert not bad, bad
        a.to_parquet(OUT / f"A_{w}_{c}.parquet", index=False)
        new = ~a.am_full.to_numpy()
        print(c, w, f"folds {folds}: {len(a):,} argmax pairs ({new.sum():,} new vs full, p>=.71 among new "
                    f"{(a.p.to_numpy()[new] >= E.T).mean():.3f}); mean n_comp_rows {a.n_comp_rows.mean():.3f}, "
                    f"sx_psum-p>=.02 {(a.sx_psum - a.p >= .02).mean():.4f}, >=.5 {(a.sx_psum - a.p >= .5).mean():.4f} "
                    f"| {time.time() - t0:.0f}s", flush=True)
        del a


if __name__ == "__main__":
    {"build": build}[sys.argv[1]](*sys.argv[2:])
