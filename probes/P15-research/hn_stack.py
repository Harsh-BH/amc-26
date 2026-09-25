"""P15: does a signed house-number delta add anything on top of m2_big p?  Stacker check, holdout only.

Input: data/work/p15/_hn_<c>.parquet from hn_check.py (fold-0 S1 rows + every competing pair, m2_big p, h1/hx).
LightGBM on [logit p] vs [logit p + hn delta class + signed delta], trained on S1-hash half 0 of fold 0, reported
on half 1. Decision = argmax per SX over all rows + threshold tuned on half 0. Per-S1 F0.5 over half-1 S1 with
n_true = true rows in the file (blocking misses are identical for both arms, so the delta is fair; the level is not).
    .venv/bin/python probes/P15-research/hn_stack.py India
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hn_check import hn_cls  # noqa: E402


def f05_per_s1(q1, tp, npred, ntrue):
    p = np.where(npred > 0, tp / np.maximum(npred, 1), 1.0)
    r = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 1.0)
    f = np.where((npred == 0) & (ntrue == 0), 1.0,
                 np.where(tp == 0, 0.0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-9)))
    return f


def argmax(h, s):
    return s == pd.Series(s).groupby(h.qx.to_numpy()).transform("max").to_numpy()


def evaluate(h, s, half, t, am=None):
    """h: all rows; s: stacked score; returns per-S1 F0.5 for fold-0 S1 of `half`."""
    am = argmax(h, s) if am is None else am
    acc = am & (s >= t)
    sel = (h.fold.to_numpy() == 0) & (h.half.to_numpy() == half)
    g = pd.DataFrame({"q1": h.q1.to_numpy()[sel], "tp": (acc & (h.is_true.to_numpy() > 0))[sel],
                      "np": acc[sel], "nt": (h.is_true.to_numpy() > 0)[sel]}).groupby("q1").sum()
    return pd.Series(f05_per_s1(g.index, g.tp.to_numpy(), g.np.to_numpy(), g.nt.to_numpy()), index=g.index)


CE_W = [0.617, 0.666, 0.261]          # P11 s2P blend (#6's CE), band 0.02 < p < 0.98, India + US


def ce_blend(h, c):
    """Replace p on holdout band pairs by #6's blended p (P11 2-epoch CE r cache, P4b blend rule)."""
    sys.path.insert(0, str(ROOT.parents[1] / "code/business_entity_resolution/src"))
    import features as F
    import pyarrow as pa
    import pyarrow.compute as pc
    r = pd.read_parquet(ROOT.parents[1] / "data/work/p11/r_s2P_holdout.parquet")
    s1, sx = F.ids("train", c)
    q1 = pc.index_in(pa.array(r.s1.to_numpy()), value_set=s1).to_numpy(zero_copy_only=False)
    qx = pc.index_in(pa.array(r.sx.to_numpy()), value_set=sx).to_numpy(zero_copy_only=False)
    ok = ~(pd.isna(q1) | pd.isna(qx))
    r = pd.DataFrame({"q1": q1[ok].astype(np.int64), "qx": qx[ok].astype(np.int64), "r": r.r.to_numpy()[ok]})
    rr = h[["q1", "qx"]].astype(np.int64).merge(r, on=["q1", "qx"], how="left").r.to_numpy(np.float32)
    p = h.p.to_numpy(np.float32)
    b = (p > 0.02) & (p < 0.98) & ~np.isnan(rr)
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    out = p.copy()
    out[b] = 1 / (1 + np.exp(-(CE_W[0] * lg(p[b]) + CE_W[1] * lg(rr[b]) + CE_W[2])))
    print(f"{c}: CE blend applied to {int(b.sum()):,} rows ({len(r):,} r in cache)", flush=True)
    return out


SUPPORT = "sup" in sys.argv[2:]


def support(h, h1, hx):
    """Collective features from the S1's other candidates: how many other SX of this S1 carry the same house number
    as this SX (n_same), the best p among them (p_same), and how many carry the S1's own number (n_s1hn)."""
    num = lambda v: pd.to_numeric(pd.Series(v).replace("", "-1")).to_numpy(np.int64)
    k = pd.DataFrame({"q1": h.q1.to_numpy(np.int64), "hx": num(hx), "h1": num(h1), "p": h.p.to_numpy(np.float32)})
    has = (k.hx >= 0).to_numpy()
    g = k.groupby(["q1", "hx"]).p
    n = g.transform("size").to_numpy() - 1
    mx = g.transform("max").to_numpy()
    k["r"] = g.rank(method="first", ascending=False).to_numpy()
    sec = k.loc[k.r == 2, ["q1", "hx", "p"]].rename(columns={"p": "s2"})
    s2 = k[["q1", "hx"]].merge(sec, on=["q1", "hx"], how="left").s2.to_numpy()   # left merge keeps row order
    p_same = np.where(k.p.to_numpy() >= mx, s2, mx)            # best p among the OTHER members
    eqh = ((k.hx == k.h1) & has).astype(np.int32)
    n_s1hn = eqh.groupby(k.q1).transform("sum").to_numpy() - eqh.to_numpy()
    return pd.DataFrame({"n_same": np.where(has, n, np.nan).astype(np.float32),
                         "p_same": np.where(has & (n > 0), p_same, np.nan).astype(np.float32),
                         "n_s1hn": n_s1hn.astype(np.float32)})


def main(c, ce=False):
    t0 = time.time()
    h = pd.read_parquet(ROOT.parents[1] / f"data/work/p15/_hn_{c}.parquet")
    if ce:
        h["p"] = ce_blend(h, c)
    h["half"] = (pd.util.hash_array(h.q1.to_numpy().astype(np.int64)) % 2).astype(np.int8)
    # rows of non-holdout S1 compete for SX but are not scored: give them half = -1 for training too
    h.loc[h.fold != 0, "half"] = -1
    h1, hx = h.h1.fillna("").to_numpy(), h.hx.fillna("").to_numpy()
    cls = hn_cls(h1, hx)
    both = (h1 != "") & (hx != "")
    eq = (both & (h1 == hx)).astype(np.float32)
    d = np.zeros(len(h), np.float32)
    d[both] = np.clip(pd.to_numeric(pd.Series(hx[both])).to_numpy() - pd.to_numeric(pd.Series(h1[both])).to_numpy(),
                      -100, 100)
    lp = np.log(np.clip(h.p.to_numpy(), 1e-6, 1 - 1e-6) / (1 - np.clip(h.p.to_numpy(), 1e-6, 1 - 1e-6)))
    X0 = pd.DataFrame({"lp": lp})
    sup = support(h, h1, hx) if SUPPORT else None
    del h1, hx
    h.drop(columns=["h1", "hx"], inplace=True)
    X1 = X0.assign(cls=cls.astype(np.float32), d=np.where(both, d, np.nan).astype(np.float32),
                   eq=eq)
    if sup is not None:
        X1 = pd.concat([X1, sup], axis=1)
    y = (h.is_true.to_numpy() > 0).astype(np.int8)
    tr = h.half.to_numpy() == 0          # out-of-sample for m2_big (folds 2-5) and the CE (folds 4-9)
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1,
               num_threads=4)
    out = {}
    # residual boosting from the raw logit: "base" = raw p itself (0 trees, t retuned), "hn" = raw logit + trees on
    # [lp, hn class, signed delta, eq]. A plain GBDT stacker on lp alone loses threshold resolution (flat plateaus
    # where the band is sparse), which cost 0.0005-0.0008 in the CE world.
    for tag, X in (("base", None), ("hn", X1)):
        if X is None:
            z = lp.astype(np.float64)
        else:
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], init_score=lp[tr], categorical_feature=["cls"]), 200)
            z = lp + m.predict(X, raw_score=True)
        s = 1 / (1 + np.exp(-z)) + 1e-9 * h.p.to_numpy(np.float64)
        am = argmax(h, s)
        best = max(((evaluate(h, s, 0, t, am).mean(), t) for t in np.arange(0.05, 0.99, 0.01)))
        out[tag] = (evaluate(h, s, 1, best[1], am), best[1])
        del s, am, z
        print(f"{c} {tag}: tune-half F0.5 {best[0]:.5f} at t {best[1]:.2f}; report half {out[tag][0].mean():.5f}",
              flush=True)
    raw = evaluate(h, h.p.to_numpy(), 1, 0.69)
    a, b = out["base"][0], out["hn"][0].reindex(out["base"][0].index)
    dlt = b - a
    print(f"{c}: raw m2_big p at t .69 report half {raw.mean():.5f}; hn - base = {dlt.mean():+.5f} "
          f"± {dlt.std() / np.sqrt(len(dlt)):.5f} (n S1 {len(dlt):,}, up {(dlt > 0).sum()}, down {(dlt < 0).sum()}); "
          f"{time.time() - t0:.0f}s")
    return raw.mean(), a.mean(), b.mean(), dlt.mean(), dlt.std() / np.sqrt(len(dlt)), len(dlt)


if __name__ == "__main__":
    assert np.allclose(f05_per_s1(None, np.array([2, 0, 0]), np.array([3, 0, 1]), np.array([2, 0, 0])),
                       [1.25 * (2 / 3) / (0.25 * 2 / 3 + 1), 1.0, 0.0])
    c = sys.argv[1] if len(sys.argv) > 1 else "India"
    ce = "ce" in sys.argv[2:]
    t0 = time.time()
    r = main(c, ce)
    sys.path.insert(0, str(ROOT.parents[1]))
    from ledger import log
    log("exp", probe="P15-research", subproblem="features", name=f"hn-signed-delta{'+support' if SUPPORT else ''}-stack-{c}{'-onCE' if ce else ''}",
        world="holdout-dense-half1" + ("(#6 p: m2_big + 2ep CE blend)" if ce else "(m2_big p, pre-CE)"),
        metrics={"f05_raw_t069": round(r[0], 5), "f05_stack_base": round(r[1], 5), "f05_stack_hn": round(r[2], 5),
                 "d_f05": round(r[3], 5), "d_se": round(r[4], 5), "n_s1": r[5]},
        config={"stacker": "lgb 200x31 residual (init_score = logit p) on [lp, hn class, signed delta, eq]; base = raw p, t retuned", "train": "fold0 half0 only (out-of-sample)",
                "decision": "argmax per SX + t tuned on half 0"},
        hypothesis="orphan siblings take hn + {1,2,3,4,5,7,9,11,13,21}; a signed delta separates them from noise",
        verdict="keep" if r[3] > 3 * r[4] else "inconclusive", runtime_s=round(time.time() - t0),
        peak_gb=round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1e6, 2))
