"""P44: group / competitor context columns for A (label-free, from #8c's p over every row of the holdout world).

  sx_n_ge_t / sx_n_ge_05  candidate S1 of the SX with #8c p >= 0.71 / >= 0.5
  s1_ncand                candidate rows of the S1;  s1_rank = rank of p among the S1's argmax pairs / s1_n_am
  s1_best_nonam           best #8c p of the S1's rows whose SX went elsewhere
  n_same_hn_acc           other accepted (p >= 0.71) argmax SX of the S1 with this SX's house number
  n_acc_hn_eq_s1          other accepted argmax SX of the S1 carrying the S1's house number
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/ctx.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402

CTX2 = ["sx_n_ge_t", "sx_n_ge_05", "s1_ncand", "s1_rank", "s1_best_nonam", "n_same_hn_acc", "n_acc_hn_eq_s1"]


def ctx2(A: pd.DataFrame, d: pd.DataFrame, h1: np.ndarray, hx: np.ndarray) -> pd.DataFrame:
    """CTX2 onto A (argmax pairs of scored S1: s1, sx, g, p, s1_n_am), from d = every row of the world (s1, sx, p, am);
    h1 / hx = house numbers of A's rows."""
    kx = pd.factorize(d.sx)[0]
    p = d.p.to_numpy()
    sx_t, sx_5 = np.bincount(kx, p >= E.T)[kx], np.bincount(kx, p >= 0.5)[kx]
    k1 = pd.factorize(d.s1)[0]
    ncand = np.bincount(k1)[k1]
    nonam = np.where(d.am.to_numpy(), -1.0, p)
    best = pd.Series(nonam).groupby(k1).transform("max").to_numpy()
    f = pd.DataFrame({"s1": d.s1, "sx": d.sx, "sx_n_ge_t": sx_t, "sx_n_ge_05": sx_5, "s1_ncand": ncand,
                      "s1_best_nonam": np.where(best < 0, np.nan, best)})
    A = A.drop(columns=[x for x in CTX2 if x in A]).merge(f, on=["s1", "sx"], how="left")
    assert A.sx_n_ge_t.notna().all()
    g = A.g.to_numpy()
    A["s1_rank"] = A.groupby("g").p.rank(ascending=False, method="first").to_numpy() / A.s1_n_am.to_numpy()
    # house-number agreement inside the S1's accepted group (by #8c p, so p' of either half never leaks)
    acc = (A.p.to_numpy() >= E.T)
    has = hx != ""
    key = pd.factorize(pd.Series(hx))[0]
    grp = pd.DataFrame({"g": g, "k": key, "a": (acc & has).astype(int)})
    same = grp.groupby(["g", "k"]).a.transform("sum").to_numpy() - grp.a.to_numpy()
    A["n_same_hn_acc"] = np.where(has, same, np.nan).astype(np.float32)
    eq = (acc & has & (hx == h1)).astype(int)
    A["n_acc_hn_eq_s1"] = (pd.Series(eq).groupby(g).transform("sum").to_numpy() - eq).astype(np.float32)
    return A


def add(c: str):
    from solution_p44 import HnTab
    A, S = E.load(c)
    d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "p", "am"])
    hn = HnTab(E.W / "p44ms/hn", "train")
    hx = hn.t[c]["sx"].take(pa.array(A.qx.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    h1 = hn.t[c]["s1"].take(pa.array(A.q1.to_numpy())).to_numpy(zero_copy_only=False).astype(object)
    A = ctx2(A, d, h1, hx)
    A.to_parquet(E.OUT / f"A_{c}.parquet")
    print(c, A[CTX2].describe().loc[["mean", "max"]].round(3).to_dict(), flush=True)


if __name__ == "__main__" and len(sys.argv) == 1:
    for c in E.C:
        add(c)


CTX3 = ["s1_gap_next", "s1_gap_prev", "sx_psum", "s1_psum", "is_s3", "n_same_src_acc"]


def add3(A: pd.DataFrame, d: pd.DataFrame, s3=None) -> pd.DataFrame:
    """More group context, from #8c p. d = every row of the world (s1, sx, p). S1 rank gaps: p minus the next lower /
    next higher p among the S1's argmax pairs (NaN at the ends); p mass of the SX's candidates and of the S1's argmax
    pairs; S3 flag; other accepted argmax SX of the S1 from the same source."""
    kx = pd.factorize(d.sx)[0]
    f = pd.DataFrame({"s1": d.s1, "sx": d.sx, "sx_psum": np.bincount(kx, d.p.to_numpy())[kx]})
    A = A.drop(columns=[x for x in CTX3 if x in A]).merge(f, on=["s1", "sx"], how="left")
    g, p = A.g.to_numpy(), A.p.to_numpy()
    o = np.lexsort((-p, g))
    go, po = g[o], p[o]
    nxt = np.r_[po[1:], np.nan]
    nxt[np.r_[go[1:] != go[:-1], True]] = np.nan
    prv = np.r_[np.nan, po[:-1]]
    prv[np.r_[True, go[1:] != go[:-1]]] = np.nan
    A.loc[A.index[o], "s1_gap_next"] = po - nxt
    A.loc[A.index[o], "s1_gap_prev"] = prv - po
    A["s1_psum"] = np.bincount(g, p)[g]
    s3 = A.sx.str.startswith("S3-").to_numpy() if s3 is None else np.asarray(s3, bool)   # s3: when sx holds codes
    A["is_s3"] = s3.astype(np.float32)
    acc = (p >= E.T).astype(int)
    for v in (0, 1):
        m = s3 == bool(v)
        cnt = np.bincount(g[m], acc[m], minlength=g.max() + 1)
        A.loc[m, "n_same_src_acc"] = cnt[g[m]] - acc[m]
    return A


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "3":
    for c in E.C:
        A, S = E.load(c)
        d = pd.read_parquet(E.W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "p"])
        A = add3(A, d)
        A.to_parquet(E.OUT / f"A_{c}.parquet")
        print(c, A[CTX3].describe().loc[["mean", "max"]].round(3).to_dict(), flush=True)
