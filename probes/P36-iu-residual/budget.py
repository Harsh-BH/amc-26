"""P36 step 1: the India/US error budget on the #34 holdout decision (out of fold), split by a per-S1 Shapley value
over 7 error types so the rows add up (P23-research budget2.py's method).
    probes/P36-iu-residual/py.sh probes/P36-iu-residual/budget.py     # -> stdout table + data/work/p36/err_{c}.parquet
"""
import itertools
import math

import numpy as np
import pandas as pd

from base import A, LB_W, NA, OUT, W, state

FN = ["block_addr", "block_noaddr", "pruner", "lowp", "stolen"]
FP = ["fp_ns", "fp_s"]
TYPES = FN + FP


def errors(s: dict) -> pd.DataFrame:
    """One row per error pair: s1, sx, g, type (+ p8c, pa, p_m2, r for in-candidate rows)."""
    pr, n_true = s["pairs"], s["n_true"]
    miss = pd.read_parquet(W / f"p23/err_{s['c']}.parquet", columns=["s1", "sx", "kind", "in_p3", "sx_addr"])
    miss = miss[miss.kind == "missed"].copy()
    noaddr = miss.sx_addr.fillna("").str.strip().str.upper().isin(NA)
    miss["type"] = np.where(miss.in_p3.astype(bool), "pruner", np.where(noaddr, "block_noaddr", "block_addr"))
    gi = pd.Series(np.arange(len(s["ids"])), index=s["ids"])
    miss["g"] = gi.reindex(miss.s1).to_numpy()
    # truth accounting: every true pair is either a candidate row or a miss
    n_c = np.bincount(pr.g[pr.y], minlength=len(gi)) + np.bincount(miss.g, minlength=len(gi))
    assert (n_c == n_true).all(), int((n_c != n_true).sum())
    fnr = pr[pr.y & ~pr.acc].copy()
    fnr["type"] = np.where(fnr.am, "lowp", "stolen")
    fpr = pr[pr.acc & ~pr.y].copy()
    fpr["type"] = np.where(n_true[fpr.g] > 0, "fp_ns", "fp_s")
    cols = ["s1", "sx", "g", "type"]
    return pd.concat([miss[cols], fnr[cols + ["p", "pa", "p_m2", "r", "am"]], fpr[cols + ["p", "pa", "p_m2", "r", "am"]]],
                     ignore_index=True)


def shapley(s: dict, e: pd.DataFrame) -> dict:
    """Per-type Shapley share of the F0.5 lost (summed over S1, / N) + the per-S1 total loss check."""
    pr, n = s["pairs"], len(s["ids"])
    tp = np.bincount(pr.g[pr.acc & pr.y], minlength=n)
    X = {t: np.bincount(e.g[e.type == t], minlength=n) for t in TYPES}
    bad = np.flatnonzero(sum(X.values()) > 0)
    tp, nt, X = tp[bad], s["n_true"][bad], {t: v[bad] for t, v in X.items()}
    v = lambda S: A.fb(tp + sum(X[q] for q in S if q in FN), tp + sum(X[q] for q in S if q in FN)
                       + sum(X[q] for q in FP if q not in S), nt)
    phi, k = {}, len(TYPES)
    for t in TYPES:
        oth, acc = [q for q in TYPES if q != t], 0.0
        for r in range(k):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(oth, r):
                acc += w * (v(S + (t,)) - v(S)).sum()
        phi[t] = acc / n
    lost = (1 - s["f34"]).sum() / n
    assert abs(sum(phi.values()) - lost) < 1e-9, (sum(phi.values()), lost)
    return phi


def oracle(s: dict, e: pd.DataFrame, t: str) -> float:
    """F0.5 gain of fixing type t alone (everything else as is), / N."""
    pr, n = s["pairs"], len(s["ids"])
    tp = np.bincount(pr.g[pr.acc & pr.y], minlength=n)
    npred = np.bincount(pr.g[pr.acc], minlength=n)
    x = np.bincount(e.g[e.type == t], minlength=n)
    new = A.fb(tp + x, npred + x, s["n_true"]) if t in FN else A.fb(tp, npred - x, s["n_true"])
    return (new - s["f34"]).sum() / n


if __name__ == "__main__":
    rows = []
    for c in LB_W:
        s = state(c)
        e = errors(s)
        e.to_parquet(OUT / f"err_{c}.parquet", index=False)
        phi = shapley(s, e)
        empty = (s["n_true"] > 0) & (np.bincount(s["pairs"].g[s["pairs"].acc], minlength=len(s["ids"])) == 0)
        print(f"{c}: F0.5 #34 {s['f34'].mean():.6f} (lost {1 - s['f34'].mean():.5f}); pairs by type "
              f"{e.type.value_counts().to_dict()}; wrongly-empty S1 {int(empty.sum())} = {empty.mean():.5f} lost")
        et = e[e.g.isin(np.flatnonzero(empty))].type.value_counts()
        print(f"   wrongly-empty S1 error pairs by type: {et.to_dict()}")
        for t in TYPES:
            rows.append(dict(c=c, type=t, pairs=int((e.type == t).sum()), shapley=phi[t], oracle=oracle(s, e, t),
                             lb=phi[t] * LB_W[c]))
        rows.append(dict(c=c, type="wrongly_empty", pairs=int(empty.sum()), shapley=empty.mean(), oracle=np.nan,
                         lb=empty.mean() * LB_W[c]))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.pivot(index="type", columns="c", values=["pairs", "shapley", "oracle", "lb"]).round(6).to_string())
    r.to_csv(OUT / "budget.csv", index=False)
