"""P34: is test's higher add rate (US 1.64x holdout) a stratum-mix shift or a model shift?
(1) adds / drops per accepted pair by stratum, holdout h1 (h0 fit) vs test, and the holdout precision of the adds;
    expected precision of test's adds = holdout per-stratum precision weighted by test's adds.
(2) R51 (P23-research): the per-S1 holdout gain (h0 -> h1) reweighted to test's per-S1 stratum mix (importance-weighted
    validation). Stratum = max runner-up p2 bin over the S1's argmax rows with 0.3 < p < 0.999 (P23's definition).
    probes/P34-a10-stack/py.sh probes/P34-a10-stack/shift.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import a10 as A
import ho

OUT = A.W / "p34"

P2L = ["none", "0-.05", ".05-.2", ".2-.5", ".5+"]
PB = [0, 0.3, 0.5, 0.6, 0.71, 0.8, 0.9, 1.01]


def s1_stratum(g, p, p2, n):
    """Per-S1 stratum code (index into P2L) from its argmax rows."""
    m = (p > 0.3) & (p < 0.999)
    top = np.full(n, -1.0)
    np.maximum.at(top, g[m], p2[m])
    return np.where(top < 0, 0, np.digitize(top, [0.05, 0.2, 0.5]) + 1)


def main():
    pd.set_option("display.width", 220)
    for c in ("India", "US"):
        s = ho.load(c)
        a, h1 = s["a"], s["h"][s["a"].g] == 1
        ph = np.load(OUT / f"ho_p_{c}_h0.npy")
        te = pd.read_parquet(OUT / f"te_feat_{c}.parquet", columns=["p", "p2", "p_new", "row"])
        # (1) pair-level: adds / drops by (p band, p2 bin)
        rows = []
        for world, p0, p1, p2, y in (("holdout_h1", a.p.to_numpy()[h1], ph[h1], a.p2.to_numpy()[h1], a.y.to_numpy()[h1]),
                                     ("test", te.p.to_numpy(), te.p_new.to_numpy(), te.p2.to_numpy(), None)):
            b, n = p0 >= A.T, p1 >= A.T
            st = pd.DataFrame({"pb": pd.cut(p0, PB, right=False), "p2b": pd.cut(p2, [-0.01, 0.05, 0.2, 0.5, 1.01],
                               labels=["0-.05", ".05-.2", ".2-.5", ".5+"]), "add": n & ~b, "drop": b & ~n,
                               "y": y if y is not None else np.nan})
            gg = st.groupby(["p2b"], observed=False)
            rows.append(pd.DataFrame({"world": world, "pairs": gg.size(), "add_rate": gg["add"].mean(),
                                      "drop_rate": gg["drop"].mean(), "adds": gg["add"].sum(), "drops": gg["drop"].sum()}))
            if y is not None:
                ho_st = st
        print(f"\n=== {c}: add / drop rate per argmax pair, by runner-up p2 bin")
        print(pd.concat(rows).reset_index().to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        # holdout precision of adds (true share) and of drops, by p2 bin and p band; test's adds weighted by it
        ad = ho_st[ho_st["add"]]
        dr = ho_st[ho_st["drop"]]
        prec = ad.groupby(["p2b", "pb"], observed=True).y.agg(["size", "mean"])
        print(f"holdout adds: {len(ad)}, precision {ad.y.mean():.3f}; drops: {len(dr)}, true share {dr.y.mean():.3f}")
        tst = pd.DataFrame({"pb": pd.cut(te.p, PB, right=False),
                            "p2b": pd.cut(te.p2, [-0.01, 0.05, 0.2, 0.5, 1.01], labels=["0-.05", ".05-.2", ".2-.5", ".5+"])})
        tadd = tst[(te.p_new >= A.T).to_numpy() & (te.p < A.T).to_numpy()]
        w = tadd.groupby(["p2b", "pb"], observed=True).size()
        pr = prec["mean"].reindex(w.index)
        cov = w[pr.notna()].sum() / w.sum()
        print(f"test adds: {len(tadd)}; expected precision from holdout strata {np.average(pr[pr.notna()], weights=w[pr.notna()]):.3f} "
              f"({cov:.1%} of test adds in strata with a holdout add)")
        # (2) R51 reweighting of the per-S1 gain
        d = (ho.f_of(s, ph >= A.T) - s["f0"])
        k_ho = s1_stratum(a.g.to_numpy(), a.p.to_numpy(), a.p2.to_numpy(), s["n"])
        s1 = pd.read_parquet(OUT / f"prob_test_{c}_a10.parquet", columns=["s1"]).s1.to_numpy(object)[te.row]
        gt, u = pd.factorize(s1)
        n_s1 = int((pd.read_parquet(A.W / "test_source1.parquet", columns=["country"]).country == c).sum())
        k_te = np.r_[s1_stratum(gt, te.p.to_numpy(), te.p2.to_numpy(), len(u)), np.zeros(n_s1 - len(u), np.int64)]
        sh_te = np.bincount(k_te, minlength=len(P2L)) / n_s1
        rep = s["h"] == 1
        sh_ho = np.bincount(k_ho[rep], minlength=len(P2L)) / rep.sum()
        mu = np.array([d[rep & (k_ho == k)].mean() if (rep & (k_ho == k)).any() else 0 for k in range(len(P2L))])
        var = np.array([d[rep & (k_ho == k)].var(ddof=1) / max((rep & (k_ho == k)).sum(), 1) if (rep & (k_ho == k)).sum() > 1 else 0
                        for k in range(len(P2L))])
        print(pd.DataFrame({"stratum": P2L, "share_holdout": sh_ho, "share_test": sh_te, "gain_h1": mu}).to_string(
            index=False, float_format=lambda v: f"{v:+.6f}"))
        rw, se = (sh_te * mu).sum(), np.sqrt((sh_te ** 2 * var).sum())
        print(f"{c}: holdout h1 gain {d[rep].mean():+.6f}; test-reweighted {rw:+.6f} ± {se:.6f} "
              f"(LB-eq x{ho.LB_W[c]}: {ho.LB_W[c] * rw:+.6f})")


if __name__ == "__main__":
    main()
