"""P57 density-matched holdout (master's request): the arms' per-S1 deltas re-weighted so the holdout S1 mix matches test's
on label-free group shape -- (argmax pairs of the S1, #34-accepted pairs of the S1), capped -- the density that differs
(test 5.8 SX per S1 vs 4.7). Padding the world with extra SX is impossible (non-candidate pairs have no m2 / CE / zoo p).
Weights w(b) = P_test(b) / P_holdout(b) per country; weighted mean and SE of the per-S1 delta on the report half.
    .venv/bin/python probes/P57-iu-transfer/dw.py [world ...]      -> data/work/p57/dw.json"""
import json
import os
import sys

os.environ.setdefault("P57_ARMS", "rall_only,sub44,sub44_psumoff,sub44r,sub44r_psumoff,sub54,rel44r,rel54")
sys.path.insert(0, "probes/P57-iu-transfer")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev57 as V  # noqa: E402

T = 0.71
W = V.E.W
CAP_AM, CAP_ACC = 9, 8


def bucket(n_am, n_acc):
    return np.minimum(n_am, CAP_AM) * 100 + np.minimum(n_acc, CAP_ACC)


def test_dist(c: str) -> pd.Series:
    A = pd.read_parquet(W / f"p50/T_{c}.parquet", columns=["g", "pa"])
    g = A.g.to_numpy()
    n_am, n_acc = np.bincount(g), np.bincount(g, A.pa.to_numpy() >= T).astype(np.int64)
    n_s1 = int((pd.read_parquet(W / "test_source1.parquet", columns=["country"]).country == c).sum())
    b = pd.Series(bucket(n_am, n_acc)).value_counts()
    b.loc[0] = b.get(0, 0) + n_s1 - len(n_am)                     # test S1 with no argmax pair
    return b / b.sum()


def wmean(d, w):
    m = (w * d).sum() / w.sum()
    return float(m), float(np.sqrt((w ** 2 * (d - m) ** 2).sum()) / w.sum())


def main(worlds):
    S = V.load_s()
    pt = {c: test_dist(c) for c in V.E.C}
    res = {}
    for w in worlds:
        D = V.load(w)
        for c in V.E.C:
            A, nt, h = D[c], S[c].n_true.to_numpy(), S[c].half.to_numpy()
            n = len(nt)
            g = A.g.to_numpy()
            n_am = np.bincount(g, minlength=n)
            n_acc = np.bincount(g, A.pa.to_numpy() >= T, minlength=n).astype(np.int64)
            b = bucket(n_am, n_acc)
            ph = pd.Series(b).value_counts(normalize=True)
            wt = (pt[c].reindex(ph.index).fillna(0) / ph).reindex(b).to_numpy()
            cover = float(pt[c].reindex(ph.index).fillna(0).sum())
            f34 = V.per_s1(A, nt, A.pa.to_numpy() >= T)
            f8 = V.per_s1(A, nt, A.p.to_numpy() >= T)
            res[f"{w}|{c}|shape"] = {"test_mass_covered": cover, "hold_mean_am": float(n_am.mean()),
                                     "test_mean_am": float((pt[c].index // 100 * pt[c]).sum()), "w_max": float(wt.max())}
            for rep in (0, 1):
                m = h == rep
                res[f"{w}|{c}|34v8|rep{rep}"] = {"u": float((f34 - f8)[m].mean()), "w": wmean((f34 - f8)[m], wt[m])}
            for name, (cols, _) in V.ARMS.items():
                s = np.empty(len(A))
                for fit_h in (0, 1):
                    mm = A.half.to_numpy() == 1 - fit_h
                    s[mm] = V.score(V.boosters(name, fit_h), A[mm], cols)
                f = V.per_s1(A, nt, s >= T)
                for rep in (0, 1):
                    m = h == rep
                    d = (f - f34)[m]
                    res[f"{w}|{c}|{name}|rep{rep}"] = {"u": float(d.mean()), "u_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                                       "w": wmean(d, wt[m])}
        del D
        json.dump(res, open(W / "p57/dw.json", "w"), indent=1, default=float)
    for w in worlds:
        for c in V.E.C:
            print(w, c, res[f"{w}|{c}|shape"])
        print(f"{w}: #34 vs #8c  unweighted / density-weighted (LB-eq, mean of both report halves)")
        for name in ["34v8"] + list(V.ARMS):
            u = sum(V.E.LB_W[c] * np.mean([res[f"{w}|{c}|{name}|rep{r}"]["u"] for r in (0, 1)]) for c in V.E.C)
            ww = sum(V.E.LB_W[c] * np.mean([res[f"{w}|{c}|{name}|rep{r}"]["w"][0] for r in (0, 1)]) for c in V.E.C)
            se = np.sqrt(sum((V.E.LB_W[c] * np.hypot(*[res[f"{w}|{c}|{name}|rep{r}"]["w"][1] for r in (0, 1)]) / 2) ** 2 for c in V.E.C))
            per = " ".join(f"{c} u {np.mean([res[f'{w}|{c}|{name}|rep{r}']['u'] for r in (0, 1)]):+.6f} w "
                           f"{np.mean([res[f'{w}|{c}|{name}|rep{r}']['w'][0] for r in (0, 1)]):+.6f}" for c in V.E.C)
            print(f"  {name:16s} u {u:+.6f}  w {ww:+.6f} ± {se:.6f}   {per}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["full", "sw"])
