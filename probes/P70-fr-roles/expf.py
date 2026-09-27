"""Rival decision rule on OUR India/US scores: per S1, rank its argmax pairs by p54 (caps 5 S2 / 6 S3, top 12), choose the
prefix k in 0..n maximising exact expected F0.5 with independent Bernoulli(p) (Poisson-binomial over prefix TP and suffix
extra trues); k=0 scores prod(1-p). Evaluated on P76's frame (sub54 p54, out of fold, P57 full holdout world) vs the shipped
p >= .71. Optional power calibration p**a (fit on one half, report the other).
    expf.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P76-segment-hill")
import dive as D
LB = {"India": 809986 / 1732544, "US": 663106 / 1732544}


def pb(p):                                   # Poisson-binomial pmf rows for a batch: p (m, k) -> (m, k+1)
    m, k = p.shape
    d = np.zeros((m, k + 1)); d[:, 0] = 1
    for j in range(k):
        d[:, 1:j + 2] = d[:, 1:j + 2] * (1 - p[:, j:j + 1]) + d[:, 0:j + 1] * p[:, j:j + 1]
        d[:, 0] *= 1 - p[:, j]
    return d


def best_k(P):                               # P (m, n) sorted desc -> argmax_k E[F0.5]
    m, n = P.shape
    E = np.zeros((m, n + 1))
    E[:, 0] = np.prod(1 - P, axis=1)
    for k in range(1, n + 1):
        a, b = pb(P[:, :k]), pb(P[:, k:]) if k < n else np.ones((m, 1))
        t, r = np.arange(k + 1)[:, None], np.arange(b.shape[1])[None, :]
        Wt = 1.25 * t / (0.25 * (t + r) + k)
        E[:, k] = np.einsum("mt,tr,mr->m", a, Wt, b)
    return E.argmax(1)


def decide(F, S, a=1.0):
    g, p, s3 = F.g.to_numpy(), np.clip(F.p54.to_numpy(), 1e-6, 1 - 1e-6) ** a, F.is_s3.to_numpy().astype(bool)
    o = np.lexsort((-p, g))
    g, p, s3, idx = g[o], p[o], s3[o], o
    rk2 = pd.Series(~s3).groupby(g).cumsum().to_numpy() * ~s3
    rk3 = pd.Series(s3).groupby(g).cumsum().to_numpy() * s3
    keep = np.where(s3, rk3 <= 6, rk2 <= 5)
    g, p, idx = g[keep], p[keep], idx[keep]
    pos = pd.Series(np.ones(len(g))).groupby(g).cumsum().to_numpy().astype(int) - 1
    g, p, idx, pos = g[pos < 12], p[pos < 12], idx[pos < 12], pos[pos < 12]
    n_s = np.bincount(g, minlength=len(S))
    acc = np.zeros(len(F), bool)
    for n in range(1, 13):
        grp = np.flatnonzero(n_s == n)
        if not len(grp): continue
        sel = np.isin(g, grp)
        gg, pp, ii, qq = g[sel], p[sel], idx[sel], pos[sel]
        rows = np.searchsorted(grp, gg)
        M = np.zeros((len(grp), n)); M[rows, qq] = pp
        k = best_k(M)
        acc[ii[qq < k[rows]]] = True
    return acc


def f05(F, S, acc):
    g, y, n = F.g.to_numpy(), F.y.to_numpy(), len(S)
    return D.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), S.n_true.to_numpy())


for c in ("India", "US"):
    F, S = D.load(c)
    base = f05(F, S, F.p54.to_numpy() >= D.T)
    h = F.groupby("g").half.first().reindex(range(len(S))).fillna(0).to_numpy()
    for a in (1.0,):
        new = f05(F, S, decide(F, S, a))
        dl = (new - base) * LB[c]
        print(f"{c} a={a}: F0.5 base {base.mean():.5f} -> expF {new.mean():.5f} | LB-eq {dl.mean() * len(S) / len(S):+.6f} per S1-mean "
              f"(x country share: {LB[c] * (new.mean() - base.mean()):+.6f}) | halves: " +
              ", ".join(f"h{int(v)} {LB[c] * (new[h == v] - base[h == v]).mean():+.6f} ± {LB[c] * (new[h == v] - base[h == v]).std() / np.sqrt((h == v).sum()):.6f}" for v in (0, 1)), flush=True)
    # power calibration: pick a on half 0, report half 1 (and reverse)
    res = {}
    for a in (0.8, 0.9, 1.1, 1.25, 1.5):
        res[a] = f05(F, S, decide(F, S, a))
    for fit, rep in ((0, 1), (1, 0)):
        best = max(res, key=lambda a: (res[a][h == fit] - base[h == fit]).mean())
        d_ = (res[best][h == rep] - base[h == rep])
        print(f"   {c} power a fit on h{fit} -> {best}; report h{rep}: {LB[c] * d_.mean():+.6f} ± {LB[c] * d_.std() / np.sqrt(len(d_)):.6f}", flush=True)
