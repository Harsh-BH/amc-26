"""P49 test 2: joint exact decode per name component vs per-SX argmax, on top of sub44r (OOF), plus the oracle headroom.

Component = ambiguous SX linked through shared twin S1 (union-find). Components with <= 4 twins and <= 4 SX are decoded
exactly; larger ones keep the per-SX rule. Truth posterior over z (each SX -> a twin of its set or 'none'):
  w(z) = Π_x q_x(z_x) · Π_(j,src) Π_{m=1..M-1} r(k_j + m) / r(k_j)
q_x = dec.py's 'all' GBDT marginals (fit on SX that touch no report-half S1), q_x(none) = 1 - Σ q (Σ q capped at 1). The second
factor is the group-size correction the per-SX marginal cannot see (a second ambiguous SX landing on the same twin).
Prediction a (same space) maximises E_z Σ_{holdout j} F0.5_j, where j's other accepted pairs are assumed true
(n_true_j = base_pred_j + M_j(z)). Report: per-S1 delta vs sub44r, both halves.
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/t2.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dec as DC  # noqa: E402
from law import ratios  # noqa: E402

MAXT, MAXX = 4, 4


def components(tw):
    par = {}

    def f(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for x, s in zip(tw.sx, tw.s1):
        ra, rb = f("x" + x), f("s" + s)
        if ra != rb:
            par[ra] = rb
    return np.array([f("x" + x) for x in tw.sx])


def fb(tp, npred, ntrue):
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def decode(comp: pd.DataFrame, base_np: dict, rtab: dict, prior: bool = True) -> dict:
    """comp: rows of one component (sx, s1, g, holdout, src, s, k). -> {sx: s1 or None}."""
    xs = list(dict.fromkeys(comp.sx))
    ts = list(dict.fromkeys(comp.s1))
    ti = {t: i for i, t in enumerate(ts)}
    opts, q = [], []
    for x in xs:
        r = comp[comp.sx == x]
        qq = r.s.to_numpy()
        z = max(qq.sum(), 1.0)
        opts.append([ti[t] for t in r.s1] + [-1])
        q.append(np.r_[qq / z, max(1 - qq.sum() / z, 1e-6)])
    Z = np.array(list(itertools.product(*[range(len(o)) for o in opts])))        # index into opts
    lab = np.stack([np.array(opts[i])[Z[:, i]] for i in range(len(xs))], 1)      # twin index or -1
    w = np.prod(np.stack([q[i][Z[:, i]] for i in range(len(xs))], 1), 1)
    if prior:
        src = comp.drop_duplicates("sx").set_index("sx").src.reindex(xs).to_numpy()
        kk = comp.set_index(["s1", "sx"]).k
        for j, t in enumerate(ts):
            for s in ("S2", "S3"):
                m = (lab == j) & (src == s)[None, :]
                M = m.sum(1)
                if M.max() < 2:
                    continue
                cand = [x for x, ss in zip(xs, src) if ss == s and (t, x) in kk.index]
                k0 = int(kk.loc[(t, cand[0])]) if cand else 0
                r = rtab[s]
                adj = np.ones(len(M))
                for mm in range(2, M.max() + 1):
                    fac = np.prod([r[min(k0 + i, len(r) - 1)] / r[min(k0, len(r) - 1)] for i in range(1, mm)])
                    adj[M == mm] = fac
                w = w * adj
    w = w / w.sum()
    hol = comp.drop_duplicates("s1").set_index("s1").holdout.reindex(ts).to_numpy()
    ef = np.zeros(len(lab))
    for j, t in enumerate(ts):
        if not hol[j]:
            continue
        b = base_np[t]
        Mz = (lab == j).sum(1)                                   # truth count per z
        Pa = (lab == j).sum(1)                                   # predicted count per a (same enumeration)
        C = ((lab[:, None, :] == j) & (lab[None, :, :] == j)).sum(2)   # a x z correct
        F = fb(b + C, (b + Pa)[:, None], (b + Mz)[None, :])
        ef += F @ w
    a = lab[int(np.argmax(ef))]
    return {x: (ts[a[i]] if a[i] >= 0 else None) for i, x in enumerate(xs)}


def main():
    import lightgbm as lgb
    D = {c: DC.load(c) for c in DC.LB_W}
    rt = {c: {s: ratios(DC.L[f"{c}|{s}"])[1] for s in ("S2", "S3")} for c in DC.LB_W}
    res = {}
    for c, d in D.items():
        tw = DC.feats(c, d["tw"])
        hh = np.where(tw.g >= 0, d["half"][tw.g.clip(0)], -1)
        for h in (0, 1):
            tw[f"h{h}"] = pd.Series(hh == h).groupby(tw.sx.to_numpy()).transform("any").to_numpy()
        tw = tw[tw.h0 | tw.h1].reset_index(drop=True)
        tw["comp"] = components(tw)
        d["tw"], d["f0"] = tw, DC.fb(d["tp"], d["npred"], d["n_true"])
    for rep in (1, 0):
        Xs, ys = [], []
        for c, d in D.items():
            full = DC.feats(c, DC.load(c)["tw"])                    # every ambiguous SX (incl. no holdout twin)
            hh = np.where(full.g >= 0, d["half"][full.g.clip(0)], -1)
            m = ~pd.Series(hh == rep).groupby(full.sx.to_numpy()).transform("any").to_numpy()
            Xs.append(full.loc[m, DC.ARMS["all"]]), ys.append(full.y_full.to_numpy()[m])
            del full
        b = lgb.train(DC.PRM, lgb.Dataset(pd.concat(Xs), np.concatenate(ys)), 150)
        for c, d in D.items():
            tw = d["tw"]
            s = b.predict(tw[DC.ARMS["all"]])
            tw = tw.assign(s=s)
            # oracle: owner if it is a holdout twin in the set, else nowhere
            orc = np.where(tw.y_full & (tw.g >= 0), 1.0, 0.0)
            f_or = DC.apply(d, tw, orc, 0.5)
            # per-SX argmax at the tau dec.py picked for this direction
            tau = {1: {"India": 0.775, "US": 0.725}, 0: {"India": 0.65, "US": 0.775}}[rep][c]
            f_ind = DC.apply(d, tw, s, tau)
            # joint decode: base predicted count per holdout twin = sub44r accepted minus its ambiguous accepted pairs
            npb = d["npred"].copy()
            old = tw[tw.acc & (tw.g >= 0)]
            np.subtract.at(npb, old.g.to_numpy(), 1)
            base_np = dict(zip(tw.s1, np.where(tw.g >= 0, npb[tw.g.clip(0)], 0)))
            size = tw.groupby("comp").agg(nt=("s1", "nunique"), nx=("sx", "nunique"))
            small = set(size[(size.nt <= D_MAXT) & (size.nx <= D_MAXX)].index)
            out = {}
            for prior in (True, False):
                choice = {}
                for k, comp in tw[tw.comp.isin(small)].groupby("comp"):
                    choice |= decode(comp, base_np, rt[c], prior)
                sj = np.array([1.0 if (x in choice and choice[x] == t) else 0.0 for x, t in zip(tw.sx, tw.s1)])
                # large components keep the per-SX rule
                big = ~tw.comp.isin(small).to_numpy()
                sj = np.where(big, np.where(s >= tau, s, 0.0), sj)
                best_is = tw.assign(sc=sj).groupby("sx").sc.transform("max")
                sj = np.where(big, sj, np.where(best_is > 0, sj, 0.0))
                out[prior] = DC.apply(d, tw, np.where(big, sj, sj), 0.5 if True else tau)
            hm = d["half"] == rep
            for nm, f in (("oracle", f_or), ("per-SX argmax", f_ind), ("joint EF + size prior", out[True]),
                          ("joint EF, no size factor", out[False])):
                dl = (f - d["f0"])[hm]
                res[(c, rep, nm)] = (dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl)))
            print(f"{c} report h{rep}: small components {len(small):,} of {len(size):,} "
                  f"({tw.comp.isin(small).mean():.3f} of rows)", flush=True)
    for rep in (1, 0):
        for nm in ("oracle", "per-SX argmax", "joint EF + size prior", "joint EF, no size factor"):
            lb = sum(DC.LB_W[c] * res[(c, rep, nm)][0] for c in DC.LB_W)
            se = np.sqrt(sum((DC.LB_W[c] * res[(c, rep, nm)][1]) ** 2 for c in DC.LB_W))
            print(f"h{rep} {nm:26s} " + " | ".join(f"{c} {res[(c, rep, nm)][0]:+.6f} ± {res[(c, rep, nm)][1]:.6f}" for c in DC.LB_W)
                  + f" | LB-eq {lb:+.6f} ± {se:.6f}", flush=True)


D_MAXT, D_MAXX = MAXT, MAXX
if __name__ == "__main__":
    # self-check: two SX, two twins, strong q -> each to its own twin; the EF decode refuses a coin flip
    cp = pd.DataFrame({"sx": ["a", "a", "b", "b"], "s1": ["A", "B", "A", "B"], "g": [0, 1, 0, 1], "holdout": True,
                       "src": "S2", "s": [0.95, 0.03, 0.03, 0.95], "k": 1})
    rt0 = {s: ratios(DC.L[f"India|{s}"])[1] for s in ("S2", "S3")}
    assert decode(cp, {"A": 3, "B": 3}, rt0) == {"a": "A", "b": "B"}
    cp.s = [0.5, 0.5, 0.5, 0.5]
    assert decode(cp, {"A": 3, "B": 3}, rt0) == {"a": None, "b": None}
    main()
