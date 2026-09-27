"""P54 idea 2 (+3b) on the India/US holdout, vs sub44r's out-of-fold decision on both halves (P49 load: p'' from P44's
half-0 fit on half-1 S1, from swapfit's half-1 fit on half-0 S1; acc = argmax & p'' >= 0.71).

  fill   S1 empty under sub44r: add its best free argmax non-sibling (hn_cls != 1) pair when p'' >= b
  model  the same candidate, gated by a small S1-level LightGBM regression of the fill's F0.5 gain
         (y * F(1 of n_true) - [n_true == 0]) on best p'', 2nd p'', #argmax rows, #candidates, name frequency, noaddr...;
         fit on one half, fill iff predicted gain > theta (theta 0 = break-even)
  dupgrp (3b) content-identical SX groups (train, lower/strip key) with no accepted copy: every copy goes to the group's
         best free non-sibling argmax S1 when p'' >= b
Per-S1 F0.5 deltas, paired SE, LB-eq = 0.468 India + 0.383 US; b picked on one half, reported on the other.

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/fill54.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dup54  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
W = ROOT / "data/work"
T = 0.71
LB_W = {"India": 0.468, "US": 0.383}
BS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
FEATS = ["pbest", "p2", "n_am", "s1_ncand", "freq", "noaddr", "hn_cls", "name_sup", "n_twins", "cat", "p8", "lr"]


def fb(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / ntrue
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(c):
    S = pd.read_parquet(W / f"p44ms/S1_{c}.parquet")
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "g", "half", "y", "hn_cls", "noaddr", "freq", "s1_ncand",
                                                               "name_sup", "n_twins", "cat", "p", "p_r_all"])
    p0, p1 = np.load(W / f"p44ms/ho_p44r_{c}.npy"), np.load(W / f"p49/ho_p44r_h1fit_{c}.npy")
    A["pa"] = np.where(A.half.to_numpy() == 1, p0, p1)
    A["acc"] = A.pa >= T
    n = len(S)
    g, y, acc = A.g.to_numpy(), A.y.to_numpy(), A.acc.to_numpy()
    return dict(S=S, A=A, n=n, tp=np.bincount(g[acc & y], minlength=n), npred=np.bincount(g[acc], minlength=n),
                nt=S.n_true.to_numpy(), half=S.half.to_numpy().astype(np.int64))


def report(D, dfs):
    """dfs[c] = per-S1 F delta array; LB-eq mean and SE per half."""
    out = {}
    for h in (0, 1):
        m = {c: dfs[c][D[c]["half"] == h] for c in LB_W}
        v = sum(LB_W[c] * m[c].mean() for c in LB_W)
        se = np.sqrt(sum((LB_W[c] * m[c].std() / np.sqrt(len(m[c]))) ** 2 for c in LB_W))
        out[f"h{h}"] = {"lbeq": round(float(v), 6), "se": round(float(se), 6), **{c: round(float(m[c].mean()), 6) for c in LB_W}}
    return out


def fill_frame(d):
    """Empty S1's best free non-sibling argmax row, plus S1-level features."""
    A, n = d["A"], d["n"]
    emp = d["npred"] == 0
    e = A[emp[A.g.to_numpy()]]
    o = e.sort_values(["g", "pa"], ascending=[True, False])
    o["rk"] = o.groupby("g").cumcount()
    p2 = o[o.rk == 1].set_index("g").pa
    nam = o.groupby("g").size()
    b = o[o.hn_cls != 1].groupby("g").head(1).set_index("g")
    b = b.assign(pbest=b.pa, p2=p2.reindex(b.index).fillna(0).to_numpy(), n_am=nam.reindex(b.index).to_numpy(),
                 p8=b.p, lr=b.p_r_all)
    nt = d["nt"][b.index.to_numpy()]
    b["gain"] = fb(b.y.astype(int), 1, nt) - (nt == 0)
    b["half"] = d["half"][b.index.to_numpy()]
    return b


def apply_adds(d, gi, ny, nadd):
    """Per-S1 F delta after adding nadd pairs (ny true) to S1 rows gi."""
    tp, npred = d["tp"].copy(), d["npred"].copy()
    np.add.at(tp, gi, ny)
    np.add.at(npred, gi, nadd)
    return fb(tp, npred, d["nt"]) - fb(d["tp"], d["npred"], d["nt"])


def pick(D, fn, grid):
    """fn(c, x) -> per-S1 delta; b picked on each half, reported on the other; plus the whole grid."""
    res = {"grid": {}}
    for x in grid:
        res["grid"][str(x)] = report(D, {c: fn(c, x) for c in LB_W})
    for fit, rep in ((0, 1), (1, 0)):
        best = max(grid, key=lambda x: res["grid"][str(x)][f"h{fit}"]["lbeq"])
        res[f"fit_h{fit}"] = {"x": best, **res["grid"][str(best)][f"h{rep}"]}
    return res


def dup_frame(c, d):
    """(3b) holdout SX with a content-identical copy; groups with no accepted copy."""
    A = d["A"]
    k = dup54.keys("train", norm=True)
    k = k[k.country.astype(str) == c]
    sx = dup54.eid(A.sx)
    a = A.assign(id=sx.to_numpy()).merge(k[["id", "key"]], on="id")
    gs = k.groupby("key").size().rename("gsize")
    a = a.join(gs, on="key")
    nacc = a.groupby("key").acc.transform("sum")
    z = a[(nacc == 0) & (a.hn_cls != 1)].sort_values("pa", ascending=False).drop_duplicates("key")
    return z                                   # one row per group: best row (g, pa, y), gsize = all train copies


def main():
    t0 = time.time()
    import lightgbm as lgb
    D = {c: load(c) for c in LB_W}
    out = {}
    F = {c: fill_frame(D[c]) for c in LB_W}
    for c in LB_W:
        b = F[c]
        out[f"frame_{c}"] = {"empty_s1": int((D[c]["npred"] == 0).sum()), "empty_nonsingleton": int(((D[c]["npred"] == 0) & (D[c]["nt"] > 0)).sum()),
                             "with_cand": len(b), "cand_nonsingleton": int((b.gain > -1).sum()), "best_true": int(b.y.sum()),
                             "calib": {f"{lo}-{hi}": [int(((b.pbest >= lo) & (b.pbest < hi)).sum()),
                                                      round(float(b.y[(b.pbest >= lo) & (b.pbest < hi)].mean()), 3) if ((b.pbest >= lo) & (b.pbest < hi)).any() else None,
                                                      round(float((D[c]["nt"][b.index[(b.pbest >= lo) & (b.pbest < hi)]] == 0).mean()), 3) if ((b.pbest >= lo) & (b.pbest < hi)).any() else None]
                                       for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .6), (.6, .71))}}
        print(c, out[f"frame_{c}"], flush=True)

    def fill(c, x):
        b = F[c][F[c].pbest >= x]
        return apply_adds(D[c], b.index.to_numpy(), b.y.astype(int).to_numpy(), np.ones(len(b), int))
    out["fill"] = pick(D, fill, BS)
    print("fill", json.dumps({k: v for k, v in out["fill"].items() if k != "grid"}), flush=True)

    # S1-level gate: pooled India + US, no country column
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50, lambda_l2=1.0, seed=0,
               deterministic=True, verbose=-1, n_estimators=200)
    allb = pd.concat([F[c].assign(cc=c) for c in LB_W])
    pred = {}
    for h in (0, 1):
        fit = allb[allb.half == h]
        m = lgb.LGBMRegressor(**prm).fit(fit[FEATS].astype(float), fit.gain)
        for c in LB_W:
            r = F[c][F[c].half != h]
            pred.setdefault(c, pd.Series(0.0, index=F[c].index))
            pred[c].loc[r.index] = m.predict(r[FEATS].astype(float))

    def model(c, th):
        b = F[c][pred[c] > th]
        return apply_adds(D[c], b.index.to_numpy(), b.y.astype(int).to_numpy(), np.ones(len(b), int))
    out["model"] = pick(D, model, [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    print("model", json.dumps({k: v for k, v in out["model"].items() if k != "grid"}), flush=True)

    # 3b: dup groups with no accepted copy
    Z = {c: dup_frame(c, D[c]) for c in LB_W}
    for c in LB_W:
        z = Z[c]
        out[f"dup_{c}"] = {"groups_no_acc": len(z), "best_true": int(z.y.sum()),
                           "calib": {f"{lo}-{hi}": [int(((z.pa >= lo) & (z.pa < hi)).sum()),
                                                    round(float(z.y[(z.pa >= lo) & (z.pa < hi)].mean()), 3) if ((z.pa >= lo) & (z.pa < hi)).any() else None]
                                     for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .71))}}
        print(c, out[f"dup_{c}"], flush=True)

    def dupf(c, x):
        z = Z[c][Z[c].pa >= x]
        return apply_adds(D[c], z.g.to_numpy(), (z.y.astype(int) * z.gsize).to_numpy(), z.gsize.to_numpy())
    out["dupgrp"] = pick(D, dupf, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print("dupgrp", json.dumps({k: v for k, v in out["dupgrp"].items() if k != "grid"}), flush=True)
    out["runtime_s"] = round(time.time() - t0)
    (HERE / "fill.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
