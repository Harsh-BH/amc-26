"""P36 step 3 (holdout): the gib recovery pass on the #34 holdout decision, gated per the brief.
The pair scorer is fit on candidate rows whose S1 is a NON-holdout train S1 (disjoint from both holdout halves);
t_g is tuned on holdout half h and reported on half 1-h (and the swap), paired per-S1 SE, India / US, LB-eq.
    PYTHONPATH=probes/P36-iu-residual probes/P36-iu-residual/py.sh probes/P36-iu-residual/ho_gib.py
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from base import A, LB_W, OUT, W, paired, state
from recover import FEATS

PRM = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=100, lambda_l2=1.0, verbose=-1,
           num_threads=4, seed=0, deterministic=True, feature_fraction=0.9)
ROUNDS = 300
TS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def frame(c: str, s: dict) -> pd.DataFrame:
    g = pd.read_parquet(OUT / f"gc_train_{c}.parquet")
    gt = pq.read_table(W / "gt_pairs.parquet")
    gt = gt.filter(pc.is_in(gt["sx"], value_set=pa.array(g.sx.unique(), pa.large_string()))).to_pandas()
    key = lambda a, b: pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str)
    g["y"] = key(g.s1, g.sx).isin(set(key(gt.s1, gt.sx))).to_numpy()
    gi = pd.Series(np.arange(len(s["ids"])), index=s["ids"])
    g["g"] = gi.reindex(g.s1).to_numpy()                     # holdout S1 index, NaN = a train S1
    g["half"] = np.where(g.g.isna(), -1, s["h"][g.g.fillna(0).astype(int)].astype(np.int64))
    g["sx_true_any"] = g.sx.isin(set(gt.sx))
    return g


def decide(g: pd.DataFrame, p: np.ndarray, t: float) -> pd.DataFrame:
    """Each queried SX goes to its best-p_g S1 when p_g >= t."""
    o = g.assign(pg=p).sort_values(["sx", "pg"], ascending=[True, False]).drop_duplicates("sx")
    return o[o.pg >= t]


def delta(s: dict, adds: pd.DataFrame) -> np.ndarray:
    """Per-S1 F0.5 change of adding the (holdout-S1) pairs in adds to the #34 decision."""
    pr, n = s["pairs"], len(s["ids"])
    a = adds[adds.g.notna()]
    gg = a.g.astype(int).to_numpy()
    tp = np.bincount(pr.g[pr.acc & pr.y], minlength=n) + np.bincount(gg[a.y.to_numpy()], minlength=n)
    npred = np.bincount(pr.g[pr.acc], minlength=n) + np.bincount(gg, minlength=n)
    return A.fb(tp, npred, s["n_true"]) - s["f34"]


if __name__ == "__main__":
    import lightgbm as lgb
    sys.path.insert(0, str(A.ROOT))
    from ledger import log
    S, G, P = {}, {}, {}
    for c in LB_W:
        t0 = time.time()
        S[c] = state(c)
        G[c] = frame(c, S[c])
        g = G[c]
        e = pd.read_parquet(OUT / f"err_{c}.parquet")
        fn = e[e.type.isin(["block_addr", "lowp", "stolen", "pruner"])]
        hit = fn.merge(g[["s1", "sx", "rank"]], on=["s1", "sx"])
        print(f"{c}: {len(g):,} rows, {g.sx.nunique():,} SX, pos {int(g.y.sum()):,}; SX whose true S1 is anywhere in the "
              f"list {g[g.y].sx.nunique() / g[g.sx_true_any].sx.nunique():.3f} of the true-somewhere SX; "
              f"#34 FN pairs found: {len(hit)} (rank 0: {int((hit['rank'] == 0).sum())}) by type "
              f"{hit.type.value_counts().to_dict()}", flush=True)
    # one pooled scorer, fit on the non-holdout train S1 rows of both countries
    tr = pd.concat([G[c][G[c].half == -1] for c in LB_W])
    b = lgb.train(PRM, lgb.Dataset(tr[FEATS], tr.y.to_numpy()), ROUNDS)
    b.save_model(str(OUT / "grecover.txt"))
    imp = pd.Series(b.feature_importance("gain"), index=FEATS)
    print("gain importance", (imp / imp.sum()).round(3).sort_values(ascending=False).to_dict())
    for c in LB_W:
        P[c] = b.predict(G[c][FEATS])
    res = {}
    for fh in (0, 1):
        rep = 1 - fh
        # tune t on half fh (LB-eq over both countries), report on half rep
        curve = []
        for t in TS:
            d = {c: delta(S[c], decide(G[c], P[c], t)) for c in LB_W}
            curve.append((t, sum(LB_W[c] * paired(d[c], S[c]["h"], fh)[0] for c in LB_W)))
        t_best = max(curve, key=lambda x: x[1])[0]
        print(f"tune half {fh}: " + " ".join(f"{t:.2f}:{v:+.6f}" for t, v in curve), flush=True)
        out = {}
        for c in LB_W:
            ad = decide(G[c], P[c], t_best)
            d = delta(S[c], ad)
            m, se = paired(d, S[c]["h"], rep)
            ah = ad[ad.half == rep]
            out[c] = dict(d=m, se=se, adds=len(ah), prec=float(ah.y.mean()) if len(ah) else np.nan,
                          s1_changed=int((d[S[c]["h"] == rep] != 0).sum()))
        lb = sum(LB_W[c] * out[c]["d"] for c in LB_W)
        lse = np.sqrt(sum((LB_W[c] * out[c]["se"]) ** 2 for c in LB_W))
        res[fh] = dict(t=t_best, lb=lb, lb_se=lse, **out)
        print(f"fit(train S1) + t tuned h{fh} (t_g {t_best}) -> report h{rep}: LB-eq {lb:+.6f} ± {lse:.6f} ({lb / lse:.1f} SE) | "
              + " | ".join(f"{c} {out[c]['d']:+.6f} ± {out[c]['se']:.6f} adds {out[c]['adds']} prec {out[c]['prec']:.3f}"
                           for c in LB_W), flush=True)
    json.dump(res, open(OUT / "ho_gib.json", "w"), indent=1, default=float)
