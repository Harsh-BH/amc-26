"""Lever 4: France vs US/India feature shift, label-free, on test X (one X file per country = a random SX subset,
every pair of those SX). Two populations: all kept pairs (pruner), and the accepted pairs (argmax per SX, p >= t).
Ranks features by standardised mean difference France - (US+India).

    .venv/bin/python probes/P14-france-shift/shift.py [t_France]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import model as M
from ledger import log

P = {c: ROOT / f"data/work/p9/m2_test_{c}_d2b6569f_6a23c2ec_0.03.parquet" for c in ("France", "India", "US")}


def load(c, nfiles=1):
    p = pd.read_parquet(P[c])
    out = []
    for f in M.F.xfiles("test", c)[:nfiles]:
        X = pd.read_parquet(f)
        X = X.merge(p, on=["q1", "qx"], how="inner")          # kept pairs only
        out.append(X)
    return pd.concat(out, ignore_index=True)


def main(tf=0.95):
    t0 = time.time()
    stats = {}
    for c in ("France", "India", "US"):
        X = load(c)
        t = tf if c == "France" else M_T
        am = M.argmax_sx(X.qx.to_numpy(), X.p.to_numpy())
        acc = am & (X.p.to_numpy() >= t)
        cols = [k for k in X.columns if k not in ("q1", "qx", "p")]
        stats[c] = {"all_mean": X[cols].mean(), "all_std": X[cols].std(), "acc_mean": X.loc[acc, cols].mean(),
                    "acc_std": X.loc[acc, cols].std(), "n_all": len(X), "n_acc": int(acc.sum()),
                    "band": float(((X.p[am] > 0.3) & (X.p[am] < 0.9)).mean())}
        del X
    rows = []
    for pop in ("all", "acc"):
        ref = (stats["India"][f"{pop}_mean"] + stats["US"][f"{pop}_mean"]) / 2
        sd = pd.concat([stats[c][f"{pop}_std"] for c in stats], axis=1).mean(axis=1).replace(0, np.nan)
        d = (stats["France"][f"{pop}_mean"] - ref) / sd
        for k in d.index:
            rows.append({"pop": pop, "feat": k, "France": stats["France"][f"{pop}_mean"][k],
                         "India": stats["India"][f"{pop}_mean"][k], "US": stats["US"][f"{pop}_mean"][k], "smd": d[k]})
    R = pd.DataFrame(rows)
    R["abs"] = R.smd.abs()
    R.to_csv(ROOT / "data/work/p14/shift.csv", index=False)
    pd.set_option("display.width", 200)
    for pop in ("all", "acc"):
        print(f"--- {pop} pairs (n France {stats['France']['n_' + pop]:,}, India {stats['India']['n_' + pop]:,}, "
              f"US {stats['US']['n_' + pop]:,})")
        print(R[R["pop"] == pop].sort_values("abs", ascending=False).head(25).round(3).to_string(index=False))
    top = R[R["pop"] == "acc"].sort_values("abs", ascending=False).head(10)
    log("exp", probe="P14-france-shift", subproblem="france-shift", name="feature-shift-test", world="test-1file",
        metrics={**{f"smd_acc_{r.feat}": round(r.smd, 3) for r in top.itertuples()},
                 **{f"band03_09_{c}": round(stats[c]["band"], 4) for c in stats}},
        config={"t_France": tf, "t_other": M_T, "files_per_country": 1, "booster": "m2_big", "pruner_tau": 0.03},
        hypothesis="France pairs differ from US/India on a few features; the big shifts point at what to fix",
        verdict="keep", runtime_s=round(time.time() - t0), peak_gb=None)


M_T = 0.69
if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.95)
