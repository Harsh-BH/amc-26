"""How far and where does the shipped stacker move p? Residual f(X) quantiles, and the decision flips vs #7 binned by
#7's p (holdout: + the true share of the flipped pairs, holdout S1 only). Aggregates only.
    .venv/bin/python probes/P17-hn-stack/resid.py [arm] [json tag, e.g. sup-t0.71]
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else "sup"
TAG = sys.argv[2] if len(sys.argv) > 2 else arm
res = json.loads((S.OUT / f"holdout_{TAG}.json").read_text())
BINS = [0, 0.3, 0.5, 0.6, 0.71, 0.8, 0.9, 0.99, 1.01]
s1, t = S.holdout_s1()
for split in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("holdout", "test")):
    for c in ("India", "US"):
        d = S.frame(split, c)
        X = S.X_of(d, arm)
        f = lgb.Booster(model_file=str(S.OUT / f"stack_{TAG.split('-t')[0]}_{c}.txt")).predict(X, raw_score=True)
        s = (1 / (1 + np.exp(-(X.lp.to_numpy(np.float64) + f)))).astype(np.float32)
        a7 = S.M.argmax_sx(d["qx"], d["p"]) & (d["p"] >= S.T7)
        a = S.M.argmax_sx(d["qx"], s) & (s >= res[c]["t"])
        if split == "holdout":          # stats over the holdout S1's own rows (complete lists), like test
            pos, y, _, _ = S.labels(d, c, s1, t)
            ho = pos >= 0
        else:
            ho = np.ones(len(f), bool)
        X, f, a7, a = X[ho].reset_index(drop=True), f[ho], a7[ho], a[ho]
        q = np.quantile(f, [0.001, 0.01, 0.5, 0.99, 0.999])
        print(f"{split} {c}: f quantiles .001/.01/.5/.99/.999 {np.round(q, 2).tolist()}, |f|>3 {int((np.abs(f) > 3).sum()):,} "
              f"of {len(f):,}; accepted {int(a7.sum()):,} -> {int(a.sum()):,}; flips {int((a != a7).sum()):,} "
              f"({(a != a7).sum() / a7.sum():.4f} of #7's accepted)", flush=True)
        cm = X.cls.value_counts(normalize=True).sort_index().round(4).to_dict()
        sm = X.drop(columns=["lp", "cls", "d", "eq"]).mean().round(3).to_dict() if arm != "hn" else {}
        print(f"  cls mix {cm}; eq {X["eq"].mean():.3f}; support means {sm}", flush=True)
        b = pd.cut(d["p"][ho], BINS, right=False)
        tab = pd.DataFrame({"bin": b, "new_acc": a & ~a7, "new_rej": a7 & ~a})
        if split == "holdout":
            tab = tab.assign(y=y[ho])
            g = tab.groupby("bin", observed=False).apply(lambda z: pd.Series({
                "new_acc": int(z.new_acc.sum()), "new_acc_true": round(z.y[z.new_acc].mean(), 3) if z.new_acc.any() else np.nan,
                "new_rej": int(z.new_rej.sum()), "new_rej_true": round(z.y[z.new_rej].mean(), 3) if z.new_rej.any() else np.nan}))
        else:
            g = tab.groupby("bin", observed=False)[["new_acc", "new_rej"]].sum()
        print(g.to_string(), flush=True)
        del d, X, f, s
