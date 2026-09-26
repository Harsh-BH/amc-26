"""P50 D: order-swap TTA readout. r_bge_poolck_holdout rows = 20k trained order then the same 20k swapped."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
d = pd.read_parquet("data/work/p50/d_sample.parquet", columns=["s1", "sx", "y", "p", "r", "p_r_all", "country"])
r = pd.read_parquet("data/work/p50/r_bge_poolck_holdout.parquet")
n = len(d)
assert len(r) == 2 * n and (r.s1.to_numpy()[:n] == d.s1.to_numpy()).all() and (r.s1.to_numpy()[n:] == d.s1.to_numpy()).all()
d["ro"], d["rs"] = r.r.to_numpy()[:n], r.r.to_numpy()[n:]
out = {"corr_logit": float(np.corrcoef(lg(d.ro), lg(d.rs))[0, 1]), "corr": float(np.corrcoef(d.ro, d.rs)[0, 1]),
       "ro_vs_p_r_all_maxdiff": float(np.abs(d.ro - d.p_r_all).max())}
for c, g in list(d.groupby("country")) + [("all", d)]:
    y = g.y.to_numpy()
    X0 = np.c_[lg(g.p), lg(g.ro)]
    X1 = np.c_[lg(g.p), lg(g.ro), lg(g.rs)]
    cv = lambda X: roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="decision_function"))
    out[c] = {"auc_r": roc_auc_score(y, g.ro), "auc_swap": roc_auc_score(y, g.rs), "auc_avg": roc_auc_score(y, (lg(g.ro) + lg(g.rs)) / 2),
              "auc_stack_p_r": cv(X0), "auc_stack_p_r_swap": cv(X1)}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("data/work/p50/d_eval.json", "w"), indent=1, default=float)
