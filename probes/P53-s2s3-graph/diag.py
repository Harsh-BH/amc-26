"""P53 diagnostic: AUC of each triangle column for y among holdout argmax pairs, overall and in #34's decision band."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = Path("data/work")
sys.path.insert(0, "probes/P53-s2s3-graph")
from tri import FEATS
for c in ("India", "US"):
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "y", "pa", "p"])
    t = pd.read_parquet(W / f"p53/tri_train_{c}.parquet").drop_duplicates(["s1", "sx"])
    A = A.merge(t, on=["s1", "sx"], how="left")
    A["d_max"] = A.t_max.fillna(0) - A.c_max.fillna(0)
    band = (A.pa > 0.3) & (A.pa < 0.95)
    print(c, "rows", len(A), "band", int(band.sum()), "band y", round(A.y[band].mean(), 3))
    for k in FEATS + ["d_max", "pa"]:
        v = A[k].fillna(-1)
        r = {"auc_all": roc_auc_score(A.y, v), "auc_band": roc_auc_score(A.y[band], v[band]),
             "cov_band": A[k][band].notna().mean()}
        # residual: AUC within pa deciles in band
        q = pd.qcut(A.pa[band], 5, labels=False, duplicates="drop")
        aucs = [roc_auc_score(A.y[band][q == i], v[band][q == i]) for i in range(q.max() + 1)
                if A.y[band][q == i].nunique() == 2]
        r["auc_in_pa_quint"] = np.mean(aucs)
        print(f"  {k:8s}", {a: round(b, 3) for a, b in r.items()})
