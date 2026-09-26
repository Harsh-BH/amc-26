"""P53 diagnostic: sub44r-accepted argmax pairs (holdout, out-of-fold p'') whose SX's runner-up S1 has much stronger
triangle support: precision of the accepted pair."""
import numpy as np, pandas as pd
from pathlib import Path
W = Path("data/work")
for c in ("India", "US"):
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "y", "p", "pa", "half"])
    A["pr"] = np.load(W / f"p44ms/ho_p44r_{c}.npy")          # h0-fit p'' (in-sample on h0)
    t = pd.read_parquet(W / f"p53/tri_train_{c}.parquet").drop_duplicates(["s1", "sx"])
    A = A.merge(t, on=["s1", "sx"], how="left")
    acc = (A.pr >= 0.71) & (A.half == 1)
    tm = A.t_max.fillna(0)
    print(c, "h1 accepted", int(acc.sum()), "precision", round(A.y[acc].mean(), 4))
    for v in (0.3, 0.5, 0.7):
        for m in (0.0, 0.2, 0.4):
            s = acc & (A.c_pmax.fillna(0) >= v) & (A.c_pmax.fillna(0) - tm >= m)
            print(f"  c_pmax>={v} margin>={m}: n={int(s.sum()):5d} precision {A.y[s].mean() if s.any() else 0:.3f}")
