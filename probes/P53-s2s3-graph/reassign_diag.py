"""P53 diagnostic: re-assignment headroom. For SX with >=2 candidate S1 (holdout world): a = #8c argmax, b = runner-up.
How often is b right and a wrong, as a function of triangle support d_t = t_max(b,y) - t_max(a,y) and p(b)?"""
import numpy as np, pandas as pd
from pathlib import Path
W = Path("data/work")
for c in ("India", "US"):
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "am"])
    t = pd.read_parquet(W / f"p53/tri_train_{c}.parquet", columns=["t_max", "t_best", "t_ncs"])
    d = pd.concat([d, t], axis=1)
    d["rk"] = d.groupby("sx").p.rank(ascending=False, method="first")
    a = d[d.rk == 1].set_index("sx"); b = d[d.rk == 2].set_index("sx")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[j.holdout_a | j.holdout_b]
    j["dt"] = j.t_max_b.fillna(0) - j.t_max_a.fillna(0)
    print(c, "SX with runner-up", len(j), " a right", round(j.y_a.mean(), 3), " b right", round(j.y_b.mean(), 3),
          " a wrong & b right", int((~j.y_a & j.y_b).sum()))
    for pb in (0.1, 0.3, 0.5):
        for m in (0.1, 0.2, 0.3, 0.5):
            s = j[(j.p_b >= pb) & (j.dt >= m)]
            print(f"  p_b>={pb} dt>={m}: n={len(s):6d}  b right {s.y_b.mean() if len(s) else 0:.3f}  a right {s.y_a.mean() if len(s) else 0:.3f}")
