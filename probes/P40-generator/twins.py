"""P40: S1 name twins (same rf core, same country, full train) vs singletons / member counts. Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT, W

gt = pd.read_parquet(W / "gt_pairs.parquet")
nmem = gt.groupby("s1").size()
for c in ("India", "US"):
    r = pd.read_parquet(W / f"rf_train_s1_{c}.parquet", columns=["id", "core"])
    r["k"] = r.groupby("core").id.transform("size")
    r["m"] = nmem.reindex(r.id).fillna(0).to_numpy()
    r["single"] = r.m == 0
    print(c, "S1:", len(r), "share with >=1 twin:", round((r.k > 1).mean(), 4))
    print("  singleton rate by twin count k:", r.groupby(r.k.clip(upper=5)).single.mean().round(4).to_dict())
    print("  mean members by k:", r.groupby(r.k.clip(upper=5)).m.mean().round(3).to_dict())
    # within twin groups of size 2: members of each
    t2 = r[r.k == 2].sort_values("core")
    a = t2.m.to_numpy().reshape(-1, 2)
    print("  k=2 twin groups:", len(a), " both singleton", round(((a == 0).all(1)).mean(), 4), " exactly one singleton",
          round(((a == 0).sum(1) == 1).mean(), 4))
    r[["id", "k", "m"]].to_parquet(OUT / f"s1twin_{c}.parquet", index=False)
