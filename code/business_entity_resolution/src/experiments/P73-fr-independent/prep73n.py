"""P73 v3 train files: syn_<T>_v3n (gen73n.py, noise-word list) + prep73's pseudo(T) -> data/work/p73/ch/train_ind<T>n.parquet."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep73 as P  # noqa: E402
for t in ("India", "US", "France"):
    s = pd.read_parquet(P.W / f"p73/syn_{t}_v3n.parquet", columns=["ta", "tb", "y", "p", "kind"])
    d = pd.concat([s, P.pseudo(t)], ignore_index=True).sample(frac=1, random_state=73).reset_index(drop=True)
    d["y"] = d.y.astype(np.int64)
    assert d.ta.notna().all() and d.tb.notna().all() and set(d.y.unique()) <= {0, 1}
    d.to_parquet(P.OUT / f"train_ind{t}n.parquet", index=False)
    print(f"train_ind{t}n: {len(d):,} pairs, true {d.y.mean():.3f}", flush=True)
