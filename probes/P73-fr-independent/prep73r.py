"""P73 'r' train files (user, 17:15: 'use our training data also'): train_ind<T>n (v3 synthetic + pseudo) + a replay slice
of REAL labelled India/US band pairs from P13's train files, OTHER country only for a LOCO target (P67's REPLAY sizes).
France gets 30k + 30k. -> data/work/p73/ch/train_ind<T>r.parquet"""
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parents[2] / "data/work"
REPLAY = {"India": {"US": 60_000}, "US": {"India": 60_000}, "France": {"India": 30_000, "US": 30_000}}
for t, src in REPLAY.items():
    d = pd.read_parquet(W / f"p73/ch/train_ind{t}n.parquet")
    parts = [d]
    for c, k in src.items():
        r = pd.read_parquet(W / f"p13/sm_data/train_{c}.parquet")
        r = r[["ta", "tb", "y"] + (["p"] if "p" in r else [])].sample(min(k, len(r)), random_state=73)
        if "p" not in r: r["p"] = np.float32(0.5)
        parts.append(r)
    o = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=73).reset_index(drop=True)
    o["y"] = o.y.astype(np.int64)
    o.to_parquet(W / f"p73/ch/train_ind{t}r.parquet", index=False)
    print(f"train_ind{t}r: {len(o):,} pairs (replay {sum(src.values()):,} from {list(src)}), true {o.y.mean():.3f}", flush=True)
