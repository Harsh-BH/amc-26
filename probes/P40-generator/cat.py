"""P40 step 1: tag every holdout-world (fold 0) true pair + every #34 candidate/miss error pair with its generator ops.
    probes/P40-generator/py.sh probes/P40-generator/cat.py tag      # -> data/work/p40/tp.parquet (s1, sx, c, ops)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ops as O  # noqa: E402

W = O.ROOT / "data/work"
OUT = W / "p40"
OUT.mkdir(exist_ok=True)


def world(name="holdout"):
    d = W / "worlds" / name
    s1 = pd.read_parquet(d / "s1.parquet").set_index("entity_id")
    sx = pd.read_parquet(d / "sx.parquet").set_index("entity_id")
    t = pd.read_parquet(d / "truth.parquet")
    return s1, sx, t


def tag_pairs(s1, sx, a, b, c=None) -> np.ndarray:
    r1, rx = s1.loc[a], sx.loc[b]
    return O.tag_frame(r1.business_name.to_numpy(), r1.business_address.to_numpy(), rx.business_name.to_numpy(),
                       rx.business_address.to_numpy(), r1.country.to_numpy() if c is None else c)


def tag_truth(n=None):
    s1, sx, t = world()
    if n:
        t = t.sample(n, random_state=0)
    t0 = time.time()
    parts = []
    for i in range(0, len(t), 100_000):
        g = t.iloc[i:i + 100_000]
        parts.append(tag_pairs(s1, sx, g.s1.to_numpy(), g.sx.to_numpy()))
        print(i, f"{time.time() - t0:.0f}s", flush=True)
    t = t.assign(c=s1.country.reindex(t.s1).to_numpy(), ops=np.concatenate(parts))
    if not n:
        t.to_parquet(OUT / "tp.parquet", index=False)
    return t


if __name__ == "__main__":
    if sys.argv[1] == "time":
        t0 = time.time()
        d = tag_truth(20_000)
        print(f"20k pairs {time.time() - t0:.1f}s")
        bits = np.array([[int(m) >> i & 1 for i in range(len(O.OPS))] for m in d.ops], bool)
        print(pd.Series(bits.mean(0), index=O.OPS).round(4).to_string())
    elif sys.argv[1] == "tag":
        tag_truth()
