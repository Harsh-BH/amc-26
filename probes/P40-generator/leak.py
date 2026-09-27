"""P40: do copies of one entity sit close in file row order or in ID number? (train, S2 and S3). Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import W

gt = pd.read_parquet(W / "gt_pairs.parquet")
rng = np.random.default_rng(0)
for src in (2, 3):
    ids = pq.read_table(W / f"train_source{src}.parquet", columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    num = pd.Series(pd.Series(ids).str[3:].astype(np.int64).to_numpy(), index=ids)
    g = gt[gt.sx.str.startswith(f"S{src}-")].copy()
    g["pos"] = pos.reindex(g.sx).to_numpy()
    g["num"] = num.reindex(g.sx).to_numpy()
    g = g.sort_values(["s1", "pos"])
    same = g.s1.to_numpy()[1:] == g.s1.to_numpy()[:-1]
    dpos = np.diff(g.pos.to_numpy())[same]
    gn = g.sort_values(["s1", "num"])
    samen = gn.s1.to_numpy()[1:] == gn.s1.to_numpy()[:-1]
    dnum = np.diff(gn.num.to_numpy())[samen]
    rp = np.abs(rng.integers(0, len(ids), len(dpos)) - rng.integers(0, len(ids), len(dpos)))
    print(f"S{src}: {len(ids):,} rows, {same.sum():,} same-entity neighbour gaps. row gap median {np.median(dpos):,.0f} "
          f"(random pair {np.median(rp):,.0f}); share gap<=10: {(dpos <= 10).mean():.5f}; id gap median {np.median(dnum):,.0f}; "
          f"share id gap<=1000: {(dnum <= 1000).mean():.5f}")
    # the S1 file: is the S1 row order related to the S2 row order of its members?
    s1ids = pq.read_table(W / "train_source1.parquet", columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
    p1 = pd.Series(np.arange(len(s1ids)), index=s1ids)
    x = p1.reindex(g.s1).to_numpy() / len(s1ids)
    y = g.pos.to_numpy() / len(ids)
    print(f"   corr(S1 row pos, S{src} row pos) = {np.corrcoef(x, y)[0, 1]:.5f}; corr(S1 id, S{src} id) = "
          f"{np.corrcoef(pd.Series(g.s1.str[3:]).astype(np.int64), g.num)[0, 1]:.5f}")
