"""Channel for the full-coverage bge CE (P13 job, phase bge_ckpt) and the e5 bi-encoder job: EVERY #6-pruned holdout
pair (the rows / order of data/work/p23/pairs_<c>.parquet) with P3's "name | addr" texts -> data/work/p45/ce/
holdout_<c>.parquet (s1, sx, ta, tb, y, p = m2_big p). train_/val_/test_ files are 64-row stubs (bge_ckpt's planner
reads their sizes; their r outputs are ignored).
    cebuild.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import rerank as rr

OUT = ROOT / "data/work/p45/ce"
OUT.mkdir(parents=True, exist_ok=True)
for c in ("India", "US"):
    pp = pd.read_parquet(ROOT / f"data/work/p23/pairs_{c}.parquet", columns=["s1", "sx", "y", "p_m2"])
    k = pq.read_table(ROOT / f"data/work/p9/m2_holdout_{c}_d2b6569f_6a23c2ec_0.03.parquet", columns=["q1", "qx"])
    ta, tb = rr.texts("train", c, k.column("q1").to_numpy(), k.column("qx").to_numpy())
    d = pd.DataFrame({"s1": pp.s1.to_numpy(), "sx": pp.sx.to_numpy(), "ta": ta, "tb": tb,
                      "y": pp.y.to_numpy(np.int8), "p": pp.p_m2.to_numpy(np.float32)})
    d.to_parquet(OUT / f"holdout_{c}.parquet", index=False, compression="zstd")
    stub = d.sample(64, random_state=0)
    for nm in (f"train_{c}", f"val_{c}"):
        stub[["ta", "tb", "y", "p"]].to_parquet(OUT / f"{nm}.parquet", index=False)
    stub[["s1", "sx", "ta", "tb", "p"]].to_parquet(OUT / f"test_{c}.parquet", index=False)
    if c == "India":
        stub[["s1", "sx", "ta", "tb", "p"]].to_parquet(OUT / "test_France.parquet", index=False)
    print(c, len(d), d.ta.str.len().mean(), d.tb.str.len().mean(), flush=True)
    del d, ta, tb, pp
