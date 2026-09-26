"""LOCO bge r (P13, holdout band pairs, s1/sx strings) -> q1/qx per train country for the ceproxy job.
    .venv/bin/python probes/P26-france-da/prep_r.py  -> data/work/p26/sm/r/r_<tag>_<country>.parquet (q1, qx, r)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import ids, W

OUT = W / "p26/sm/r"
OUT.mkdir(parents=True, exist_ok=True)
tags = ("bge_locoUS", "bge_locoIndia")
for c in ("India", "US"):
    s1, sx = ids(W / "p3/txt", c)
    for t in tags:
        r = pd.read_parquet(W / f"p13/r/r_{t}_holdout.parquet")
        q1 = pc.index_in(pa.array(r.s1.to_numpy()), value_set=s1).to_numpy(zero_copy_only=False)
        qx = pc.index_in(pa.array(r.sx.to_numpy()), value_set=sx).to_numpy(zero_copy_only=False)
        ok = ~(pd.isna(q1) | pd.isna(qx))
        pd.DataFrame({"q1": q1[ok].astype(np.int32), "qx": qx[ok].astype(np.int32), "r": r.r.to_numpy()[ok].astype(np.float32)}
                     ).to_parquet(OUT / f"r_{t}_{c}.parquet", index=False)
        print(c, t, int(ok.sum()), "of", len(r))
