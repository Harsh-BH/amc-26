"""Test pairs that need the full-coverage CE: #8c's argmax pairs (prob_test_<c>_a10 am) with no band r (te_pairs r NaN)
-> data/work/p45/cetest/keys/<c>.parquet (q1, qx = rows of P3's test text files; s1, sx). Band pairs keep P13's r."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import rerank as rr

OUT = ROOT / "data/work/p45/cetest/keys"
OUT.mkdir(parents=True, exist_ok=True)
for c in ("India", "US"):
    a = pq.read_table(ROOT / f"data/work/p34/prob_test_{c}_a10.parquet", columns=["s1", "sx", "am"])
    t = pq.read_table(ROOT / f"data/work/p34/te_pairs_{c}.parquet", columns=["s1", "sx", "r"])
    assert a.column("s1").equals(t.column("s1")) and a.column("sx").equals(t.column("sx")), c
    m = a.column("am").to_numpy() & np.isnan(t.column("r").to_numpy(zero_copy_only=False))
    s1 = a.column("s1").to_pandas()[m].reset_index(drop=True)
    sx = a.column("sx").to_pandas()[m].reset_index(drop=True)
    del a, t
    q1, qx = rr.rows("test", c, s1, sx)
    pd.DataFrame({"q1": q1.astype(np.int32), "qx": qx.astype(np.int32), "s1": s1, "sx": sx}).to_parquet(OUT / f"{c}.parquet", index=False)
    print(c, int(m.sum()), "argmax pairs without band r", flush=True)
