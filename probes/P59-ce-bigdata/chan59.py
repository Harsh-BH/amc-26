"""P59 scoring channels for the retrained CE (P13's bge_ckpt path, unchanged).

    chan59.py holdout   # P44's holdout frame A_<c> (every #8c argmax pair of a scored holdout S1) -> texts
                        # -> data/work/p59/ho_data/holdout_<c>.parquet (+ 64-row stubs for the planner)
    chan59.py testkeys  # EVERY test #8c argmax pair (prob_test_<c>_a10 am; = r_all_test_<c>'s rows)
                        # -> data/work/p59/cetest/keys/<c>.parquet (q1, qx, s1, sx) for P45's solution_p45c.py
"""
import gc, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P4b-rerank")]
import rerank as rr

W = ROOT / "data/work"
OUT = W / "p59"
STUB = W / "p45/cetest/stub"


def holdout():
    d = OUT / "ho_data"
    d.mkdir(parents=True, exist_ok=True)
    for f in STUB.glob("*.parquet"):
        if not f.name.startswith("holdout_"):
            shutil.copy(f, d / f.name)
    for c in ("India", "US"):
        A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["s1", "sx", "y", "p"])
        q1, qx = rr.rows("train", c, A.s1, A.sx)
        ta, tb = rr.texts("train", c, q1, qx)
        pd.DataFrame({"s1": A.s1.to_numpy(), "sx": A.sx.to_numpy(), "ta": ta, "tb": tb, "y": A.y.to_numpy(np.int8),
                      "p": A.p.to_numpy(np.float32)}).to_parquet(d / f"holdout_{c}.parquet", index=False, compression="zstd")
        print(c, len(A), "holdout argmax pairs, peak", round(rr.peak(), 2), "GB", flush=True)
        del A, q1, qx, ta, tb
        gc.collect()


def testkeys():
    d = OUT / "cetest/keys"
    d.mkdir(parents=True, exist_ok=True)
    for c in ("India", "US"):
        t = pq.read_table(W / f"p34/prob_test_{c}_a10.parquet", columns=["s1", "sx", "am"])
        t = t.filter(pc.field("am"))
        s1, sx = t.column("s1").to_pandas(), t.column("sx").to_pandas()
        del t
        ref = pq.read_table(W / f"p45/oof/r_all_test_{c}.parquet", columns=["s1", "sx"]).to_pandas()
        assert len(ref) == len(s1) and (ref.s1.to_numpy() == s1.to_numpy()).all() and (ref.sx.to_numpy() == sx.to_numpy()).all(), \
            f"{c}: argmax pairs != r_all_test rows"
        del ref
        q1, qx = rr.rows("test", c, s1, sx)
        pd.DataFrame({"q1": q1.astype(np.int32), "qx": qx.astype(np.int32), "s1": s1, "sx": sx}).to_parquet(d / f"{c}.parquet", index=False)
        print(c, len(s1), "test argmax pairs, peak", round(rr.peak(), 2), "GB", flush=True)
        del s1, sx, q1, qx
        gc.collect()


if __name__ == "__main__":
    {"holdout": holdout, "testkeys": testkeys}[sys.argv[1]]()
