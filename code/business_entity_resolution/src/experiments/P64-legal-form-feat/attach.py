"""P64: legal-form signatures for the rows of a q1/qx frame, from P3's cached texts (q1 / qx row order).
    .venv/bin/python probes/P64-legal-form-feat/attach.py <world> <c>   # A_<world>_<c> (P57) -> data/work/p64/lf_<world>_<c>.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path[:0] = ["probes/P64-legal-form-feat", "code/business_entity_resolution/src"]
import block as B  # noqa: E402
import features as F  # noqa: E402
import lf  # noqa: E402

OUT = Path("data/work/p64")


def names(split: str, c: str, q1: np.ndarray, qx: np.ndarray) -> tuple:
    """(S1 names at q1, SX names at qx) from the cached text tables."""
    s1 = pq.read_table(B.TXT[split] / f"{c}_s1.parquet", columns=["name"]).column(0).combine_chunks()
    n1 = s1.take(pa.array(q1)).to_numpy(zero_copy_only=False)
    del s1
    files, offs = F.sx_files(split, c)
    nx = np.empty(len(qx), object)
    for f, lo, hi in zip(files, offs[:-1], offs[1:]):
        m = (qx >= lo) & (qx < hi)
        if m.any():
            col = pq.read_table(f, columns=["name"]).column(0).combine_chunks()
            nx[m] = col.take(pa.array(qx[m] - lo)).to_numpy(zero_copy_only=False)
    return n1, nx


def sigs(n: np.ndarray) -> np.ndarray:
    u, inv = np.unique(n.astype(str), return_inverse=True)
    return lf.sig(pd.Series(u)).to_numpy(object)[inv]


def main(w: str, c: str, split: str = "train"):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_parquet(Path("data/work/p57") / f"A_{w}_{c}.parquet", columns=["q1", "qx"])
    n1, nx = names(split, c, A.q1.to_numpy(), A.qx.to_numpy())
    d = pd.DataFrame({"lf1": sigs(n1), "lfx": sigs(nx)})
    d["lf_cls"] = lf.pair_class(d.lf1, d.lfx)
    d.to_parquet(OUT / f"lf_{w}_{c}.parquet")
    print(c, w, len(d), d.lf_cls.value_counts().sort_index().to_dict(), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
