"""P45 full-coverage CE on test (SageMaker GPU job): builds P3's "name | addr" texts for one slice of a country's
keys on the instance, then runs P13's scoring path unchanged (p13sol.py = solution_p13.py, phase bge_ckpt).

Channel 'train' = amc26-p45/data/cetest/: keys/<c>.parquet (q1, qx, s1, sx), txt/<c>_s1.parquet + <c>_sx_NNN.parquet
(P3's test text files, q1 / qx = row positions, sx files in sorted order), stub/ (64-row holdout_/train_/val_/test_
files that bge_ckpt's planner and holdout step read).
hp: country, part [k, n] (contiguous k-th of n slices) + P13's (phases, ckpt, s3, deadline_s).
-> r/r_bge_pool_test_<country>.parquet (s1, sx, r) of the slice, in hp s3 (P13's writer).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def join(t: pa.Table) -> np.ndarray:                     # = rerank._join
    return np.array(pc.binary_join_element_wise(t["name"], t["addr"], pa.scalar(" | ", pa.large_string())).to_pylist(), dtype=object)


def texts(txt: Path, c: str, q1: np.ndarray, qx: np.ndarray):
    """= rerank.texts on the shipped text files."""
    u1, i1 = np.unique(q1, return_inverse=True)
    a = join(pq.read_table(txt / f"{c}_s1.parquet", columns=["name", "addr"]).take(u1))[i1]
    ux, ix = np.unique(qx, return_inverse=True)
    tx, off = np.empty(len(ux), dtype=object), 0
    for p in sorted(txt.glob(f"{c}_sx_*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        lo, hi = np.searchsorted(ux, [off, off + n])
        if hi > lo:
            tx[lo:hi] = join(pq.read_table(p, columns=["name", "addr"]).take(ux[lo:hi] - off))
        off += n
    assert all(x is not None for x in tx), "every qx inside the text files"
    return a, tx[ix]


def build(src: Path, work: Path, c: str, part) -> int:
    k = pd.read_parquet(src / "keys" / f"{c}.parquet")
    lo, hi = len(k) * part[0] // part[1], len(k) * (part[0] + 1) // part[1]
    k = k.iloc[lo:hi]
    ta, tb = texts(src / "txt", c, k.q1.to_numpy(), k.qx.to_numpy())
    work.mkdir(parents=True, exist_ok=True)
    for f in (src / "stub").glob("*.parquet"):
        shutil.copy(f, work / f.name)
    pd.DataFrame({"s1": k.s1.to_numpy(), "sx": k.sx.to_numpy(), "ta": ta, "tb": tb,
                  "p": np.full(len(k), 0.5, np.float32)}).to_parquet(work / f"test_{c}.parquet", index=False)
    return len(k)


def main(train_dir, test_dir, model_dir, out_dir, params):
    import p13sol
    c, part = params["country"], params.get("part", [0, 1])
    work = Path("/tmp/p45c")
    n = build(Path(train_dir), work, c, part)
    print(f"[p45c] {c} part {part}: {n:,} pairs -> {work}", flush=True)
    p13sol.main(work, test_dir, model_dir, out_dir, params)
