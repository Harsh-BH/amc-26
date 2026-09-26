"""P26 laptop-side prep (light, <1 GB): per-record house numbers in q1/qx order + fold-0 truth counts, for the AWS job.
    .venv/bin/python probes/P26-france-da/prep.py
-> data/work/p26/sm/meta/{train_India,train_US,test_France}.npz  (h1: int64 per q1, hx: int64 per qx, -1 = none;
   l1/lx: digit count; train also t0_q1, t0_n = every fold-0 S1 and its true-match count, 0 = singleton)
France = P14's IDF-fixed world (data/work/p14/w_idfx), whose X / pruned keys index its own P3 text files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p26/sm/meta"


def ids(txt: Path, c: str):
    s1 = pq.read_table(txt / f"{c}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    fs = sorted(txt.glob(f"{c}_sx_*.parquet"))
    sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in fs])
    return s1, sx


def hn(rf: Path, want: pa.Array):
    t = pq.read_table(rf, columns=["id", "hn"])
    pos = pc.index_in(want, value_set=t.column("id")).to_numpy(zero_copy_only=False)
    assert not pd.isna(pos).any(), f"{rf.name}: ids without rule features"
    s = pd.Series(t.column("hn").to_numpy(zero_copy_only=False)[pos.astype(np.int64)])
    s = s.where(s.str.len().between(1, 9), "")                  # 10+ digit "house numbers" are junk; int64-safe
    v = pd.to_numeric(s.replace("", "-1")).to_numpy(np.int64)
    return v, s.str.len().to_numpy(np.int8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["s1"]).s1.value_counts()
    fo = pd.read_parquet(W / "s1_folds.parquet").set_index("entity_id").fold
    for split, c, txt, wd in (("train", "India", W / "p3/txt", W), ("train", "US", W / "p3/txt", W),
                              ("test", "France", W / "p14/w_idfx/p3/txt_test", W / "p14/w_idfx")):
        s1, sx = ids(txt, c)
        h1, l1 = hn(wd / f"rf_{split}_s1_{c}.parquet", s1)
        hx, lx = hn(wd / f"rf_{split}_sx_{c}.parquet", sx)
        d = dict(h1=h1, l1=l1, hx=hx, lx=lx)
        if split == "train":
            e1 = pd.Series(s1.to_numpy(zero_copy_only=False))
            f = fo.reindex(e1).to_numpy()
            q = np.flatnonzero(f == 0)
            d |= dict(t0_q1=q.astype(np.int32), t0_n=gt.reindex(e1.iloc[q]).fillna(0).to_numpy(np.int32),
                      fold=f.astype(np.int8))
        np.savez(OUT / f"{split}_{c}.npz", **d)
        print(f"{split} {c}: S1 {len(h1):,} (hn {np.mean(h1 >= 0):.3f}), SX {len(hx):,} (hn {np.mean(hx >= 0):.3f})"
              + (f", fold-0 S1 {len(d['t0_q1']):,}, singletons {np.mean(d['t0_n'] == 0):.4f}" if split == "train" else ""),
              flush=True)


if __name__ == "__main__":
    main()
