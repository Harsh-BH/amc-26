"""Leakage check for r_all: ce_bge_pool.pt's training pairs (P13 channel train_/val_<c>: texts only) vs the holdout.
(1) fold labels of the holdout S1 (s1_folds) — P4b band() draws training pairs from folds >= 3 only;
(2) exact "name | addr" text overlap: holdout scored S1 texts vs CE training S1 texts (ta), holdout argmax SX vs tb;
(3) per-S1 gain of the r_all arm split by 'S1 has an argmax SX whose text is in CE training'."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import features as F
import rerank as rr

out = {}
folds = pd.read_parquet(ROOT / "data/work/s1_folds.parquet")
print("s1_folds columns", folds.columns.tolist(), flush=True)
for c in ("India", "US"):
    ids = pd.read_parquet(ROOT / f"data/work/p23/s1ctx_{c}.parquet", columns=["n_true"]).index
    fcol = [x for x in folds.columns if x != folds.columns[0]][0] if folds.index.name is None else folds.columns[0]
    fs = folds.set_index(folds.columns[0])[fcol] if folds.index.name is None else folds[fcol]
    fh = fs.reindex(ids)
    tr = pd.concat([pd.read_parquet(f"/tmp/p45_{k}_{c}.parquet", columns=["ta", "tb"]) for k in ("train", "val")])
    s1, sx = F.ids("train", c)
    pos = pd.Index(s1.to_numpy(zero_copy_only=False))
    q1 = pos.get_indexer(ids)
    ta_h, _ = rr.texts("train", c, q1, np.zeros(len(q1), np.int64))
    ho = pd.read_parquet(ROOT / f"data/work/p34/ho_{c}.parquet", columns=["s1", "sx"])
    qx = pd.Index(sx.to_numpy(zero_copy_only=False)).get_indexer(ho.sx)
    q1h = pos.get_indexer(ho.s1)
    _, tb_h = rr.texts("train", c, q1h, qx)
    s1_in = pd.Series(ta_h).isin(set(tr.ta)).to_numpy()
    sx_in = pd.Series(tb_h).isin(set(tr.tb)).to_numpy()
    out[c] = dict(holdout_s1=len(ids), fold_counts=fh.value_counts(dropna=False).to_dict(), s1_text_in_ce_train=int(s1_in.sum()),
                  argmax_pairs=len(ho), argmax_sx_text_in_ce_train=int(sx_in.sum()))
    pd.DataFrame({"s1": ho.s1, "sx_in": sx_in}).to_parquet(ROOT / f"data/work/p45/leak_sx_{c}.parquet", index=False)
    print(c, out[c], flush=True)
