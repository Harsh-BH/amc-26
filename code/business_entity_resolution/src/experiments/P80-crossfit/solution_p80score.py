"""P80 scoring job (SageMaker, CPU): score pairs with both stage-1 cross-fit M2 boosters (m2_cf_A / m2_cf_B), logit-averaged.
params what=train01: train channel = P4 train_* X files -> rows of folds 0 and 1 (holdout + early-stop fold) -> sc01_<file>
params what=test:    train channel = P4 test_* X files  -> every pair                                   -> sct_<file>
Boosters come in the test channel (m2_cf_A.txt, m2_cf_B.txt). Output columns: q1, qx, [fold, is_true], p_A, p_B, p."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}


def lg(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main(train_dir, test_dir, model_dir, out_dir, params):
    import lightgbm as lgb
    t0 = time.time()
    what = params.get("what", "test")
    bs = {k: lgb.Booster(model_file=str(Path(test_dir) / f"m2_cf_{k}.txt")) for k in ("A", "B")}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(train_dir).glob("*.parquet"))
    nt = os.cpu_count() or 8
    print(f"{what}: {len(files)} files, cpus {nt}", flush=True)
    for f in files:
        names = pq.read_schema(f).names
        cols = [c for c in names if c not in DROP]
        extra = [c for c in ("fold", "is_true") if c in names]
        t = pq.read_table(f, columns=["q1", "qx"] + extra + cols).to_pandas()
        if what == "train01":
            t = t[t.fold.isin([0, 1])]
        if not len(t):
            continue
        X = t[cols].to_numpy(np.float32)
        pa_ = bs["A"].predict(X, num_threads=nt)
        pb_ = bs["B"].predict(X, num_threads=nt)
        p = 1 / (1 + np.exp(-(lg(pa_) + lg(pb_)) / 2))
        o = t[["q1", "qx"] + extra].copy()
        o["p_A"], o["p_B"], o["p"] = pa_.astype(np.float32), pb_.astype(np.float32), p.astype(np.float32)
        o.to_parquet(out_dir / f"{'sc01' if what == 'train01' else 'sct'}_{f.name}", index=False)
        print(f"{f.name}: {len(t):,} pairs, corr(A,B) {np.corrcoef(pa_, pb_)[0, 1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        del t, X
    print(f"done {time.time() - t0:.0f}s", flush=True)
