"""Score every TEST pair (data/work/p4/X/test_*.parquet, P4's 65 features) with the stage-1 cross-fit boosters m2_cf_A /
m2_cf_B, averaged in logit space: the test-side counterpart of the out-of-fold train scores a stage C trained on folds 2-5
will see. Output: data/work/p80/test_p/<xfile> (q1, qx, p_A, p_B, p), small (ids + 3 floats).
    score_test.py BOOSTER_DIR          (dir holding m2_cf_A.txt and m2_cf_B.txt, e.g. out/<job>/model)"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}


def main(bdir):
    import lightgbm as lgb
    t0 = time.time()
    bdir = Path(bdir)
    bs = {k: lgb.Booster(model_file=str(next(bdir.rglob(f"m2_cf_{k}.txt")))) for k in ("A", "B")}
    out = W / "p80/test_p"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((W / "p4/X").glob("test_*.parquet"))
    for f in files:
        cols = [c for c in pq.read_schema(f).names if c not in DROP]
        t = pq.read_table(f, columns=["q1", "qx"] + cols).to_pandas()
        X = t[cols].to_numpy(np.float32)
        pa_, pb_ = (bs[k].predict(X, num_iteration=bs[k].best_iteration or None) for k in ("A", "B"))
        lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        p = 1 / (1 + np.exp(-(lg(pa_) + lg(pb_)) / 2))
        pd.DataFrame({"q1": t.q1.to_numpy(), "qx": t.qx.to_numpy(), "p_A": pa_.astype(np.float32), "p_B": pb_.astype(np.float32),
                      "p": p.astype(np.float32)}).to_parquet(out / f.name, index=False)
        print(f"{f.name}: {len(t):,} pairs, corr(A,B) {np.corrcoef(pa_, pb_)[0, 1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        del t, X


if __name__ == "__main__":
    main(sys.argv[1])
