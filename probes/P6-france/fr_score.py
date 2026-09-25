"""Score one test country's P4-format X files with a LightGBM booster and print its label-free output profile
(fr_diag.profile / anchor) across thresholds. Saves (q1, qx, p, rule_acc) to data/work/p6/prob_test_<c>_<model>.parquet.

    .venv/bin/python probes/P6-france/fr_score.py <booster.txt> France [US India]
    X files: data/work/p6/p4out/X/test_<c>_sx_*.parquet, else data/work/p4/X/test_<c>_sx_*.parquet
"""
import sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parents[1] / "data/work"
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import fr_diag as D

TS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def xfiles(c):
    """A complete X set (one file per P3 test SX text file) from p4x.py's or P4's output."""
    need = len(sorted((W / "p3/txt_test").glob(f"{c}_sx_*.parquet")))
    for d in (W / "p6/p4out/X", W / "p4/X"):
        f = sorted(d.glob(f"test_{c}_sx_*.parquet"))
        if len(f) == need:
            return f
    raise FileNotFoundError(f"no complete test X for {c} ({need} files)")


def score(model_path: Path, c: str) -> pd.DataFrame:
    out = W / f"p6/prob_test_{c}_{model_path.stem}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    m = lgb.Booster(model_file=str(model_path))
    parts = []
    for p in xfiles(c):
        X = pd.read_parquet(p, columns=list(dict.fromkeys(["q1", "qx", "rule_acc", *m.feature_name()])))
        parts.append(pd.DataFrame({"q1": X.q1.to_numpy(), "qx": X.qx.to_numpy(), "rule_acc": X.rule_acc.to_numpy(),
                                   "p": m.predict(X[m.feature_name()], num_threads=8).astype(np.float32)}))
        del X
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(out, index=False)
    return d


def report(d: pd.DataFrame, n_s1: int, n_sx: int, ts=TS) -> pd.DataFrame:
    q1, qx, p = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy()
    rows = [D.profile(q1, qx, p, n_s1, n_sx, t) | D.anchor(p, d.rule_acc.to_numpy(), t) for t in ts]
    return pd.DataFrame(rows).set_index("t")


def sizes(c):
    import pyarrow.parquet as pq
    txt = W / "p3/txt_test"
    n1 = pq.ParquetFile(txt / f"{c}_s1.parquet").metadata.num_rows
    nx = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(txt.glob(f"{c}_sx_*.parquet")))
    return n1, nx


if __name__ == "__main__":
    mp = Path(sys.argv[1])
    for c in sys.argv[2:]:
        t0 = time.time()
        d = score(mp, c)
        n1, nx = sizes(c)
        r = report(d, n1, nx)
        print(f"== {c} ({mp.stem}): {len(d):,} pairs, {n1:,} S1, {nx:,} SX, {time.time() - t0:.0f}s")
        print(r.round(4).to_string(), flush=True)
