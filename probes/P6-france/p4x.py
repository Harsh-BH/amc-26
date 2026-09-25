"""P4's run.py stages (rf -> ctx -> feat) with output redirected to data/work/p6/p4out, single-process rules
features (2 GB cap). Same code, same format as data/work/p4/X, so P4's and these files are interchangeable.

    .venv/bin/python probes/P6-france/p4x.py train India
"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "p4-match/probes/P4-match")]
import featrun as R          # P4 feature runner (was run.py)

R.OUT = HERE.parents[1] / "data/work/p6/p4out"
_feats = R.rules._feats
R.rules.feats = lambda df, chunk=150_000, procs=1: pd.concat(
    [_feats(df.iloc[i:i + chunk]) for i in range(0, len(df), chunk)], ignore_index=True)
if __name__ == "__main__":
    split, c = sys.argv[1], sys.argv[2]
    if split == "train":      # train candidates stay P4's data/work/p4/cands_train_full_<c>
        R.cands = lambda s, cc: pd.read_parquet(HERE.parents[1] / f"data/work/p4/cands_train_full_{cc}.parquet",
                                                columns=["q1", "qx", *R.BLOCK, "is_true", "fold"])
    for step in sys.argv[3:] or ["rf", "ctx", "feat"]:
        {"rf": R.rf_step, "ctx": R.ctx_step, "feat": R.feat_step}[step](split, c)
