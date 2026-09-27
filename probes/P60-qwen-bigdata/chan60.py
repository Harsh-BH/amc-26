"""P60 train channel: the folds 1-3 pairs of P59's fold 1-9 CE channel, i.e. exactly the labelled pairs P38's Qwen
never saw (P38 = P13's channel = folds >= 4; fold 0 = #8c holdout, never touched; P59's val = 20k fold-1 band pairs,
excluded from its train). P38's LoRAs are warm-started on these, so each model ends up having seen all of folds 1-9.

    chan60.py   # -> data/work/p60/sm_data/train_<c>.parquet (ta, tb, y, p), rows of P59's train_<c> with fold <= 3

P59's train_<c> carries no fold column, so its row selection (build59.chan) is replayed on the fold/inband columns:
same concat order, same samples (random_state=0 depends only on the frame length) -> same rows, checked on y and p.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p60/sm_data"
COLS = ["is_true", "fold", "p", "inband"]


def sel(c: str) -> pd.DataFrame:
    """build59.chan's train rows for country c, in its order (fold, is_true, p)."""
    b = pd.concat([pd.read_parquet(W / f"p59/trainband12_{c}.parquet", columns=COLS),
                   pd.read_parquet(W / f"p4b/trainband_{c}.parquet", columns=COLS)], ignore_index=True)
    v = b[(b.fold == 1) & b.inband].sample(20_000, random_state=0)
    b = b.drop(v.index)
    ib, ob = b[b.inband], b[~b.inband]
    return pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=0)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in ("India", "US"):
        s = sel(c)
        t = pq.read_table(W / f"p59/sm_data/train_{c}.parquet").to_pandas()
        assert len(t) == len(s) and (t.y.to_numpy() == s.is_true.to_numpy()).all() \
            and np.allclose(t.p.to_numpy(), s.p.to_numpy()), f"{c}: replay != P59's train rows"
        new = s.fold.to_numpy() <= 3
        t[new].to_parquet(OUT / f"train_{c}.parquet", index=False, compression="zstd")
        print(c, f"P59 train {len(t):,} -> folds 1-3 {new.sum():,} (true {t.y[new].mean():.3f}, band "
              f"{s.inband.to_numpy()[new].mean():.3f}); folds>=4 {(~new).sum():,}; by fold",
              s.fold.value_counts().sort_index().to_dict(), flush=True)
        del s, t


if __name__ == "__main__":
    sys.exit(main())
