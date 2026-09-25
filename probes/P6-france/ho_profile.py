"""The same label-free profile / anchor numbers as fr_score.py, on the dense holdout (fold 0, argmax over all train S1)
of US and India, next to the TRUE profile -- the yardstick for reading France's.

    .venv/bin/python probes/P6-france/ho_profile.py data/work/p6/models/pooled.txt
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import loco_dense as LD
import fr_diag as D
from fr_score import TS

T4, M = LD.T4, LD.M


def scored(model_path: Path) -> pd.DataFrame:
    """fold-0 rows of every train pair: s1, sx (country-offset ids), p, is_true, am_all, rule_acc, country."""
    out = HERE.parents[1] / f"data/work/p6/S_holdout_{model_path.stem}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    m = lgb.Booster(model_file=str(model_path))
    ra = []

    def sc(X):
        ra.append(X.rule_acc.to_numpy()[X.fold.to_numpy() == 0])
        return M.predict(m, X)
    S = T4.score_folds(sc, folds=(0,))
    S["rule_acc"] = np.concatenate(ra)
    S.to_parquet(out, index=False)
    return S


if __name__ == "__main__":
    mp = Path(sys.argv[1])
    S = scored(mp)
    tn, grp = T4.truth_all(0)
    for ci, c in enumerate(T4.COUNTRIES):
        d = S[S.country.to_numpy() == ci]
        t = tn[(grp == c).to_numpy()]
        # re-index S1 to 0..n-1 over EVERY fold-0 S1 of the country (S1 without pairs count as empty)
        q1 = pd.Index(t.index).get_indexer(d.s1)
        qx = pd.factorize(d.sx)[0]
        am = d.am_all.to_numpy()
        rows = []
        for th in TS:
            keep = am & (d.p.to_numpy() >= th)
            k = np.bincount(q1[keep], minlength=len(t))
            rows.append({"t": th, "s1_empty": (k == 0).mean(), "matches_per_s1": k.mean(),
                         **{f"size_{s}": (k == s).mean() for s in (1, 2, 3, 4)}, "size_5+": (k >= 5).mean(),
                         **D.anchor(d.p.to_numpy(), d.rule_acc.to_numpy(), th),
                         "anchor_prec": float(d.is_true.to_numpy()[d.rule_acc.to_numpy() > 0].mean())})
        tv = t.to_numpy()
        true = {"s1_empty": (tv == 0).mean(), "matches_per_s1": tv.mean(),
                **{f"size_{s}": (tv == s).mean() for s in (1, 2, 3, 4)}, "size_5+": (tv >= 5).mean()}
        print(f"== {c} holdout ({mp.stem}); TRUE:", {k: round(float(v), 4) for k, v in true.items()})
        print(pd.DataFrame(rows).set_index("t").round(4).to_string(), flush=True)
