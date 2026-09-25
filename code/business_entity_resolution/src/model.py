"""M2 pair matcher (P4-match): LightGBM on the 65 features of features.py. Submission #5: booster
data/work/p4/m2_big.txt, trained on train folds 2-5 (#4: m2_ladder.txt, fold 2 only; run.py --recipe switches).
train_matcher.py rebuilds it from the train data when absent. Submission #6: only the pairs the pruner keeps
(prune.py, q >= TAU | rule_acc) are scored; TAU = None scores every blocked pair (#4 / #5).

    score(pairs, f1, fx) -> (keep, p)   in memory (dev world, injected frames): pairs = run.assemble's P3 ∪
                                    rule-accepted pairs, f1 / fx = features.rec tables; p for the kept pairs
    split_probs(split, c) -> (q1, qx, p)   full scale: every kept candidate pair of a country, from the X files
    argmax_sx(qx, p)                the decision's exclusivity: each S2/S3 only at its best S1
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from worlds import WORK

BOOSTER = WORK / "p4" / "m2_big.txt"      # #5; #4 = m2_ladder.txt
FOLDS = (2, 3, 4, 5)                       # the train folds BOOSTER is fitted on; #4 = (2,)
TAU = 0.03                                 # pruner threshold (#6); None = no pruning (#4 / #5)
CACHE = WORK / "p9"                     # M2 p per split / country, keyed by the booster's hash
DROP = {"s1", "sx", "q1", "qx", "y", "is_true", "fold", "country"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=12)
_M = None


def feats(X: pd.DataFrame) -> list:
    return [c for c in X.columns if c not in DROP]


def predict(model, X: pd.DataFrame) -> np.ndarray:
    if not len(X):
        return np.zeros(0, np.float32)
    return model.predict(X[model.feature_name()], num_threads=PARAMS["num_threads"]).astype(np.float32)


def load():
    global _M
    if _M is None:
        import lightgbm as lgb
        if not BOOSTER.exists():
            import train_matcher
            train_matcher.main(BOOSTER, FOLDS)
        _M = lgb.Booster(model_file=str(BOOSTER))
    return _M


def tag(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def argmax_sx(qx: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bool mask: the pair is its SX's highest-p candidate (ties -> the first row; one S1 per SX)."""
    o = np.lexsort((-p, qx))
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, qx[o][1:] != qx[o][:-1]]] if len(o) else o] = True
    return m


def kept(X: pd.DataFrame) -> np.ndarray:
    """The pruner's keep mask over X (all True when TAU is None)."""
    if TAU is None:
        return np.ones(len(X), bool)
    import prune
    return prune.prune(X, prune.score(prune.load_model(), X), TAU)


def score(pairs: pd.DataFrame, f1: pd.DataFrame, fx: pd.DataFrame) -> tuple:
    """(keep mask, M2 p of the kept pairs) for the complete candidate list of an in-memory pool (pairs: q1, qx,
    block columns, acc). The features see the full list, so the context columns are the unpruned ones."""
    if not len(pairs):
        return np.zeros(0, bool), np.zeros(0, np.float32)
    X = F.featurize(pairs, f1, fx, pairs.q1.to_numpy(), pairs.qx.to_numpy(), pairs.acc.to_numpy())
    keep = kept(X)
    return keep, predict(load(), X[keep])


def split_probs(split: str, c: str) -> pd.DataFrame:
    """q1, qx, p for every kept candidate pair of one country (P3 ∪ rule-accepted, P3 text rows), predicted from the
    X files (features.build writes the missing ones first; they hold the FULL blocking list). holdout = the train X
    files cut to every pair of the S2/S3 that a holdout (fold 0) S1 competes for, so each SX's argmax still sees all
    its train S1. Cached as data/work/p9/m2_<split>_<c>_<booster hash>[_<pruner hash>_<tau>].parquet."""
    # ponytail: keyed by the booster / pruner only -- delete data/work/p9/m2_* (and p4/X) whenever the features change
    if TAU is None:
        path = CACHE / f"m2_{split}_{c}_{tag(BOOSTER)}.parquet"
    else:
        import prune
        prune.load_model()                              # trains it first when absent
        path = CACHE / f"m2_{split}_{c}_{tag(BOOSTER)}_{tag(prune.MODEL)}_{TAU}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    xs = "train" if split == "holdout" else split
    F.build(xs, c)
    m, out = load(), []
    for p in F.xfiles(xs, c):
        X = pd.read_parquet(p)
        if split == "holdout":                          # a file holds every pair of its SX
            X = X[np.isin(X.qx.to_numpy(), X.qx.to_numpy()[X.fold.to_numpy() == 0])]
        X = X[kept(X)]
        out.append(pd.DataFrame({"q1": X.q1.to_numpy(np.int32), "qx": X.qx.to_numpy(np.int32), "p": predict(m, X)}))
        del X
        F.trim()
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame({"q1": np.zeros(0, np.int32),
                                                                    "qx": np.zeros(0, np.int32),
                                                                    "p": np.zeros(0, np.float32)})
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path, index=False)
    return d
