"""Learned pruning (meta-blocking, P10-candidates) between P3 blocking and the matcher (submission #6).

A small LightGBM (300 trees x 31 leaves) on 30 of the 65 pair features -- blocking-level signals plus name / address
similarities -- scores every blocked pair; a pair reaches the matcher (m2_big + the cross-encoder) only if q >= TAU,
or it is a rule-accepted pair (guard rail). The X context columns (s1_* / sx_*, block ranks) stay computed over the
FULL blocking list, so m2_big's p on a kept pair is exactly the p it had before pruning -- the pruner only removes
pairs. Test: 15.87 -> 4.83 candidates per S1 at a holdout cost of -0.00024 F0.5 (P10 RESULT.md).

    python code/business_entity_resolution/src/prune.py train   # train folds 2,3 (never 0/1) -> data/work/p10/pruner.txt
    python code/business_entity_resolution/src/prune.py         # self-check

    q = score(load_model(), X)          # X: DataFrame with FEATS
    keep = prune(X, q)                  # bool mask
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from worlds import WORK

MODEL = WORK / "p10" / "pruner.txt"
F17 = ["block_score", "block_rank_s1", "block_rank_sx", "via_r", "via_f", "s1_n", "sx_n", "s1_margin", "sx_margin",
       "sx_gap", "core_eq", "core_jacc", "hn_eq", "hn_diff", "city_eq", "state_eq", "rule_acc"]    # blocking-level only
# + name / address string similarities: at equal budget they cut the dropped selected pairs 1,278 -> 57 (P10)
FEATS = F17 + ["name_tsort", "name_tset", "name_cos", "core_tset", "addr_cos", "addr_jacc", "addr_empty_x",
               "core_ntok_x", "core_tsort", "addr_tset", "is_web_x", "web_eq", "indic_x"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=0, deterministic=True)
ROUNDS = 300
TAU = 0.03          # the Pareto knee (P10): test 4.83 candidates/S1, F0.5 -0.00024 vs #5
_M = None


def train(folds=(2, 3), feats=FEATS, rounds=ROUNDS, out=MODEL, params=PARAMS):
    """Every candidate pair of the train S1 in `folds` (the train X files, countries in train_matcher's order)."""
    import lightgbm as lgb
    import train_matcher
    assert not {0, 1} & set(folds), "never train on folds 0/1"
    t0 = time.time()
    Xs, ys = [], []
    for c in train_matcher.countries():
        for p in F.xfiles("train", c):
            t = pq.read_table(p, columns=feats + ["is_true"], filters=[("fold", "in", list(folds))])
            Xs.append(np.column_stack([t.column(f).to_numpy().astype(np.float32) for f in feats]))
            ys.append(t.column("is_true").to_numpy().astype(np.float32))
            del t
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys
    m = lgb.train(params, lgb.Dataset(X, y, feature_name=feats), rounds)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out))
    print(f"pruner: {len(y):,} pairs ({y.mean():.3f} true) folds {folds}, {len(feats)} feats, {rounds} trees, "
          f"{time.time() - t0:.0f}s -> {out}", flush=True)
    return m


def load_model(path=MODEL):
    """The pruner; trained from the train X files (train_matcher.prepare builds them) when the file is absent."""
    global _M
    if _M is None:
        import lightgbm as lgb
        if not path.exists():
            import train_matcher
            train_matcher.prepare()
            train(out=path)
        _M = lgb.Booster(model_file=str(path))
    return _M


def score(m, X: pd.DataFrame) -> np.ndarray:
    if not len(X):
        return np.zeros(0, np.float32)
    return m.predict(X[m.feature_name()].to_numpy(np.float32), num_threads=8).astype(np.float32)


def prune(X: pd.DataFrame, q: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Kept mask: pruner q >= tau, plus every rule-accepted pair (the rules' own precision is 0.99)."""
    return (q >= tau) | (X.rule_acc.to_numpy() > 0)


def _check():
    X = pd.DataFrame({"rule_acc": [0, 1, 0, 0]})
    assert prune(X, np.array([0.5, 0.0, 0.009, 0.01], np.float32), 0.01).tolist() == [True, True, False, True]
    print("prune self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL
        assert not out.exists(), f"{out} exists: pass another path (it is the pipeline's pruner)"
        train(out=out)
    else:
        _check()
