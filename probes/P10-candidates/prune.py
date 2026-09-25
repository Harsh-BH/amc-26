"""P10 learned pruning (meta-blocking) between P3 blocking and the matcher.

A small LightGBM (300 trees x 31 leaves) on 30 of the pair features already in P4's X files -- blocking-level
signals plus name / address similarities -- scores every blocked pair; a pair reaches the matcher only if q >= TAU,
or it is a rule-accepted pair (guard rail). The X context columns (s1_* / sx_*, block ranks) stay computed over the
FULL blocking list, so m2_big's p on a kept pair is exactly the p it had before pruning -- the pruner only removes pairs.

    prune.py train          # folds 2,3 of train (never 0/1) -> data/work/p10/pruner.txt
    prune.py                # self-check

    from prune import load_model, score, prune
    q = score(load_model(), X)          # X: DataFrame with FEATS
    keep = prune(X, q)                  # bool mask
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
from prep import OUT, COUNTRIES, peak_gb, xfiles

MODEL = OUT / "pruner.txt"
F17 = ["block_score", "block_rank_s1", "block_rank_sx", "via_r", "via_f", "s1_n", "sx_n", "s1_margin", "sx_margin",
       "sx_gap", "core_eq", "core_jacc", "hn_eq", "hn_diff", "city_eq", "state_eq", "rule_acc"]    # blocking-level only
# + name / address string similarities: at equal budget they cut the dropped selected pairs 1,278 -> 57 (RESULT.md)
FEATS = F17 + ["name_tsort", "name_tset", "name_cos", "core_tset", "addr_cos", "addr_jacc", "addr_empty_x",
               "core_ntok_x", "core_tsort", "addr_tset", "is_web_x", "web_eq", "indic_x"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=0, deterministic=True)
ROUNDS = 300
TAU = 0.03          # the Pareto knee (RESULT.md): test 4.83 candidates/S1, F0.5 -0.00024 vs #5


def train(folds=(2, 3), feats=FEATS, rounds=ROUNDS, out=MODEL, params=PARAMS):
    import lightgbm as lgb
    assert not {0, 1} & set(folds), "never train on folds 0/1"
    t0 = time.time()
    Xs, ys = [], []
    for c in COUNTRIES:
        for p in xfiles("train", c):
            t = pq.read_table(p, columns=feats + ["is_true"], filters=[("fold", "in", list(folds))])
            Xs.append(np.column_stack([t.column(f).to_numpy().astype(np.float32) for f in feats]))
            ys.append(t.column("is_true").to_numpy().astype(np.float32))
            del t
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys
    m = lgb.train(params, lgb.Dataset(X, y, feature_name=feats), rounds)
    m.save_model(str(out))
    print(f"pruner: {len(y):,} pairs ({y.mean():.3f} true) folds {folds}, {len(feats)} feats, {rounds} trees, "
          f"{time.time() - t0:.0f}s, peak {peak_gb()} GB -> {out.name}", flush=True)
    return m


def load_model(path=MODEL):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))


def score(m, X: pd.DataFrame) -> np.ndarray:
    return m.predict(X[m.feature_name()].to_numpy(np.float32), num_threads=8).astype(np.float32)


def prune(X: pd.DataFrame, q: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Kept mask: pruner q >= tau, plus every rule-accepted pair (the rules' own precision is 0.99)."""
    return (q >= tau) | (X.rule_acc.to_numpy() > 0)


def _check():
    X = pd.DataFrame({"rule_acc": [0, 1, 0, 0]})
    assert prune(X, np.array([0.5, 0.0, 0.009, 0.01], np.float32), 0.01).tolist() == [True, True, False, True]
    print("prune self-check ok")


if __name__ == "__main__":
    {"train": lambda: train(), "check": _check}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
