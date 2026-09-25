"""Pair matcher behind ONE interface: score(pairs, f1, fx) -> np.ndarray of match probabilities, one per pair.

    pairs  DataFrame: q1, qx = row positions into f1 / fx; block_score, block_rank_s1, block_rank_sx, via from P3
           blocking (NaN / -1 / "" for a pair that only the rule keys proposed); acc = in the rules.match accept set
    f1, fx rules.feats frames of the S1 pool and of the S2/S3 records (id, core, web, is_web, state, city, hn, places)

STUB until P4's matcher lands (probes/P4-match): logistic regression on the block features + rule flags. run.py
fits it on the dev world (train fold 1) whenever data/work/p8/model_stub.pkl is absent, and stores the global
threshold of the fallback decision rule next to it (T).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as R
from worlds import WORK

ARTIFACT = WORK / "p8" / "model_stub.pkl"
FEATS = ["in_p3", "block_score", "log_rank_s1", "log_rank_sx", "via_r", "via_f", "key", "web", "addr_sc",
         "addr_ok", "acc"]
_M = None


def pair_feats(pairs: pd.DataFrame, f1: pd.DataFrame, fx: pd.DataFrame, chunk: int = 1_000_000) -> np.ndarray:
    """(n, len(FEATS)) float32. Rule flags are recomputed per pair on aligned rows, 1M pairs at a time."""
    out = np.zeros((len(pairs), len(FEATS)), np.float32)
    for i in range(0, len(pairs), chunk):
        p = pairs.iloc[i:i + chunk]
        a, b = f1.iloc[p.q1.to_numpy()].reset_index(drop=True), fx.iloc[p.qx.to_numpy()].reset_index(drop=True)
        core, web = a.core.to_numpy(), b.web.to_numpy()
        sc = R.addr_score(a, b)
        via = p.via.astype(object)
        rank = lambda r: np.log1p(np.where(r.to_numpy() >= 0, r.to_numpy(), 30))
        out[i:i + chunk] = np.column_stack([
            p.block_score.notna(), p.block_score.fillna(0), rank(p.block_rank_s1), rank(p.block_rank_sx),
            via.isin(["r", "r+f"]), via.isin(["f", "r+f"]),
            (core == b.core.to_numpy()) & (core != ""),
            b.is_web.to_numpy() & (web == a.core.str.replace(" ", "", regex=False).to_numpy()) & (b.web.str.len() >= 4).to_numpy(),
            sc, sc >= 2, p.acc.to_numpy()]).astype(np.float32)
    return out


def fit(X: np.ndarray, y: np.ndarray):
    """X = pair_feats rows, y = is-true-pair."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000).fit(X, y)


def save(lr, t: float):
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_bytes(pickle.dumps({"lr": lr, "t": t, "feats": FEATS}))


def load() -> dict:
    global _M
    if _M is None:
        _M = pickle.loads(ARTIFACT.read_bytes())
    return _M


def score(pairs: pd.DataFrame, f1: pd.DataFrame, fx: pd.DataFrame) -> np.ndarray:
    return load()["lr"].predict_proba(pair_feats(pairs, f1, fx))[:, 1].astype(np.float32)
