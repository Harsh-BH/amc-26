"""House-number stage after the cross-encoder (#8c / #9s), from P15-research (hn_check, hn_stack, sib_rule) and
P17-hn-stack (stack.py).

The data generator makes orphan "siblings" of a business: an S2/S3 at the S1's house number + d, d in
{1,2,3,4,5,7,9,11,13,21} (P12), while noise on true pairs is a digit drop / add or +-1 / +-2. M2 only sees hn_eq /
hn_diff, so two stages use the SIGNED delta:

    stacked(m, X_of(q1, p, h1, hx))   residual LightGBM on the blended p (#8c: US only), X = logit p + house-number
                                      class / delta / equality + support among the S1's other candidates
    sib_mask(h1, hx)                  the zero-training rule (#9s: France): delta in {3,4,5,7,9,11,13,21} -> p := 0
    train(X, y, tr, out)              the stacker's fit: holdout S1 of P5's half 0, init_score = logit p (P17)
    .venv/bin/python code/business_entity_resolution/src/stack.py      # self-check
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worlds import WORK

SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}                  # the generator's sibling offsets (P12)
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])            # the ones noise on a true pair (almost) never produces
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=4,
           seed=0, deterministic=True, lambda_l2=1.0)
ROUNDS = 200
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def hn_of(split: str, side: str, c: str, ids: pa.Array, q: np.ndarray) -> np.ndarray:
    """House number ('' = none) of record rows q of ids (S1 pool or SX corpus), from the rules.feats cache."""
    import pandas as pd
    rf = pd.read_parquet(WORK / f"rf_{split}_{side}_{c}.parquet", columns=["id", "hn"])
    pos = pc.index_in(ids, value_set=pa.array(rf.id.to_numpy())).to_numpy(zero_copy_only=False)
    return rf.hn.fillna("").to_numpy(object)[pos.astype(np.int64)][q]


def hn_cls(h1: np.ndarray, hx: np.ndarray) -> np.ndarray:
    """0 none/equal, 1 sibling-like (+d, d in SIB), 2 digit drop/add (one is a suffix/prefix of the other),
    3 small negative / other small, 4 other."""
    out = np.zeros(len(h1), np.int8)
    for k, (a, b) in enumerate(zip(h1, hx)):
        if not a or not b or a == b:
            continue
        d = int(b) - int(a)
        if d in SIB:
            out[k] = 1
        elif len(a) != len(b) and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
            out[k] = 2
        elif abs(d) <= 21:
            out[k] = 3
        else:
            out[k] = 4
    return out


def support(q1: np.ndarray, p: np.ndarray, h1: np.ndarray, hx: np.ndarray) -> pd.DataFrame:
    """Collective features from the S1's other candidates: how many other SX of this S1 carry the same house number
    as this SX (n_same), the best p among them (p_same), and how many carry the S1's own number (n_s1hn)."""
    num = lambda v: pd.to_numeric(pd.Series(v).replace("", "-1")).to_numpy(np.int64)
    k = pd.DataFrame({"q1": np.asarray(q1, np.int64), "hx": num(hx), "h1": num(h1), "p": np.asarray(p, np.float32)})
    has = (k.hx >= 0).to_numpy()
    g = k.groupby(["q1", "hx"]).p
    n = g.transform("size").to_numpy() - 1
    mx = g.transform("max").to_numpy()
    k["r"] = g.rank(method="first", ascending=False).to_numpy()
    sec = k.loc[k.r == 2, ["q1", "hx", "p"]].rename(columns={"p": "s2"})
    s2 = k[["q1", "hx"]].merge(sec, on=["q1", "hx"], how="left").s2.to_numpy()   # left merge keeps row order
    p_same = np.where(k.p.to_numpy() >= mx, s2, mx)            # best p among the OTHER members
    eqh = ((k.hx == k.h1) & has).astype(np.int32)
    n_s1hn = eqh.groupby(k.q1).transform("sum").to_numpy() - eqh.to_numpy()
    return pd.DataFrame({"n_same": np.where(has, n, np.nan).astype(np.float32),
                         "p_same": np.where(has & (n > 0), p_same, np.nan).astype(np.float32),
                         "n_s1hn": n_s1hn.astype(np.float32)})


def X_of(q1: np.ndarray, p: np.ndarray, h1: np.ndarray, hx: np.ndarray) -> pd.DataFrame:
    """P17's X (arm 'sup') over the list the rows hold: lp, cls, d, eq + support."""
    both = (h1 != "") & (hx != "")
    dd = np.full(len(h1), np.nan, np.float32)
    dd[both] = np.clip(pd.to_numeric(pd.Series(hx[both])).to_numpy(np.float64)
                       - pd.to_numeric(pd.Series(h1[both])).to_numpy(np.float64), -100, 100)
    X = pd.DataFrame({"lp": lg(p).astype(np.float32), "cls": hn_cls(h1, hx).astype(np.float32), "d": dd,
                      "eq": (both & (h1 == hx)).astype(np.float32)})
    return pd.concat([X, support(q1, p, h1, hx)], axis=1)


def train(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, out: Path):
    """Residual LightGBM (200 x 31, lr 0.05, min_data_in_leaf 200, lambda_l2 1, init_score = logit p) on rows tr."""
    import lightgbm as lgb
    m = lgb.train(PRM, lgb.Dataset(X[tr], y[tr].astype(np.int8), init_score=X.lp.to_numpy()[tr],
                                   categorical_feature=["cls"]), ROUNDS)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out))
    return m


def stacked(m, X: pd.DataFrame) -> np.ndarray:
    if not len(X):
        return np.zeros(0, np.float32)
    return (1 / (1 + np.exp(-(X.lp.to_numpy(np.float64) + m.predict(X, raw_score=True))))).astype(np.float32)


def sib_mask(h1: np.ndarray, hx: np.ndarray) -> np.ndarray:
    """Generated-sibling pairs: both house numbers present and hn_x - hn_1 in SIB3."""
    ok = (h1 != "") & (hx != "")
    d = np.zeros(len(h1), np.int64)
    d[ok] = pd.to_numeric(pd.Series(hx[ok])).to_numpy() - pd.to_numeric(pd.Series(h1[ok])).to_numpy()
    return ok & np.isin(d, SIB3)


def _check():
    assert list(sib_mask(np.array(["10", "10", "10", "", "100"]), np.array(["13", "12", "7", "3", "121"]))) == \
        [True, False, False, False, True]
    h1, hx = np.array(["10", "10", "10", "10", ""], object), np.array(["13", "1", "10", "300", "4"], object)
    assert hn_cls(h1, hx).tolist() == [1, 2, 0, 4, 0]
    s = support(np.array([0, 0, 0, 1]), np.array([.9, .5, .7, .8], np.float32), np.array(["5", "5", "5", ""], object),
                np.array(["7", "7", "5", "7"], object))
    assert s.n_same.tolist() == [1, 1, 0, 0] and np.allclose(s.p_same[:2], [0.5, 0.9]) and s.n_s1hn.tolist() == [1, 1, 0, 0]
    X = X_of(np.array([0, 0]), np.array([.9, .2], np.float32), np.array(["10", "10"], object), np.array(["13", ""], object))
    assert X.columns.tolist() == ["lp", "cls", "d", "eq", "n_same", "p_same", "n_s1hn"] and X.d[0] == 3 and np.isnan(X.d[1])
    print("stack self-check ok")


if __name__ == "__main__":
    _check()
