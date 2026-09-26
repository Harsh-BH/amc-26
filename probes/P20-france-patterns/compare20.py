"""P20: label-free comparison of France probability files vs a base, at each file's t_eq_d4 (France empty-S1 rate =
D4's 6.85%, the best France LB operating point) unless a t is given: empty rate, matches per S1, S1 churn, and the
house-number lens on the pairs a variant adds / drops (hn equal = true-like, hn + SIB3 / +1,+2 = generated sibling:
siblings keep the hn only ~2-6% in France, 2.7% US). Aggregates only.
    .venv/bin/python probes/P20-france-patterns/compare20.py BASE.parquet[:t] VAR.parquet[:t] ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT)]
import model as M
from patterns import hn_rel

W = ROOT / "data/work"
N_S1, D4 = 259_452, 0.0685


def accepted(d: pd.DataFrame, t: float) -> pd.DataFrame:
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy())
    return d[am & (d.p.to_numpy() >= t)]


def t_eq_d4(d: pd.DataFrame) -> float:
    am = M.argmax_sx(pd.factorize(d.sx)[0], d.p.to_numpy())
    top = d[am].groupby("s1").p.max()
    ts = np.round(np.arange(0.80, 0.9951, 0.005), 3)
    e = np.array([1 - (top >= t).sum() / N_S1 for t in ts])
    return float(ts[np.abs(e - D4).argmin()])


_HN = None


def hn():
    global _HN
    if _HN is None:
        a = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"])
        b = pd.read_parquet(W / "rf_test_sx_France.parquet", columns=["id", "hn"])
        _HN = pd.concat([a, b]).set_index("id").hn
    return _HN


def lens(pairs: pd.DataFrame) -> dict:
    if not len(pairs):
        return {}
    h = hn()
    r = pd.Series([hn_rel(x, y) for x, y in zip(h.reindex(pairs.s1).fillna(""), h.reindex(pairs.sx).fillna(""))])
    return r.value_counts(normalize=True).round(3).to_dict()


def load(spec: str):
    f, _, t = spec.partition(":")
    d = pd.read_parquet(f, columns=["s1", "sx", "p"])
    return Path(f).stem.replace("prob_test_France_", ""), d, (float(t) if t else t_eq_d4(d))


def main(specs):
    bn, bd, bt = load(specs[0])
    ba = accepted(bd, bt)
    bl = ba.groupby("s1").sx.agg(lambda x: ",".join(sorted(x)))
    rows = [{"name": bn, "t": bt, "empty": 1 - len(bl) / N_S1, "mps": len(ba) / N_S1, "pairs": len(bd)}]
    print(rows[0], flush=True)
    for s in specs[1:]:
        n, d, t = load(s)
        a = accepted(d, t)
        al = a.groupby("s1").sx.agg(lambda x: ",".join(sorted(x)))
        u = bl.index.union(al.index)
        ch = int((bl.reindex(u).fillna("") != al.reindex(u).fillna("")).sum())
        k = lambda x: x.s1 + "|" + x.sx
        add, drop = a[~k(a).isin(set(k(ba)))], ba[~k(ba).isin(set(k(a)))]
        r = {"name": n, "t": t, "empty": 1 - len(al) / N_S1, "mps": len(a) / N_S1, "pairs": len(d), "s1_changed": ch,
             "added": len(add), "dropped": len(drop), "hn_added": lens(add), "hn_dropped": lens(drop)}
        rows.append(r)
        print(r, flush=True)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
