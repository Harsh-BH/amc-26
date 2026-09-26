"""P35: label-free twin support for address-less S2/S3 records, on any scored pair frame (s1, sx, p, noaddr, core, src).

A twin of an address-less SX `a` under candidate S1 X = another SX `b` that
  * is a candidate of the same X, has an address, and is CONFIDENT at X: X is b's argmax and p(b, X) >= tau1;
  * has a near-identical name core: rapidfuzz ratio(core_a, core_b) >= tau_sim.
twin_frame() returns one row per (X, a) pair with >= 1 twin: s1, sx, tw_p (best twin p), tw_sim (best sim),
n_tw (twins), tw_x (a twin from the other source exists).
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Indel

T = {"India": 0.71, "US": 0.71}                     # #8c's India/US t


def argmax_mask(sx: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = pd.factorize(sx)[0]
    o = np.lexsort((-p, q))
    m = np.zeros(len(p), bool)
    m[o[np.r_[True, q[o][1:] != q[o][:-1]]] if len(o) else o] = True
    return m


def twin_frame(d: pd.DataFrame, am: np.ndarray, tau1: float = 0.9, tau_sim: float = 0.9) -> pd.DataFrame:
    p = d.p.to_numpy()
    conf = am & (p >= tau1) & ~d.noaddr.to_numpy()
    A = d.loc[d.noaddr.to_numpy(), ["s1", "sx", "core", "src"]]
    B = d.loc[conf, ["s1", "sx", "core", "src", "p"]].rename(columns={"sx": "b", "core": "core_b", "src": "src_b"})
    j = A.merge(B, on="s1")
    j = j[j.sx.to_numpy() != j.b.to_numpy()]
    sim = np.fromiter((Indel.normalized_similarity(x, y) for x, y in zip(j.core, j.core_b)), float, len(j))
    j = j[sim >= tau_sim].assign(sim=sim[sim >= tau_sim])
    j["x"] = j.src.to_numpy() != j.src_b.to_numpy()
    return (j.groupby(["s1", "sx"], sort=False)
             .agg(tw_p=("p", "max"), tw_sim=("sim", "max"), n_tw=("b", "size"), tw_x=("x", "any")).reset_index())


if __name__ == "__main__":
    d = pd.DataFrame({"s1": ["X", "X", "Y", "Y"], "sx": ["a", "b", "a", "c"], "p": [0.3, 0.95, 0.6, 0.2],
                      "noaddr": [True, False, True, False], "core": ["abc foods", "abc foods", "abc foods", "abc foods"],
                      "src": [2, 3, 2, 3]})
    tw = twin_frame(d, argmax_mask(d.sx.to_numpy(), d.p.to_numpy()))
    assert tw[["s1", "sx"]].values.tolist() == [["X", "a"]] and tw.tw_p.iloc[0] == 0.95 and tw.tw_x.iloc[0], tw
    print("twins self-check ok")
