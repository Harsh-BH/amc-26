"""P35: group decision for ADDRESS-LESS S2/S3 records. Every candidate row (X, a) of an address-less SX a gets
label-free group features; a residual LightGBM (init_score = logit p) re-scores them, then each address-less SX is
re-assigned to its best S1 by the new p' (kept if p' >= t). Rows of SX with an address are untouched.

Features (all from the scored pair frame itself + name cores; no labels):
  own:        lp, rank (0 = argmax), margin (lp - best other lp), n_cand
  twin at X:  tw_p, tw_sim, n_tw, tw_x (confident addressed SX of X with a near-identical core; tw_x = other source),
              tw_same (a same-source twin)
  X's load:   n_conf (confident SX at X, excluding a), n_conf_src (same source as a), n_conf_na (address-less ones),
              x_best (best p at X of other SX)
  group:      n_sup (candidates of a with a twin), sup_rank, n_hungry (candidates with n_conf == 0),
              n_same1 (candidates whose S1 core equals X's), sim1 (a's core vs X's core)
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Indel

from twins import argmax_mask, twin_frame

lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
FEATS = ["lp", "rank", "margin", "n_cand", "tw_p", "tw_sim", "n_tw", "tw_x", "tw_same", "n_conf", "n_conf_src",
         "n_conf_na", "x_best", "n_sup", "sup_rank", "n_hungry", "n_same1", "sim1"]


def features(d: pd.DataFrame, core1: pd.Series, t: float, tau1: float = 0.71, tau_sim: float = 0.85) -> pd.DataFrame:
    """d: s1, sx, p, noaddr, core, src (one country). core1: S1 id -> name core. Returns the address-less rows of d
    (original index kept) with FEATS."""
    p = d.p.to_numpy()
    am = argmax_mask(d.sx.to_numpy(), p)
    conf = am & (p >= t)
    na = d.noaddr.to_numpy()
    s1c = pd.factorize(d.s1)[0]
    ns1 = s1c.max() + 1
    # X's load, per S1 (counted over all its SX), minus the row itself
    n_conf = np.bincount(s1c, conf, ns1)
    src = d.src.to_numpy()
    n_conf2, n_conf3 = np.bincount(s1c, conf & (src == 2), ns1), np.bincount(s1c, conf & (src == 3), ns1)
    n_conf_na = np.bincount(s1c, conf & na, ns1)
    # best and 2nd-best p per S1 -> best p of OTHER SX at X
    o = np.lexsort((-p, s1c))
    first = np.r_[True, s1c[o][1:] != s1c[o][:-1]]
    b1 = np.zeros(ns1); b1[s1c[o][first]] = p[o][first]
    second = np.r_[False, ~first[1:]] & np.r_[True, first[:-1]]            # the row right after each group start
    b2 = np.zeros(ns1); b2[s1c[o][second]] = p[o][second]
    is_b1 = np.zeros(len(p), bool); is_b1[o[first]] = True
    x = d.loc[na, ["s1", "sx", "p", "core", "src"]].copy()
    ix = np.flatnonzero(na)
    g1 = s1c[ix]
    cf = conf[ix].astype(float)
    x["lp"] = lg(x.p.to_numpy())
    x["n_conf"] = n_conf[g1] - cf
    x["n_conf_src"] = np.where(src[ix] == 2, n_conf2[g1], n_conf3[g1]) - cf
    x["n_conf_na"] = n_conf_na[g1] - cf
    x["x_best"] = np.where(is_b1[ix], b2[g1], b1[g1])
    # own rank / margin among the SX's candidates
    q = pd.factorize(x.sx)[0]
    x["n_cand"] = np.bincount(q)[q]
    x["rank"] = x.groupby(q).p.rank(ascending=False, method="first").to_numpy() - 1
    lp = x.lp.to_numpy()
    o2 = np.lexsort((-lp, q)); f2 = np.r_[True, q[o2][1:] != q[o2][:-1]]
    top = np.full(q.max() + 1, -10.0); top[q[o2][f2]] = lp[o2][f2]
    s2 = np.r_[False, ~f2[1:]] & np.r_[True, f2[:-1]]
    sec = np.full(q.max() + 1, -10.0); sec[q[o2][s2]] = lp[o2][s2]
    x["margin"] = np.where(x["rank"].to_numpy() == 0, lp - sec[q], lp - top[q])
    # twin support at X
    tw = twin_frame(d.assign(noaddr=na), am, tau1, tau_sim)
    j = d.loc[conf & ~na, ["s1", "core", "src"]].rename(columns={"core": "core_b", "src": "src_b"})
    same = x[["s1", "sx", "core", "src"]].merge(j, on="s1")
    same = same[(same.src == same.src_b) &
                np.fromiter((Indel.normalized_similarity(a, b) >= tau_sim for a, b in zip(same.core, same.core_b)),
                            bool, len(same))]
    k = x[["s1", "sx"]].reset_index().merge(tw, on=["s1", "sx"], how="left").set_index("index")
    for col in ("tw_p", "tw_sim", "n_tw"):
        x[col] = k[col].fillna(0).to_numpy()
    x["tw_x"] = k.tw_x.fillna(False).astype(float).to_numpy()
    x["tw_same"] = pd.MultiIndex.from_frame(x[["s1", "sx"]]).isin(
        pd.MultiIndex.from_frame(same[["s1", "sx"]].drop_duplicates())).astype(float)
    sup = x.tw_p.to_numpy() > 0
    x["n_sup"] = np.bincount(q, sup)[q]
    x["sup_rank"] = np.where(sup, x.assign(s=np.where(sup, lp, -99)).groupby(q).s.rank(ascending=False, method="first")
                             .to_numpy() - 1, -1)
    hungry = x.n_conf.to_numpy() == 0
    x["n_hungry"] = np.bincount(q, hungry)[q]
    c1 = core1.reindex(x.s1).fillna("").to_numpy()
    x["sim1"] = np.fromiter((Indel.normalized_similarity(a, b) for a, b in zip(x.core, c1)), float, len(x))
    x["n_same1"] = pd.DataFrame({"q": q, "c": c1}).groupby(["q", "c"]).c.transform("size").to_numpy() - 1
    return x.drop(columns=["core"])


def decide(d: pd.DataFrame, x: pd.DataFrame, p_new: np.ndarray, t: float) -> np.ndarray:
    """pred mask over d: SX with an address keep #8c's argmax & p >= t; address-less SX take argmax of p_new."""
    p = d.p.to_numpy().copy()
    p[d.index.get_indexer(x.index)] = p_new
    return argmax_mask(d.sx.to_numpy(), p) & (p >= t)


if __name__ == "__main__":
    d = pd.DataFrame({"s1": ["X", "X", "Y", "Y", "Z"], "sx": ["a", "b", "a", "c", "a"],
                      "p": [0.3, 0.95, 0.6, 0.2, 0.1], "noaddr": [True, False, True, False, True],
                      "core": ["abc foods"] * 5, "src": [2, 3, 2, 3, 2]})
    x = features(d, pd.Series({"X": "abc foods", "Y": "abc food", "Z": "xyz"}), 0.71)
    assert list(x.index) == [0, 2, 4], x
    assert x.loc[0, "tw_p"] == 0.95 and x.loc[2, "tw_p"] == 0 and x.loc[0, "n_conf"] == 1 and x.loc[2, "n_conf"] == 0
    assert x.loc[2, "rank"] == 0 and x.loc[0, "rank"] == 1 and x.n_sup.tolist() == [1, 1, 1]
    assert x.n_hungry.tolist() == [2, 2, 2] and x.loc[0, "n_same1"] == 0
    pred = decide(d, x, np.array([0.9, 0.2, 0.1]), 0.71)
    assert pred.tolist() == [True, True, False, False, False], pred
    print("gdec self-check ok")
