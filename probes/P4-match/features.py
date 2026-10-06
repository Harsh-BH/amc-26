"""P4 pair features. Record fields are computed once per record (rec), pair features are vectorised over aligned
row arrays: rapidfuzz cpdist for string similarities, sparse row-wise products for TF-IDF cosines and token /
place overlaps, numpy group stats for the candidate-list context. No per-pair Python loop.

    r1, rx = rec(s1_df), rec(sx_df)                       # raw rows -> record fields (P2 to_latin, P1 normalize)
    X = featurize(pairs, r1, rx)                          # pairs: s1, sx ids (+ block_score, ranks, via if blocked)
    .venv/bin/python probes/P4-match/features.py          # self-check

Country is never a feature (France is unseen); key counts come from the S1 pool being matched (`pool_counts`).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P3-block"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import block as B
import normalize as N
import rules

_INDIC = "[ऀ-෿]"
_TOK = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=2 ** 22, binary=True, norm=None,
                         alternate_sign=False, dtype=np.float32)
_PLACE = HashingVectorizer(analyzer=lambda s: s.split("|") if s else [], n_features=2 ** 22, binary=True,
                           norm=None, alternate_sign=False, dtype=np.float32)
_VECS = None


def vecs():
    """P3's hashed TF-IDF (train S2/S3 fit): name char-3grams and address words, for per-field cosines."""
    global _VECS
    if _VECS is None:
        import pickle
        _VECS = pickle.loads((ROOT / "data/work/p3/vecs_hash.pkl").read_bytes())
    return _VECS


def rec(df: pd.DataFrame, txt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw rows (entity_id, business_name, business_address, country) -> one row of fields per record.
    txt = P3 block.texts output for the same rows (data/work/p3/txt*), recomputed when None."""
    return assemble(rules._feats(df), B.texts(df, B.CFG) if txt is None else txt, df.business_name)


def assemble(rf: pd.DataFrame, txt: pd.DataFrame, raw_name: pd.Series) -> pd.DataFrame:
    """rules._feats output + P3 texts + raw names (all row-aligned) -> the record table featurize() reads."""
    f = rf.reset_index(drop=True).copy()
    # .array keeps pandas' arrow-backed strings (to_numpy() would build a Python str per row)
    f["name"], f["addr"], f["st"] = txt.name.array, txt.addr.array, txt.st.array
    f["acro"] = N.acronym(f.core).astype("str").array
    f["sq"] = f.core.str.replace(" ", "", regex=False)
    f["indic"] = raw_name.fillna("").str.contains(_INDIC, regex=True).to_numpy()
    return f


def pool_counts(r1: pd.DataFrame) -> pd.Series:
    """(country|core) -> number of S1 in the pool with that core: chain-ness, and the rule's key fan-out."""
    k = r1.country.astype(str) + "|" + r1.core.astype(str)
    return k[r1.core != ""].value_counts()


def _cos(A: sp.csr_matrix, Bm: sp.csr_matrix) -> np.ndarray:
    """Row-wise dot of aligned CSR rows."""
    return np.asarray(A.multiply(Bm).sum(1)).ravel().astype(np.float32)


def _l2(X):
    return B._l2(X.astype(np.float32).tocsr())


def mats(r: pd.DataFrame) -> dict:
    """Per-record sparse rows, computed once per record table: L2 TF-IDF of name / address (P3 vectorizers),
    binary core tokens, address tokens and places."""
    v = vecs()
    return {"name": _l2(v["name"].transform(r.name)), "addr": _l2(v["addr"].transform(r.addr)),
            "ctok": _TOK.transform(r.core).tocsr(), "atok": _TOK.transform(r.addr).tocsr(),
            "place": _PLACE.transform(r.places).tocsr()}


def _group_ctx(key: np.ndarray, sc: np.ndarray, pre: str) -> dict:
    """Per-group candidate count, best score, gap to best, margin over the best OTHER candidate of the group."""
    u, inv, n = np.unique(key, return_inverse=True, return_counts=True)
    best = np.full(len(u), -np.inf, np.float32)
    np.maximum.at(best, inv, sc)
    top = sc == best[inv]
    # second best: max over the rest once one top copy is removed
    first = np.zeros(len(sc), bool)
    o = np.flatnonzero(top)
    first[o[np.unique(inv[o], return_index=True)[1]]] = True
    sec = np.full(len(u), -np.inf, np.float32)
    np.maximum.at(sec, inv[~first], sc[~first])
    other = np.where(first, sec[inv], best[inv])
    return {f"{pre}_n": n[inv].astype(np.float32), f"{pre}_gap": (best[inv] - sc).astype(np.float32),
            f"{pre}_margin": np.where(np.isfinite(other), sc - other, 1.0).astype(np.float32)}


def block_cols(pairs: pd.DataFrame) -> dict:
    """P3 block features of each pair (NaN / 0 for pairs the blocker did not produce)."""
    n = len(pairs)
    get = lambda c: pairs[c].to_numpy(np.float32) if c in pairs else np.full(n, np.nan, np.float32)
    via = pairs.via.astype("category") if "via" in pairs else pd.Series(pd.Categorical([""] * n))
    return {"block_score": get("block_score"), "block_rank_s1": get("block_rank_s1"),
            "block_rank_sx": get("block_rank_sx"),
            "via_r": via.isin(["r", "r+f"]).to_numpy(np.float32), "via_f": via.isin(["f", "r+f"]).to_numpy(np.float32),
            "blocked": np.isfinite(get("block_score")).astype(np.float32)}


def context(pairs: pd.DataFrame, q1: np.ndarray, qx: np.ndarray) -> pd.DataFrame:
    """Block features + candidate-list context. Must see ALL pairs of every S1 / SX it scores (not a chunk).
    At full scale run.py computes the S1 side over the whole table and the SX side per SX file."""
    out = block_cols(pairs)
    s = np.nan_to_num(out["block_score"])
    return pd.DataFrame(out | _group_ctx(q1, s, "s1") | _group_ctx(qx, s, "sx"))


def _take(r: pd.DataFrame, col: str, ix: np.ndarray) -> list:
    return r[col].iloc[ix].tolist()


def _fz(scorer, a, b, **kw):
    return (cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / 100).astype(np.float32)


def _overlap(A, Bm):
    """|a & b|, |a|, |b| for aligned binary rows."""
    return _cos(A, Bm), np.asarray(A.sum(1)).ravel().astype(np.float32), np.asarray(Bm.sum(1)).ravel().astype(np.float32)


def strings(r1: pd.DataFrame, rx: pd.DataFrame, i: np.ndarray, j: np.ndarray, counts: pd.Series,
            m1: dict, mx: dict) -> pd.DataFrame:
    """String / field similarities for aligned rows i (in r1) and j (in rx); m1 / mx = mats(r1) / mats(rx)."""
    o = {}
    for col in ("name", "core"):
        a, b = _take(r1, col, i), _take(rx, col, j)
        o[f"{col}_ratio"] = _fz(fuzz.ratio, a, b)
        o[f"{col}_partial"] = _fz(fuzz.partial_ratio, a, b)
        o[f"{col}_tset"] = _fz(fuzz.token_set_ratio, a, b)
        o[f"{col}_tsort"] = _fz(fuzz.token_sort_ratio, a, b)
        o[f"{col}_jw"] = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
        if col == "core":
            aa, bb = np.array(a, object), np.array(b, object)
            o["core_eq"] = ((aa == bb) & (aa != "")).astype(np.float32)
            inter, na, nb = _overlap(m1["ctok"][i], mx["ctok"][j])
            o["core_jacc"] = inter / np.maximum(na + nb - inter, 1)
            o["core_inter"] = inter
            o["core_ntok_1"], o["core_ntok_x"] = na.astype(np.float32), nb.astype(np.float32)
            o["core_first_eq"] = np.array([x.split(" ", 1)[0] == y.split(" ", 1)[0] and x != ""
                                           for x, y in zip(a, b)], np.float32)
        o[f"{col}_len_1"] = np.array([len(x) for x in a], np.float32)
        o[f"{col}_len_x"] = np.array([len(x) for x in b], np.float32)
        del a, b
    o["name_cos"] = _cos(m1["name"][i], mx["name"][j])
    o["addr_cos"] = _cos(m1["addr"][i], mx["addr"][j])
    a, b = _take(r1, "addr", i), _take(rx, "addr", j)
    o["addr_tset"] = _fz(fuzz.token_set_ratio, a, b)
    o["addr_ratio"] = _fz(fuzz.ratio, a, b)
    inter, na, nb = _overlap(m1["atok"][i], mx["atok"][j])
    o["addr_jacc"] = inter / np.maximum(na + nb - inter, 1)
    o["addr_empty_1"], o["addr_empty_x"] = (na == 0).astype(np.float32), (nb == 0).astype(np.float32)
    del a, b
    # structured address fields (P1 address_fields via rules._feats)
    sc = np.zeros(len(i), np.float32)
    for col, w in (("hn", 4), ("city", 2), ("state", 1)):
        a, b = np.array(_take(r1, col, i), object), np.array(_take(rx, col, j), object)
        both = (a != "") & (b != "")
        eq = both & (a == b)
        o[f"{col}_eq"], o[f"{col}_both"] = eq.astype(np.float32), both.astype(np.float32)
        o[f"{col}_diff"] = (both & ~eq).astype(np.float32)
        sc += w * eq
        del a, b
    pin, o["place_n_1"], o["place_n_x"] = _overlap(m1["place"][i], mx["place"][j])
    o["place_inter"] = pin
    o["addr_rule"] = sc + (pin > 0)
    # websites, acronyms, keys
    web, isw = np.array(_take(rx, "web", j), object), rx.is_web.to_numpy(bool)[j]
    sq1, sqx = np.array(_take(r1, "sq", i), object), np.array(_take(rx, "sq", j), object)
    hw = web != ""
    o["web_x"] = hw.astype(np.float32)
    o["is_web_x"] = isw.astype(np.float32)
    o["web_eq"] = (hw & (web == sq1)).astype(np.float32)
    o["web_in"] = np.array([w != "" and s != "" and (w in s or s in w) for w, s in zip(web, sq1)], np.float32)
    o["web_ratio"] = np.where(hw, _fz(fuzz.ratio, web.tolist(), sq1.tolist()), np.nan).astype(np.float32)
    o["web_partial"] = np.where(hw, _fz(fuzz.partial_ratio, web.tolist(), sq1.tolist()), np.nan).astype(np.float32)
    o["sq_eq"] = ((sq1 == sqx) & (sq1 != "")).astype(np.float32)
    ac1, acx = np.array(_take(r1, "acro", i), object), np.array(_take(rx, "acro", j), object)
    o["acro"] = (((ac1 != "") & (ac1 == sqx)) | ((acx != "") & (acx == sq1))).astype(np.float32)
    k1 = pd.Series(_take(r1, "country", i)) + "|" + pd.Series(_take(r1, "core", i))
    kx = pd.Series(_take(rx, "country", j)) + "|" + pd.Series(_take(rx, "core", j))
    o["chain_1"] = k1.map(counts).fillna(0).to_numpy(np.float32)
    o["chain_x"] = kx.map(counts).fillna(0).to_numpy(np.float32)
    o["indic_x"] = rx.indic.to_numpy(bool)[j].astype(np.float32)
    o["is_s3"] = np.char.startswith(np.array(_take(rx, "id", j), dtype="U3"), "S3").astype(np.float32)
    return pd.DataFrame(o)


def featurize(pairs: pd.DataFrame, s1_df: pd.DataFrame, sx_df: pd.DataFrame, counts: pd.Series | None = None,
              chunk: int = 1_000_000, rule_acc: set | None = None, m1: dict | None = None,
              mx: dict | None = None) -> pd.DataFrame:
    """pairs: s1, sx ids (+ P3 block columns when blocked); s1_df / sx_df: rec() tables holding every id in pairs.
    counts: pool_counts over the full S1 pool being matched (defaults to s1_df). rule_acc: set of (s1, sx) accepted
    by rules.match against that pool. m1 / mx: mats() of s1_df / sx_df (pass to reuse across calls).
    Returns float32 features aligned with pairs."""
    i = pd.Index(s1_df.id).get_indexer(pairs.s1)
    j = pd.Index(sx_df.id).get_indexer(pairs.sx)
    assert (i >= 0).all() and (j >= 0).all(), "every pair id has a record"
    counts = pool_counts(s1_df) if counts is None else counts
    m1, mx = m1 or mats(s1_df), mx or mats(sx_df)
    X = context(pairs, i, j)
    parts = [strings(s1_df, sx_df, i[c:c + chunk], j[c:c + chunk], counts, m1, mx) for c in range(0, len(pairs), chunk)]
    X = pd.concat([X, pd.concat(parts, ignore_index=True)], axis=1)
    if rule_acc is not None:
        X["rule_acc"] = np.array([(a, b) in rule_acc for a, b in zip(pairs.s1, pairs.sx)], np.float32)
    return X.astype(np.float32)


def _check():
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "US", "India"],
                       "business_name": ["Twisted Deli LLC", "Metro Holding Co", "Sunrise Traders Pvt Ltd"],
                       "business_address": ["4 Dix Road, Woburn, MA", "3860 Pin Oak Road, Raleigh, NC",
                                            "12 MG Road, Pune, Maharashtra"]})
    sx = pd.DataFrame({"entity_id": ["S2-1", "S3-1", "S2-2"], "country": ["US", "US", "India"],
                       "business_name": ["twisteddeli.com", "TWISTED DELI", "सनराइज ट्रेडर्स"],
                       "business_address": ["##4 Dix Rd, <NULL>, Woburn", "4 DIX ROAD, WOBURN, MA", "Pune, महाराष्ट्र"]})
    pairs = pd.DataFrame({"s1": ["S1-1", "S1-1", "S1-2", "S1-3"], "sx": ["S2-1", "S3-1", "S3-1", "S2-2"],
                          "block_score": [.8, .9, .2, .5], "via": ["r", "r+f", "f", "r"]})
    X = featurize(pairs, rec(s1), rec(sx), rule_acc={("S1-1", "S3-1")})
    assert len(X) == 4 and X.dtypes.eq(np.float32).all()
    assert X.core_eq.tolist() == [0, 1, 0, 0] and X.rule_acc.tolist() == [0, 1, 0, 0], X[["core_eq", "rule_acc"]]
    assert X.web_eq[0] == 1 and X.hn_eq[1] == 1 and X.hn_eq[2] == 0 and X.city_eq[1] == 1
    assert X.name_cos[1] > X.name_cos[2] and X.addr_cos[1] > 0.5 > X.addr_cos[2]
    assert X.sx_n.tolist() == [1, 2, 2, 1] and X.s1_n.tolist() == [2, 2, 1, 1]
    assert X.sx_margin[1] > 0 > X.sx_margin[2] and X.s1_gap[1] == 0 and abs(X.s1_gap[0] - 0.1) < 1e-6
    assert X.indic_x.tolist() == [0, 0, 0, 1] and X.is_s3.tolist() == [0, 1, 1, 0]
    assert X.core_ratio[3] > 0.5, "P2 to_latin maps the Indic name before similarity"
    print("features self-check ok", X.shape)


if __name__ == "__main__":
    _check()
