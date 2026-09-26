"""P3 candidate generation: ONE sparse TF-IDF product per country, S2/S3 -> S1, then both directions derived from it.

    from block import candidates, fit_views, CFG
    vecs = fit_views(train_s2_s3_sample)            # train text only; hashed, so unseen tokens get the max IDF
    cands = candidates(s1_df, sx_df, CFG, vecs)     # DataFrame[s1, sx, block_score, block_rank_s1, block_rank_sx, via]
    python src/block.py                             # self-check on a toy world
    python src/block.py fit                         # (re)fit the hashed TF-IDF on 1M train S2/S3 -> data/work/p3/vecs_hash.pkl
    python src/block.py test France                 # full-scale candidates -> data/work/p3/cands_test_France.parquet
    python src/block.py train India                 # every train S1 of a country (the dense-holdout source, 35 min)

Text: P2 to_latin (learned Indic table) -> anyascii -> lowercase / strip punctuation; P1 state -> state group.
View "na" = L2([0.5 * name char_wb-3gram TF-IDF | 0.5 * address word TF-IDF]); cosine via sparse_dot_topn.
Search: every SX takes its top-10 S1 of the same state group (an SX with no state: top-30 of the whole country).
Keep: each SX's top-2 S1 (top-10 if its address is empty) + each S1's top-10 among the SX that listed it.
Full corpus (holdout S1 vs 10.3M train S2/S3, 2.2M S1 pool): pair recall 0.9916 at 14.2 candidates / S1.
Scale: country_pairs() streams SX chunks; split_cands() is the per-country full run (texts cached on disk, ids via
arrow): 35 min for all of test, peak 2.7 GB on India at shard=300k (lower CFG["shard"] to fit a smaller box).
Promoted from probes/P3-block/{block,full}.py (P3 evidence: probes/P3-block/RESULT.md).
"""
import json
import pickle
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anyascii import anyascii
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sparse_dot_topn import sp_matmul_topn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import normalize as N
import translit as T
from worlds import WORK

P3 = WORK / "p3"
VECS = P3 / "vecs_hash.pkl"

CFG = {
    "views": {},                                     # view -> real forward top-k (S1 -> SX); {} = derived only
    "rev": {"na": 10},                               # view -> reverse top-K (SX -> S1): the one sparse product
    "keep_r": 2,                                     # final: each SX's top-2 S1 ...
    "keep_f": 10,                                    # ... + each S1's top-10 among the SX that listed it
    "k_noaddr": 10,                                  # SX with an empty address (name-only, chains) keep top-10 S1
    "k_nostate": 30,                                 # search depth for SX without a state (feeds the per-S1 top-10)
    "w_name": 0.5,                                   # name weight inside the "na" view (full-density optimum)
    "min_score": 0.0,
    "translit": "p2",                                # "p2" (P2 learned table, then anyascii) | "anyascii" | "none"
    "norm": "basic",                                 # "basic" (lowercase + strip) | "p1" (P1 name_core / addr_norm)
    "partition": True,                               # search only S1/SX of the same state group (+ no-state SX)
    "fold": False,                                   # P2 phonetic fold of the name (both sides) before n-grams
    "hash": True,                                    # hashed features: unseen tokens get max IDF (France)
    "shard": 300_000,                                # SX rows per chunk (RAM knob: ~2.5 GB at 600k for India)
    "qchunk": 200_000,                               # queries per sp_matmul_topn call
    "threads": 15,
}

_NULL = re.compile(r"(?i)<?\bnull\b>?")
_WEB = re.compile(r"(?i)\bwww\.|\.(com|net|org|co\.in|in|co|biz|info|fr|us)\b|https?://")
# ASCII-only class is safe only AFTER anyascii; without it use the explicit punctuation class.
_NONALNUM = re.compile(r"[^a-z0-9]+")
_PUNCT = r"[\s,.;:'\"()\[\]{}\-&/|#!?*+_@%$~`^<>=]+"


def _latin(s: pd.Series, how: str) -> pd.Series:
    if how == "none":
        return s
    if how == "p2":
        s = T.to_latin(s)
    s = s.astype(object)
    m = ~s.map(str.isascii)
    s[m] = s[m].map(anyascii)
    return pd.Series(s, dtype="string")


def clean(s: pd.Series, how: str, is_addr: bool = False) -> pd.Series:
    s = s.fillna("").astype("string")
    if is_addr:
        s = s.str.replace(_NULL.pattern, " ", regex=True)
    else:
        s = s.str.replace(_WEB.pattern, " ", regex=True)
    s = _latin(s, how).str.lower()
    s = s.str.replace(_NONALNUM.pattern if how != "none" else _PUNCT, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


# State codes the generator swaps on true pairs (train world): Telangana <-> Andhra Pradesh (3.2k pairs),
# "Washington, DC" read as WA. Everything else agrees on 99.94% of true pairs whose SX has a state.
STATE_GROUP = {"tg": "ap", "dc": "wa"}


def texts(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """name / addr strings the vectorizers see, plus st = state group ('' = none found)."""
    how = cfg["translit"]
    name, addr = df.business_name.fillna("").astype("string"), df.business_address.fillna("").astype("string")
    if how == "p2":
        name, addr = _latin(name, "p2"), _latin(addr, "p2")
        how = "anyascii"
    out = {}
    if cfg.get("norm") == "p1":
        f = N.address_fields(addr, df.country)
        out["name"] = N.name_core(name, df.country).to_numpy()
        out["addr"] = f.addr_norm.str.replace(",", " ", regex=False).to_numpy()
        st = f.state.to_numpy()
    else:
        out["name"] = clean(name, how).to_numpy()
        out["addr"] = clean(addr, how, True).to_numpy()
        st = N.state(addr, df.country).to_numpy() if cfg.get("partition") else np.full(len(df), "")
    if cfg.get("fold"):
        out["name"] = T.fold(pd.Series(out["name"], dtype="string")).to_numpy()
    out["st"] = pd.Series(st, dtype=object).map(lambda x: STATE_GROUP.get(x, x)).to_numpy() \
        if cfg.get("partition") else np.full(len(df), "*", dtype=object)
    return pd.DataFrame({k: pd.Series(v, dtype="string") for k, v in out.items()})


def fit_views(train_sx: pd.DataFrame, cfg: dict = CFG, sample: int = 1_000_000, seed: int = 0, min_df: int = 2) -> dict:
    """Fit the name / addr vectorizers on (a sample of) train S2/S3 text. Returns {view: vectorizer}.
    cfg["hash"]: hashed features + IDF from train, so a token never seen in train (French street names: a third
    of France's address tokens) gets the maximum IDF instead of being dropped by a fixed vocabulary."""
    t = texts(train_sx.sample(n=min(sample, len(train_sx)), random_state=seed), cfg)
    an = {"name": dict(analyzer="char_wb", ngram_range=(3, 3)), "addr": dict(analyzer="word", token_pattern=r"\S+")}
    if not cfg.get("hash"):
        kw = dict(dtype=np.float32, sublinear_tf=True, min_df=min_df)
        return {v: TfidfVectorizer(**an[v], **kw).fit(t[v]) for v in an}
    nf = {"name": 2 ** 18, "addr": 2 ** 20}
    return {v: make_pipeline(HashingVectorizer(**an[v], n_features=nf[v], alternate_sign=False, norm=None,
                                               dtype=np.float32),
                             TfidfTransformer(sublinear_tf=True)).fit(t[v]) for v in an}


def _l2(X: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()).astype(np.float32)
    n[n == 0] = 1
    return sp.csr_matrix(sp.diags(1 / n) @ X, dtype=np.float32)


def mats(t: pd.DataFrame, views, vecs: dict, cfg: dict) -> dict:
    """{view: CSR float32} for the requested views; name/addr are transformed once and shared with "na"."""
    base = {v: vecs[v].transform(t[v]).astype(np.float32).tocsr() for v in ("name", "addr")
            if v in views or "na" in views}
    w = cfg["w_name"]
    out = {}
    if "na" in views:
        out["na"] = _l2(sp.hstack([base["name"] * w, base["addr"] * (1 - w)], format="csr"))
    out.update({v: base[v] for v in views if v in base})
    return out


def topk(Q: sp.csr_matrix, CTs, k: int, cfg: dict):
    """Top-k of Q @ C.T per Q row. C comes as (cols, C.T in CSR) pieces, cols = global id of each C row (an int
    offset also works), so a big side never has to sit in RAM whole.
    Returns (qi, ci, score) int32/int32/float32, sorted by qi then score desc."""
    parts = []
    for cols, CT in CTs:
        if CT.shape[1] == 0 or Q.shape[0] == 0:
            continue
        for q0 in range(0, Q.shape[0], cfg["qchunk"]):
            R = sp_matmul_topn(Q[q0:q0 + cfg["qchunk"]], CT, top_n=k, threshold=cfg["min_score"] or None,
                               n_threads=cfg["threads"]).tocoo()
            ci = R.col.astype(np.int32) + cols if np.isscalar(cols) else np.asarray(cols, np.int32)[R.col]
            parts.append((R.row.astype(np.int32) + q0, ci, R.data.astype(np.float32)))
        parts = [_best(*map(np.concatenate, zip(*parts)), k)]   # fold pieces: <= k per query, sorted
    return parts[0] if parts else (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.float32),)


def _best(qi, ci, sc, k):
    o = np.lexsort((-sc, qi))
    qi, ci, sc = qi[o], ci[o], sc[o]
    keep = ranks(qi) < k
    return qi[keep], ci[keep], sc[keep]


def ranks(qi: np.ndarray) -> np.ndarray:
    """0-based rank within each run of equal qi (input already sorted by qi, score desc)."""
    start = np.r_[0, np.flatnonzero(np.diff(qi)) + 1] if len(qi) else np.zeros(0, int)
    return (np.arange(len(qi)) - np.repeat(start, np.diff(np.r_[start, len(qi)]))).astype(np.int32)


def _groups(st: pd.Series) -> dict:
    """{state group: row numbers}; '' = no state found."""
    return {g: np.asarray(ix, np.int32) for g, ix in pd.Series(st.to_numpy()).groupby(st.to_numpy()).indices.items()}


def index(ta: pd.DataFrame, vecs: dict, cfg: dict, fwd_rows=None) -> dict:
    """S1-pool side for ONE country (ta = texts of the S1 pool): per-state-group transposed matrices for the
    reverse search, per-group query matrices for the (optional) real forward search."""
    fk = {v: k for v, k in cfg["views"].items() if k}
    rk = {v: k for v, k in cfg.get("rev", {}).items() if k}
    A = mats(ta, set(fk) | set(rk), vecs, cfg)
    ga = _groups(ta.st)
    loose = ga.pop("", np.zeros(0, np.int32))                       # S1 with no state: in every group
    fr = np.arange(len(ta), dtype=np.int32) if fwd_rows is None else np.asarray(fwd_rows, np.int32)
    fq = {g: np.intersect1d(r, fr).astype(np.int32) for g, r in ga.items()}   # forward queries per group
    fq[""] = np.intersect1d(loose, fr).astype(np.int32)                        # ... loose S1 search all SX
    ga = {g: np.union1d(r, loose).astype(np.int32) for g, r in ga.items()}
    AT = {v: {g: A[v][r].T.tocsr() for g, r in ga.items()} for v in rk}
    for v in rk:
        AT[v][""] = A[v].T.tocsr() if len(ga) != 1 else next(iter(AT[v].values()))
    ga[""] = np.arange(len(ta), dtype=np.int32)
    return {"fk": fk, "rk": rk, "ga": ga, "AT": AT, "fq": fq, "Af": {v: {g: A[v][q] for g, q in fq.items()} for v in fk}}


VIEWS, DIRS = ["na", "name", "addr"], ["f", "r"]


def _hits(q1, qx, sc, rk, view, dirn) -> pd.DataFrame:
    """Compact hit rows (fixed categories so concatenation keeps them categorical)."""
    n = len(q1)
    return pd.DataFrame({"q1": q1, "qx": qx, "score": sc, "rank": rk,
                         "view": pd.Categorical.from_codes(np.full(n, VIEWS.index(view), np.int8), VIEWS),
                         "dir": pd.Categorical.from_codes(np.full(n, DIRS.index(dirn), np.int8), DIRS)})


def search(ix: dict, tb: pd.DataFrame, vecs: dict, cfg: dict, keep_rev=None) -> pd.DataFrame:
    """One SX shard (tb = its texts) against the S1 index. Reverse: top-k S1 per SX. Forward: shard-local top-k SX
    per S1 (fold across shards with `fold`). With cfg["partition"], an SX in state group g searches only the S1 of g;
    an SX without a state, or whose group has no S1, searches the whole pool; the forward search mirrors it.
    keep_rev: bool per S1 row; reverse hits on other S1 are dropped at once (evaluating a subset of a full pool).
    Returns hits: q1 (S1 row), qx (row in tb), score, rank, view, dir ("f" | "r")."""
    fk, rk, ga, fq = ix["fk"], ix["rk"], ix["ga"], ix["fq"]
    Bm = mats(tb, set(fk) | set(rk), vecs, cfg)
    gb = _groups(tb.st)
    nost = gb.get("", np.zeros(0, np.int32))
    out = []
    for v, k in fk.items():
        got = []
        for g, q in fq.items():
            cols = (np.union1d(gb.get(g, np.zeros(0, np.int32)), nost).astype(np.int32) if g
                    else np.arange(len(tb), dtype=np.int32))
            qi, ci, sc = topk(ix["Af"][v][g], [(cols, Bm[v][cols].T.tocsr())], k, cfg)
            got.append((q[qi], ci, sc))
        qi, ci, sc = _best(*map(np.concatenate, zip(*got)), k)
        out.append(_hits(qi, ci, sc, ranks(qi), v, "f"))
    for v, k in rk.items():
        for g, rows in gb.items():
            g = g if g in ga else ""
            kg = max(k, cfg.get("k_noaddr", 0), cfg.get("k_nostate", 0)) if g == "" else k   # address-less SX are in ""
            qi, ci, sc = topk(Bm[v][rows], [(ga[g], ix["AT"][v][g])], kg, cfg)
            m = slice(None) if keep_rev is None else keep_rev[ci]
            out.append(_hits(ci[m], rows[qi[m]], sc[m], ranks(qi)[m], v, "r"))
    return pd.concat(out, ignore_index=True)


def fold(h: pd.DataFrame, ks: dict, key: str = "q1") -> pd.DataFrame:
    """Keep the top ks[view] rows per (key, view) by score and re-rank them (0 = best)."""
    h = h.sort_values([key, "score"], ascending=[True, False], kind="stable")
    h = h.assign(rank=h.groupby([key, "view"], sort=False).cumcount().astype(np.int32))
    return h[h["rank"].to_numpy() < h.view.astype(str).map(ks).to_numpy()]


def derive(r: pd.DataFrame, kr, kf: int, key: str = "q1") -> pd.DataFrame:
    """Final candidate rule from reverse hits only: each SX's top-kr S1 ('r'; kr may be an array aligned with r, e.g.
    larger for address-less SX), plus each S1's top-kf SX among the SX that listed it ('f' - equals a real forward
    top-kf on dev: 0.9979 vs 0.9979 recall)."""
    a = r[r["rank"].to_numpy() < kr].assign(via="r")
    b = fold(r, {v: kf for v in VIEWS}, key).assign(via="f")
    return pd.concat([a, b], ignore_index=True)


def run_views(a: pd.DataFrame, b: pd.DataFrame, vecs: dict, cfg: dict, fwd_rows=None, keep_rev=None,
              log=lambda *_: None) -> pd.DataFrame:
    """All forward + reverse hits for ONE country (raw frames a = S1 pool, b = SX; b streamed in cfg["shard"])."""
    ix = index(texts(a, cfg), vecs, cfg, fwd_rows)
    out = []
    for off in range(0, len(b), cfg["shard"]):
        t0 = time.time()
        h = search(ix, texts(b.iloc[off:off + cfg["shard"]], cfg), vecs, cfg, keep_rev)
        h["qx"] += off
        out.append(h)
        log(f"    shard {off:,} in {time.time() - t0:.0f}s")
    h = pd.concat(out, ignore_index=True)
    f = h["dir"] == "f"
    return pd.concat([h[~f], fold(h[f], ix["fk"])], ignore_index=True)


def country_pairs(ta: pd.DataFrame, chunks, vecs: dict, cfg: dict = CFG, log=lambda *_: None) -> pd.DataFrame:
    """The final candidate rule for ONE country, streamed. ta = texts of the S1 pool; chunks yields (row offset,
    texts of an SX chunk) in order. Keeps each SX's top keep_r S1 (k_noaddr for an SX with an empty address) and each
    S1's top keep_f among all SX that listed it. numpy only: RAM ~ S1 index + one chunk + (#S1 x keep_f) pool.
    Returns int pairs: q1 (S1 row), qx (global SX row), block_score, block_rank_s1, block_rank_sx, via (r | f | r+f)."""
    ix = index(ta, vecs, cfg)
    del ta                                          # frees the pool texts if the caller passed them inline
    kr0, kna = cfg["keep_r"], max(cfg["keep_r"], cfg.get("k_noaddr", 0))
    kept, pool = [], {}
    for off, tb in chunks:
        t0 = time.time()
        h = search(ix, tb, vecs, cfg)
        q1, qxl = h.q1.to_numpy(np.int64), h.qx.to_numpy(np.int64)
        sc, rk, view = h.score.to_numpy(np.float32), h["rank"].to_numpy(), h.view.cat.codes.to_numpy()
        rev = (h["dir"] == "r").to_numpy(bool)
        del h
        m = rev & (rk < np.where((tb.addr == "").to_numpy(bool)[qxl], kna, kr0))
        kept.append((q1[m] << 32) | (qxl[m] + off))
        kept.append(sc[m])
        for v in np.unique(view):                   # per-S1 top keep_f of every view, folded across chunks
            mv = view == v
            k = max(cfg["keep_f"], ix["fk"].get(VIEWS[v], 0))
            new = (q1[mv], qxl[mv] + off, sc[mv])
            pool[v] = _best(*map(np.concatenate, zip(pool[v], new)), k) if v in pool else _best(*new, k)
        log(f"    chunk {off:,}+{len(tb):,} in {time.time() - t0:.0f}s")
    del ix
    return finalize(kept[0::2], kept[1::2], list(pool.values()))


def finalize(rkeys: list, rscores: list, pools: list) -> pd.DataFrame:
    """Union of reverse picks (int64 keys q1 << 32 | qx, with scores) and per-S1 pools ((q1, qx, score) arrays) ->
    one row per pair: q1, qx, block_score (max), block_rank_s1, block_rank_sx, via (r | f | r+f)."""
    pk = [(np.asarray(p[0], np.int64) << 32) | np.asarray(p[1], np.int64) for p in pools]
    key = np.concatenate(rkeys + pk)
    sc = np.concatenate(rscores + [p[2] for p in pools]).astype(np.float32)
    isr = np.concatenate([np.ones(sum(map(len, rkeys)), np.int8), np.zeros(sum(map(len, pk)), np.int8)])
    del pk
    o = np.argsort(key, kind="stable")
    key, sc, isr = key[o], sc[o], isr[o]
    idx = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, int)
    uk = key[idx]
    g = pd.DataFrame({"q1": (uk >> 32).astype(np.int32), "qx": (uk & 0xFFFFFFFF).astype(np.int64),
                      "block_score": np.maximum.reduceat(sc, idx) if len(idx) else sc[:0]})
    r = np.maximum.reduceat(isr, idx) == 1 if len(idx) else np.zeros(0, bool)
    f = np.minimum.reduceat(isr, idx) == 0 if len(idx) else np.zeros(0, bool)
    g["via"] = pd.Categorical.from_codes(np.select([r & f, r], [2, 0], 1).astype(np.int8), ["r", "f", "r+f"])
    for col, by in (("block_rank_s1", "q1"), ("block_rank_sx", "qx")):
        o = np.lexsort((-g.block_score.to_numpy(), g[by].to_numpy()))
        rr = np.empty(len(g), np.int32)
        rr[o] = ranks(g[by].to_numpy()[o])
        g[col] = rr
    return g[["q1", "qx", "block_score", "block_rank_s1", "block_rank_sx", "via"]]


def candidates(s1_df: pd.DataFrame, sx_df: pd.DataFrame, cfg: dict = CFG, vecs: dict | None = None,
               log=print) -> pd.DataFrame:
    """Candidate pairs within country (see country_pairs). `vecs` from fit_views(train S2/S3); fitted on sx_df if None
    (toy use only). Returns DataFrame[s1, sx, block_score, block_rank_s1, block_rank_sx, via]; block_rank_s1 /
    block_rank_sx = rank by block_score among the final pairs of that S1 / SX (0 = best)."""
    vecs = vecs or fit_views(sx_df, cfg)
    cols = ["s1", "sx", "block_score", "block_rank_s1", "block_rank_sx", "via"]
    out = []
    for c in sorted(set(s1_df.country) & set(sx_df.country)):
        t0 = time.time()
        a, b = s1_df[s1_df.country == c], sx_df[sx_df.country == c]
        chunks = ((off, texts(b.iloc[off:off + cfg["shard"]], cfg)) for off in range(0, len(b), cfg["shard"]))
        f = country_pairs(texts(a, cfg), chunks, vecs, cfg)
        f["s1"], f["sx"] = a.entity_id.to_numpy()[f.q1], b.entity_id.to_numpy()[f.qx]
        out.append(f[cols])
        log(f"{c}: {len(a):,} S1 x {len(b):,} SX -> {len(f):,} pairs ({len(f) / max(len(a), 1):.1f}/S1) "
            f"in {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def _check():
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "US", "France"],
                       "business_name": ["Twisted Deli", "Metro Holding Co", "Bien & Frères SAS"],
                       "business_address": ["4 Dix Road, Woburn, MA", "3860 Pin Oak Road, Raleigh, NC",
                                            "23 Boulevard de la Renaissance, Saint-Nazaire"]})
    sx = pd.DataFrame({"entity_id": ["S2-1", "S3-1", "S2-2", "S3-2", "S2-3", "S2-4"],
                       "country": ["US", "US", "US", "US", "France", "India"],
                       "business_name": ["twisteddeli.com", "TWISTED DELI CENTER", "Metro Htdning Company",
                                         "@nothing", "BIEN ET FRERES", "Twisted Deli"],
                       "business_address": ["##4 Dix Rd, <NULL>, Woburn", "4 DIX ROAD, WOBURN, MA",
                                            "3860 PIN OAK RD, RALEIGH, NC", "", "23 BD DE LA RENAISSANCE", "x"]})
    fit = pd.concat([s1, sx])
    cfg = {**CFG, "views": {"na": 2, "name": 1, "addr": 1}, "rev": {"na": 1}, "partition": False, "hash": False}
    c = candidates(s1, sx, cfg, fit_views(fit, cfg, min_df=1), log=lambda *_: None)
    got = set(zip(c.s1, c.sx))
    assert {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-3", "S2-3")} <= got, got
    assert ("S1-1", "S2-4") not in got, "never block across countries"
    cp = candidates(s1, sx, {**cfg, "partition": True}, fit_views(fit, cfg, min_df=1), log=lambda *_: None)
    assert set(zip(cp.s1, cp.sx)) >= {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2")}, "state partition"
    cd = candidates(s1, sx, CFG, fit_views(fit, CFG), log=lambda *_: None)     # the recommended config
    assert set(zip(cd.s1, cd.sx)) >= {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-3", "S2-3")}, cd
    assert ("S1-1", "S2-4") not in set(zip(cd.s1, cd.sx))
    r = pd.DataFrame({"q1": [0, 1, 0, 1, 2], "qx": [0, 0, 1, 1, 2], "score": [.9, .8, .3, .7, .5], "rank": [0, 1, 1, 0, 0],
                      "view": "na", "dir": "r"})
    d = derive(r, kr=1, kf=1)
    assert set(zip(d.q1, d.qx, d.via)) == {(0, 0, "r"), (1, 1, "r"), (2, 2, "r"), (0, 0, "f"), (1, 0, "f"), (2, 2, "f")}
    assert c.groupby("s1").block_rank_s1.min().eq(0).all() and c.groupby("sx").block_rank_sx.min().eq(0).all()
    assert not c.duplicated(["s1", "sx"]).any() and set(c.via.astype(str)) <= {"r", "f", "r+f"}
    assert clean(pd.Series(["Karrah Érwin, P.Á.", "सनराइज"]), "anyascii").str.isascii().all()
    print("block self-check ok")


# ---------------------------------------------------------------- full scale (from probes/P3-block/full.py)
COLS = ["entity_id", "business_name", "business_address", "country"]
TXT = {"train": P3 / "txt", "test": P3 / "txt_test"}      # cached texts, P3's layout
CHUNK = 500_000                                      # SX rows per cached text file


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _batches(name: str, country: str | None = None, frac: float = 1.0):
    """Stream data/work/<name>.parquet in 250k-row batches (optionally one country, optionally sampled)."""
    import pyarrow.parquet as pq
    for i, rb in enumerate(pq.ParquetFile(WORK / f"{name}.parquet").iter_batches(250_000, columns=COLS)):
        d = rb.to_pandas()
        if country is not None:
            d = d[d.country == country]
        yield d.sample(frac=frac, random_state=i) if frac < 1 else d


def vecs(n: int = 1_000_000) -> dict:
    """The train-fitted hashed TF-IDF (name char-3 + address words). Fitted on a seeded 1M-row sample of train
    S2+S3 when data/work/p3/vecs_hash.pkl is absent (36 s, 1.9 GB) -- train text only, never test."""
    if not VECS.exists():
        parts = [d for i in (2, 3) for d in _batches(f"train_source{i}", frac=n / 2 / 5_000_000)]
        v = fit_views(pd.concat(parts, ignore_index=True), CFG, sample=n)
        VECS.parent.mkdir(parents=True, exist_ok=True)
        VECS.write_bytes(pickle.dumps(v))
    return pickle.loads(VECS.read_bytes())


def idf_country(vec: dict, sx_text: pd.DataFrame, share: float, seed: int = 0) -> dict:
    """P14's France fix (#9a): the train IDF refit with a seeded sample of an unseen country's own S2/S3 texts added
    to the IDF pool at `share` (0.144 = France's share of test S2/S3). The train df is recovered from idf_ (smooth
    idf: idf = ln((1 + n) / (1 + df)) + 1); idf_ = ln((1 + n + m) / (1 + df + df_c)) + 1. vec is changed in place."""
    for view in ("name", "addr"):
        hv, tf = vec[view].steps[0][1], vec[view].steps[1][1]
        idf = tf.idf_.astype(np.float64)
        n = round(np.exp(idf.max() - 1) - 1)                       # max idf = ln(1 + n) + 1 (df = 0)
        df = np.rint((1 + n) / np.exp(idf - 1) - 1)
        m = round(n * share / (1 - share))
        X = hv.transform(sx_text[view].sample(n=min(m, len(sx_text)), random_state=seed)).tocsc()
        dfc = np.diff(X.indptr)                                    # docs containing each hashed feature
        tf.idf_ = (np.log((1 + n + X.shape[0]) / (1 + df + dfc)) + 1).astype(tf.idf_.dtype)
    return vec


def idf_domain(vec: dict) -> dict:
    """P14's train-only fallback (#9d): the address IDF of the French domain words already listed in normalize.py
    (street types, articles, régions + départements, house-number words) lowered to the median train IDF of the US
    street types. City names stay at max IDF: no list of them exists without external data. In place."""
    hv, tf = vec["addr"].steps[0][1], vec["addr"].steps[1][1]
    fr = {k: w for k, w in N.STREET.items() if k in {"rue", "bd", "impasse", "allee", "chemin", "route", "faubourg",
                                                     "quai", "cours", "sainte", "avenue", "av", "boulevard", "place",
                                                     "saint"}}
    words = set(fr) | set(fr.values()) | {"de", "des", "du", "la", "le", "les", "l", "d", "et", "en", "sur", "sous",
                                          "aux", "au", "ndeg", "no", "bis", "ter", "france"}
    for reg, deps in N.FR_REGIONS.items():
        words |= {w for x in [reg, *deps] for w in x.split()}
    ix = lambda ws: hv.transform([" ".join(ws)]).indices
    val = float(np.median(tf.idf_[ix(["st", "rd", "dr", "ave", "ln", "ct", "blvd"])]))
    idf = tf.idf_.copy()
    j = ix(sorted(words))
    idf[j] = np.minimum(idf[j], val)
    tf.idf_ = idf
    return vec


def _txt(d: pd.DataFrame) -> pd.DataFrame:
    t = texts(d, CFG)
    t.insert(0, "entity_id", d.entity_id.to_numpy())
    return t


def prep(split: str, country: str):
    """Texts (P2 translit, cleaning, state group) of one country: <country>_s1 + <country>_sx_NNN chunks."""
    txt = TXT[split]
    txt.mkdir(parents=True, exist_ok=True)
    p = txt / f"{country}_s1.parquet"
    if not p.exists():
        _txt(pd.concat(_batches(f"{split}_source1", country), ignore_index=True)).to_parquet(p, index=False)
    buf, n = [], 0
    for src in (f"{split}_source2", f"{split}_source3"):
        for d in _batches(src, country):
            buf.append(d)
            if sum(map(len, buf)) >= CHUNK:
                _write_txt(buf, txt / f"{country}_sx_{n:03d}.parquet")
                buf, n = [], n + 1
    if buf:
        _write_txt(buf, txt / f"{country}_sx_{n:03d}.parquet")


def _write_txt(buf, p):
    if not p.exists():
        _txt(pd.concat(buf, ignore_index=True)).to_parquet(p, index=False)


def split_cands(split: str, country: str, log=print) -> Path:
    """CFG on one country of a split (every S1 of it, every S2/S3) -> data/work/p3/cands_<split>_<country>.parquet
    (s1, sx ids + block_score / ranks / via), with train-fitted vectorizers. Ids are mapped with arrow (no Python strings), S1 texts freed once
    indexed. A country with no S2/S3 writes an empty file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    prep(split, country)
    v, txt = vecs(), TXT[split]
    files = sorted(txt.glob(f"{country}_sx_*.parquet"))
    offs = np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in files])
    n1 = pq.ParquetFile(txt / f"{country}_s1.parquet").metadata.num_rows
    sh = CFG["shard"]

    def chunks():
        for o, p in zip(offs, files):
            tb = pd.read_parquet(p, columns=["name", "addr", "st"])
            for i in range(0, len(tb), sh):
                yield o + i, tb.iloc[i:i + sh].reset_index(drop=True)

    t0 = time.time()
    f = country_pairs(pd.read_parquet(txt / f"{country}_s1.parquet", columns=["name", "addr", "st"]),
                      chunks(), v, CFG, log=lambda m: log(m, f"peak {peak_gb():.2f} GB")) if files else None
    out = P3 / f"cands_{split}_{country}.parquet"
    if f is None or len(f) == 0:
        pd.DataFrame({"s1": pd.Series([], dtype="string"), "sx": pd.Series([], dtype="string")}).to_parquet(out)
        return out
    ids1 = pq.read_table(txt / f"{country}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    idx = pa.concat_arrays([pq.read_table(p, columns=["entity_id"]).column(0).combine_chunks() for p in files])
    t = pa.table({"s1": ids1.take(pa.array(f.q1.to_numpy())), "sx": idx.take(pa.array(f.qx.to_numpy())),
                  "block_score": f.block_score.to_numpy(), "block_rank_s1": f.block_rank_s1.to_numpy(),
                  "block_rank_sx": f.block_rank_sx.to_numpy(), "via": pa.array(f.via.astype(str).to_numpy())})
    pq.write_table(t, out)
    meta = {"country": country, "n_s1": n1, "n_sx": int(offs[-1]), "pairs": len(f), "cands_per_s1": len(f) / n1,
            "s1_no_cands": 1 - f.q1.nunique() / n1, "secs": time.time() - t0, "peak_gb": peak_gb()}
    out.with_suffix(".json").write_text(json.dumps(meta))
    log(json.dumps(meta))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _check()
    elif sys.argv[1] == "fit":
        VECS.unlink(missing_ok=True)
        vecs()
    else:
        split_cands(sys.argv[1], sys.argv[2])
