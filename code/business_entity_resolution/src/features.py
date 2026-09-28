"""Pair features of the M2 matcher (P4-match): 65 label-free float32 columns per candidate pair. Record fields are
computed once per record (rec), pair features are vectorised over aligned row arrays: rapidfuzz cpdist for string
similarities, sparse row-wise products for TF-IDF cosines and token / place overlaps, numpy group stats for the
candidate-list context. No per-pair Python loop. Country is never a feature (France is unseen).

    r1, rx = rec(s1_df), rec(sx_df)          # raw rows -> record fields (P2 to_latin, P1 normalize, P3 texts)
    X = featurize(pairs, r1, rx)             # in memory: pairs = s1, sx ids (+ P3 block columns when blocked)
    build(split, country)                    # full scale: data/work/p4/X/<split>_<country>_sx_NNN.parquet
    .venv/bin/python code/business_entity_resolution/src/features.py    # self-check

The context features need the COMPLETE candidate list of every S1 and SX of a country: never featurize a sample.
Full scale follows P4's featrun.py: q1 = row of P3's text file <country>_s1, qx = row of the concatenated
<country>_sx_* files; S1 context over the whole candidate table, SX context per SX file (a file holds every pair
of its SX); pairs the rules accept but P3 missed are added with NaN context (blocked = 0). Every feature is
rounded to 3 decimals, as the booster was trained.
"""
import ctypes
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")   # plain malloc, so trim() can hand RAM back

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block as B
import normalize as N
import rules
from worlds import WORK

XDIR, CTX = WORK / "p4" / "X", WORK / "p4" / "ctx"
RF = ["id", "country", "core", "web", "is_web", "state", "city", "hn", "places"]     # rules._feats columns
BLOCK = ["block_score", "block_rank_s1", "block_rank_sx", "via"]
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
        _VECS = B.vecs()
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
    """float32 CSR rows, L2-normalised (block._l2)."""
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
    """Block features + candidate-list context over the given (complete) list of blocked pairs."""
    out = block_cols(pairs)
    s = np.nan_to_num(out["block_score"])
    return pd.DataFrame(_group_ctx(q1, s, "s1") | out | _group_ctx(qx, s, "sx"))


def _take(r: pd.DataFrame, col: str, ix: np.ndarray) -> list:
    """Values of column col at rows ix of r, as a list."""
    return r[col].iloc[ix].tolist()


def _fz(scorer, a, b, **kw):
    """A rapidfuzz scorer over the aligned pairs (a[k], b[k]), scaled to [0, 1] float32."""
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


def featurize(pairs: pd.DataFrame, r1: pd.DataFrame, rx: pd.DataFrame, i: np.ndarray, j: np.ndarray,
              rule_acc: np.ndarray, chunk: int = 250_000) -> pd.DataFrame:
    """In-memory X for the COMPLETE candidate list of a pool: pairs holds the P3 block columns (block_score NaN
    = a pair only the rules proposed), i / j = rows of r1 / rx, rule_acc = accepted by rules.match. Same values as
    build() writes: context over the blocked pairs only, NaN context for rule-only pairs, all rounded to 3."""
    blk = pairs.block_score.notna().to_numpy() if "block_score" in pairs else np.zeros(len(pairs), bool)
    C = context(pairs[blk], i[blk], j[blk]).astype(np.float32).round(3)
    X = pd.DataFrame(np.nan, index=pd.RangeIndex(len(pairs)), columns=C.columns, dtype=np.float32)
    X.loc[blk] = C.to_numpy()
    X.loc[~blk, "blocked"] = 0.0
    counts, m1, mx = pool_counts(r1), mats(r1), mats(rx)
    S = pd.concat([strings(r1, rx, i[a:a + chunk], j[a:a + chunk], counts, m1, mx)
                   for a in range(0, len(pairs), chunk)] or [pd.DataFrame()], ignore_index=True)
    S["rule_acc"] = np.asarray(rule_acc, np.float32)
    return pd.concat([X, S.round(3)], axis=1)


# ---------------------------------------------------------------- full scale (P4 featrun.py)
def sx_files(split, c):
    """(P3's SX text files of a split / country, sorted; their cumulative row offsets from 0): qx is a
    global row over these files."""
    files = sorted(B.TXT[split].glob(f"{c}_sx_*.parquet"))
    return files, np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in files])


def raw_sx(split, c):
    """(txt file, raw rows in its order): block.prep streamed source2 then source3 in 250k batches, country-filtered,
    and cut a file at whole batches -- the same stream re-cut at the file sizes gives the same rows."""
    files, offs = sx_files(split, c)
    sizes, buf, k = np.diff(offs), [], 0
    for src in ("source2", "source3"):
        for d in B._batches(f"{split}_{src}", c):
            buf.append(d)
            while k < len(files) and sum(map(len, buf)) >= sizes[k]:
                a = pd.concat(buf, ignore_index=True)
                ids = pq.read_table(files[k], columns=["entity_id"]).column(0).to_pandas()
                assert (a.entity_id.iloc[:sizes[k]].to_numpy() == ids.to_numpy()).all(), files[k]
                yield files[k], a.iloc[:sizes[k]].reset_index(drop=True)
                buf, k = [a.iloc[sizes[k]:]], k + 1
    assert k == len(files)


def pieces(path, sizes):
    """Consecutive row ranges of a parquet file as DataFrames (streamed; never the whole file in RAM)."""
    it, buf = pq.ParquetFile(path).iter_batches(100_000), None
    for n in sizes:
        while buf is None or buf.num_rows < n:
            b = pa.Table.from_batches([next(it)])
            buf = b if buf is None else pa.concat_tables([buf, b])
        yield buf.slice(0, n).to_pandas()
        buf = buf.slice(n)


def trim():
    """Hand freed heap back to the OS (RSS stays at the transient peak otherwise)."""
    pa.default_memory_pool().release_unused()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def ids(split, c) -> tuple:
    """(S1 ids, SX ids) as arrow arrays in q1 / qx row order."""
    files, _ = sx_files(split, c)
    s1 = pq.read_table(B.TXT[split] / f"{c}_s1.parquet", columns=["entity_id"]).column(0).combine_chunks()
    sx = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files]) \
        if files else pa.array([], pa.string())
    return s1, sx


def s1_table(split, c):
    """Record table of the whole S1 pool of a country (P3 text order) + its rule features (run.feats cache)."""
    t = pd.read_parquet(B.TXT[split] / f"{c}_s1.parquet")
    rf = pd.read_parquet(WORK / f"rf_{split}_s1_{c}.parquet")
    rf = rf.set_index("id").reindex(t.entity_id.rename("id")).reset_index()
    assert rf.country.notna().all(), "rule features for every S1"
    return assemble(rf[RF], t, pd.Series("", index=t.index))    # S1 is always Latin: indic only matters on SX


def cands(split, c) -> pd.DataFrame:
    """P3 candidate pairs of a whole country as int rows (q1, qx) + block columns: data/work/p3/cands_<split>_<c>
    (block.split_cands, rebuilt when absent). Train reuses P4's identical export cands_train_full_<c> if present."""
    p4 = WORK / "p4" / f"cands_train_full_{c}.parquet"
    if split == "train" and p4.exists():
        return pd.read_parquet(p4, columns=["q1", "qx", *BLOCK])
    t = pq.read_table(B.split_cands(split, c) if not (B.P3 / f"cands_{split}_{c}.parquet").exists()
                      else B.P3 / f"cands_{split}_{c}.parquet")
    id1, idx = ids(split, c)
    q1, qx = pc.index_in(t.column("s1"), value_set=id1), pc.index_in(t.column("sx"), value_set=idx)
    assert q1.null_count == 0 and qx.null_count == 0, "every candidate id is a record of the split"
    return t.select(BLOCK).to_pandas().assign(q1=q1.to_numpy().astype(np.int32), qx=qx.to_numpy().astype(np.int32))


def _key(a, b):
    """int64 pair key a << 32 | b."""
    return (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)


def xfiles(split, c) -> list:
    """The X feature files of a split / country, one per SX text file."""
    return [XDIR / f"{split}_{f.stem}.parquet" for f in sx_files(split, c)[0]]


def ctx_step(split, c):
    """S1-side context over the whole candidate table (an S1's pairs span every SX file), rows re-cut by SX file
    -> data/work/p4/ctx/. The SX side is computed per file in feat_step: a file holds every pair of its SX."""
    CTX.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = cands(split, c)
    d["via"] = d.via.astype("category")
    s1c = _group_ctx(d.q1.to_numpy(), np.nan_to_num(d.block_score.to_numpy(np.float32)), "s1")
    o = np.argsort(d.qx.to_numpy(), kind="stable")
    d = d.iloc[o].reset_index(drop=True)
    for k, v in s1c.items():
        d[k] = v[o]
    del s1c, o
    files, offs = sx_files(split, c)
    cut = np.searchsorted(d.qx.to_numpy(), offs)
    for k, f in enumerate(files):
        d.iloc[cut[k]:cut[k + 1]].to_parquet(CTX / f"{split}_{f.stem}.parquet", index=False)
    print(f"ctx {split} {c}: {len(d):,} pairs in {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)


def feat_step(split, c, only=None, out=XDIR, chunk=250_000):
    """All features per SX file -> out/<split>_<c>_sx_NNN.parquet: q1, qx, [is_true, fold on train], 65 features.
    Needs ctx_step's files and the rules.feats caches rf_<split>_{s1,sx}_<c>. only = file stems to (re)build."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fold1 = tk = None
    if split == "train":      # truth as sorted int keys (q1 << 32 | qx), ids mapped in arrow; first, while RAM is empty
        ids1, idx = ids(split, c)
        gt = pq.read_table(WORK / "gt_pairs.parquet")
        a = pc.index_in(gt.column("s1"), value_set=ids1).to_numpy(zero_copy_only=False)
        b = pc.index_in(gt.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
        ok = ~np.isnan(a)
        tk = np.sort(_key(a[ok].astype(np.int64), b[ok].astype(np.int64)))
        fo = pq.read_table(WORK / "s1_folds.parquet")
        fold1 = fo.column("fold").to_numpy()[pc.index_in(ids1, value_set=fo.column("entity_id").combine_chunks())
                                             .to_numpy()].astype(np.int8)
        del gt, fo, idx, a, b, ids1
        trim()
    r1 = s1_table(split, c)
    trim()
    m1, counts = mats(r1), pool_counts(r1)
    ix1 = pd.Index(r1.id)
    trim()
    print(f"feat {split} {c}: S1 pool {len(r1):,} ready in {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)
    files, offs = sx_files(split, c)
    rfs = pieces(WORK / f"rf_{split}_sx_{c}.parquet", np.diff(offs))   # rules.feats cache, in P3 text order
    for k, (f, raw) in enumerate(raw_sx(split, c)):
        rf = next(rfs)
        p = out / f"{split}_{f.stem}.parquet"
        if p.exists() or (only is not None and f.stem not in only):
            continue
        t0 = time.time()
        off = offs[k]
        assert (rf.id.to_numpy() == raw.entity_id.to_numpy()).all()
        rx = assemble(rf[RF], pd.read_parquet(f), raw.business_name)
        mx = mats(rx)
        acc, _ = rules.match(r1[RF], rf[RF], chunk=100_000, by_state=True, keep_cands=False)   # chunk: France chains
        ak = _key(ix1.get_indexer(acc.s1), pd.Index(rx.id).get_indexer(acc.sx) + off)
        X = pd.read_parquet(CTX / f"{split}_{f.stem}.parquet")
        b = block_cols(X)
        X = pd.concat([X.drop(columns=[k for k in b if k in X] + ["via"]), pd.DataFrame(
            b | _group_ctx(X.qx.to_numpy(), np.nan_to_num(b["block_score"]), "sx"))], axis=1)
        extra = np.setdiff1d(ak, _key(X.q1, X.qx))
        if len(extra):
            X = pd.concat([X, pd.DataFrame({"q1": (extra >> 32).astype(np.int32),
                                            "qx": (extra & 0xFFFFFFFF).astype(np.int32), "blocked": np.float32(0)})],
                          ignore_index=True)
        if split == "train":
            X["fold"] = fold1[X.q1.to_numpy()]
            kk = _key(X.q1, X.qx)
            X["is_true"] = tk[np.minimum(np.searchsorted(tk, kk), len(tk) - 1)] == kk
        i, j = X.q1.to_numpy(), X.qx.to_numpy() - off
        acc_pair = np.isin(_key(X.q1, X.qx), ak)
        id_ = [col for col in ("q1", "qx", "is_true", "fold") if col in X]
        fc = [col for col in X.columns if col not in id_]
        w = None
        for a in range(0, len(X), chunk):              # one chunk at a time straight to the writer (RAM)
            S = strings(r1, rx, i[a:a + chunk], j[a:a + chunk], counts, m1, mx)
            S["rule_acc"] = acc_pair[a:a + chunk].astype(np.float32)
            part = X.iloc[a:a + chunk].reset_index(drop=True)
            t = pa.Table.from_pandas(pd.concat([part[id_], part[fc].astype(np.float32).round(3), S.round(3)], axis=1),
                                     preserve_index=False)
            w = w or pq.ParquetWriter(p.with_suffix(".tmp"), t.schema, compression="zstd")
            w.write_table(t)
            del S, part, t
        w.close()
        p.with_suffix(".tmp").rename(p)
        print(f"  {f.stem}: {len(raw):,} SX, {len(X):,} pairs (+{len(extra):,} rule-only) in {time.time() - t0:.0f}s, "
              f"peak {B.peak_gb():.2f} GB", flush=True)
        del X, rx, mx, rf, raw
        trim()


def build(split, c):
    """Every X file of one split / country that is missing (resumable per SX file). The caller (run.py) makes sure
    the P3 texts / candidates (block.split_cands) and the rules.feats caches exist."""
    if all(p.exists() for p in xfiles(split, c)):
        return
    ctx_step(split, c)
    feat_step(split, c)
    for p in CTX.glob(f"{split}_{c}_sx_*.parquet"):
        p.unlink()


def _check():
    """Self-check on a toy pool: 65 float32 columns, NaN context for rule-only pairs, core / web / hn / city
    equality, cosines, context counts and the Indic transliteration."""
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "US", "India"],
                       "business_name": ["Twisted Deli LLC", "Metro Holding Co", "Sunrise Traders Pvt Ltd"],
                       "business_address": ["4 Dix Road, Woburn, MA", "3860 Pin Oak Road, Raleigh, NC",
                                            "12 MG Road, Pune, Maharashtra"]})
    sx = pd.DataFrame({"entity_id": ["S2-1", "S3-1", "S2-2", "S3-2"], "country": ["US", "US", "India", "US"],
                       "business_name": ["twisteddeli.com", "TWISTED DELI", "सनराइज ट्रेडर्स", "Metro Holding"],
                       "business_address": ["##4 Dix Rd, <NULL>, Woburn", "4 DIX ROAD, WOBURN, MA", "Pune, महाराष्ट्र",
                                            "3860 Pin Oak Rd, Raleigh, NC"]})
    # pair 4 (S1-2, S3-2) is rule-only: no block columns -> NaN context, blocked 0
    pairs = pd.DataFrame({"block_score": [.8, .9, .2, .5, np.nan], "via": ["r", "r+f", "f", "r", None],
                          "block_rank_s1": [1, 0, 0, 0, np.nan], "block_rank_sx": [0, 0, 1, 0, np.nan]})
    i, j = np.array([0, 0, 1, 2, 1]), np.array([0, 1, 1, 2, 3])
    X = featurize(pairs, rec(s1), rec(sx), i, j, rule_acc=np.array([0, 1, 0, 0, 1]))
    assert len(X) == 5 and X.dtypes.eq(np.float32).all() and X.shape[1] == 65, X.shape
    assert X.core_eq.tolist() == [0, 1, 0, 0, 1] and X.rule_acc.tolist() == [0, 1, 0, 0, 1], X[["core_eq", "rule_acc"]]
    assert X.web_eq[0] == 1 and X.hn_eq[1] == 1 and X.hn_eq[2] == 0 and X.city_eq[1] == 1
    assert X.name_cos[1] > X.name_cos[2] and X.addr_cos[1] > 0.5 > X.addr_cos[2]
    assert X.sx_n[:4].tolist() == [1, 2, 2, 1] and X.s1_n[:4].tolist() == [2, 2, 1, 1]
    assert X.sx_margin[1] > 0 > X.sx_margin[2] and X.s1_gap[1] == 0 and abs(X.s1_gap[0] - 0.1) < 1e-6
    assert X.blocked.tolist() == [1, 1, 1, 1, 0] and X.iloc[4][["s1_n", "sx_n", "via_r"]].isna().all()
    assert X.indic_x.tolist() == [0, 0, 0, 1, 0] and X.is_s3.tolist() == [0, 1, 1, 0, 1]
    assert X.core_ratio[3] > 0.5, "P2 to_latin maps the Indic name before similarity"
    print("features self-check ok", X.shape)


if __name__ == "__main__":
    _check()
