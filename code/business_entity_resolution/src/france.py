"""#31's France (LB 0.9856), label-free throughout. Ported from probes P20 (rescore, R1), P26 (house-number features,
c_hn), P28 (bxneg), P29 (paren_hn, key tier, cap), P30 (veto) and P31 (the chain). run.py drives it (--recipe sub31*):

  world(base)  data/work/p9/fr31 = #9ab's idf144 France world (base) with rf_test_sx_France's house numbers re-parsed
               where an address segment opens with '(n)' ('(520) RUE X': 1.6% of France S2/S3, P6 / P29)
  chn          (own process, ER_WORK = that world) #9ab's X rescored with R1 (the generator's French name fillers NOISE
               leave the core), #6's pruner, then c_hn = m2_big's recipe + 5 house-number features computed over the
               full candidate list (P26) -> p per kept pair
  post(...)    after the bge blend, at France's t: key tier (p := 1), bxneg (p := max(p, 0.99)), optional veto (p := 0),
               cap (an SX over 5 S2 / 6 S3 per S1 loses every pair)

    python code/business_entity_resolution/src/france.py        # self-check
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import normalize as N
from worlds import WORK

C = "France"
NOISE = {"fils", "cie", "associes", "freres", "frs"}   # R1: '& Fils', '& Cie', 'et Associés', 'Frères' keep the hn 81-88%
PAREN = r"(^|,\s*)\((\d+)\)"                          # '(520) RUE X' -> '520 RUE X'
HN = ["hn_cls", "hn_d", "hn_nsame", "hn_nssame", "hn_ns1hn"]
SIB = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]                # the generator's sibling hn offsets (P12)
SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]                     # the ones noise on a true pair (almost) never produces
FLOOR, BOOST, CAP = 0.5, 0.99, {"S2": 5, "S3": 6}      # tier floor, bxneg's p, train's max matches per S1 and source
SLICE = 150_000


def strip(core: str) -> str:
    """R1: the core without the filler words (unless nothing else is left)."""
    keep = [t for t in core.split() if t not in NOISE]
    return " ".join(keep) if keep else core


def paren_hn(addr: pd.Series) -> pd.Series:
    return addr.fillna("").astype(str).str.replace(PAREN, r"\1\2", regex=True)


def world(base: Path) -> Path:
    """base's links + a fixed rf_test_sx_France (only hn changes: a '(n)' segment keeps its digits, so it never was
    a place) + its own p9 cache dir."""
    v = base.parent / "fr31"
    (v / "p9").mkdir(parents=True, exist_ok=True)
    for p in base.iterdir():
        q = v / p.name
        if p.name not in ("rf_test_sx_France.parquet", "p9") and not q.is_symlink():
            q.symlink_to(p.resolve())
    rf = v / "rf_test_sx_France.parquet"
    if not rf.exists():
        r = pd.read_parquet(base / rf.name)
        a = pd.concat([pq.read_table(WORK / f"test_source{k}.parquet", columns=["entity_id", "business_address", "country"],
                                     filters=[("country", "==", C)]).to_pandas() for k in (2, 3)], ignore_index=True)
        a = a.set_index("entity_id").reindex(r.id)
        m = a.business_address.str.contains(PAREN, regex=True).fillna(False).to_numpy()
        old = N.house_number(a.business_address[m], a.country[m]).to_numpy()
        assert (old == r.hn.to_numpy()[m]).all(), "rf hn is not normalize.house_number of the raw address"
        r.loc[m, "hn"] = N.house_number(paren_hn(a.business_address[m]), a.country[m]).to_numpy()
        r.to_parquet(rf.with_suffix(".tmp"), index=False)
        rf.with_suffix(".tmp").rename(rf)
        print(f"{C}: '(n)' house numbers re-parsed on {int(m.sum()):,} S2/S3, {int((old != r.hn.to_numpy()[m]).sum()):,} "
              f"changed -> {rf}", flush=True)
    return v


# ---------------------------------------------------------------- c_hn (P26) on the R1 rescore (P20)
def _hn(rf: Path, want: pa.Array):
    """Per-record house number as int64 (-1 = none; 10+ digits are junk) and its digit count, in `want` order."""
    t = pq.read_table(rf, columns=["id", "hn"])
    pos = pc.index_in(want, value_set=t.column("id")).to_numpy(zero_copy_only=False)
    assert not pd.isna(pos).any(), f"{rf.name}: ids without rule features"
    s = pd.Series(t.column("hn").to_numpy(zero_copy_only=False)[pos.astype(np.int64)])
    s = s.where(s.str.len().between(1, 9), "")
    return pd.to_numeric(s.replace("", "-1")).to_numpy(np.int64), s.str.len().to_numpy(np.int8)


def meta(split: str, c: str, id1: pa.Array, idx: pa.Array, work: Path = WORK) -> dict:
    """hn_feats' per-record house numbers of a split / country (rows of id1 / idx) from its rules.feats files."""
    return dict(zip(("h1", "l1"), _hn(work / f"rf_{split}_s1_{c}.parquet", id1))) | \
        dict(zip(("hx", "lx"), _hn(work / f"rf_{split}_sx_{c}.parquet", idx)))


def hn_feats(q1, qx, nc, M) -> np.ndarray:
    """[n, 5] float32 label-free house-number features over the full candidate list: hn_cls (0 none / equal,
    1 sibling offset, 2 digit drop / add, 3 other small, 4 far), hn_d (hx - h1 clipped to +-100), hn_nsame (other SX of
    the S1 with this SX's hn), hn_nssame (their best name_cos), hn_ns1hn (other SX of the S1 carrying the S1's hn).
    M: h1 / l1 per q1, hx / lx per qx (_hn)."""
    a, la, b, lb = M["h1"][q1], M["l1"][q1], M["hx"][qx], M["lx"][qx]
    both = (a >= 0) & (b >= 0)
    dd = b - a
    diff = both & (dd != 0)
    sib = diff & np.isin(dd, SIB)
    ls, ll = np.minimum(la, lb).astype(np.int64), np.maximum(la, lb).astype(np.int64)
    short, lng = np.where(la < lb, a, b), np.where(la < lb, b, a)
    ends = (lng % (10 ** ls)) == short
    starts = (lng // (10 ** np.clip(ll - ls, 0, 18))) == short
    dig = diff & ~sib & (la != lb) & (ends | starts)
    small = diff & ~sib & ~dig & (np.abs(dd) <= 21)
    far = diff & ~sib & ~dig & ~small
    cls = (1 * sib + 2 * dig + 3 * small + 4 * far).astype(np.float32)
    d = np.where(both, np.clip(dd, -100, 100), np.nan).astype(np.float32)
    n = len(q1)
    nsame, nssame = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    hx = np.flatnonzero(b >= 0)
    if len(hx):                                         # groups (q1, hx) among the SX that carry a house number
        key = (q1[hx].astype(np.int64) << 31) | b[hx]
        v = np.nan_to_num(nc[hx], nan=0.0)
        o = np.lexsort((-v, key))
        ks, vs = key[o], v[o]
        start = np.r_[True, ks[1:] != ks[:-1]]
        gid = np.cumsum(start) - 1
        size = np.bincount(gid)
        first = np.flatnonzero(start)
        gsec = np.where(size >= 2, vs[np.minimum(first + 1, len(vs) - 1)], np.nan)
        other = np.where(start, gsec[gid], vs[first][gid])          # best name_cos among the OTHER members
        ns = (size[gid] - 1).astype(np.float32)
        nsame[hx[o]] = ns
        nssame[hx[o]] = np.where(ns > 0, other, np.nan)
    eqh = (both & (dd == 0)).astype(np.float64)
    ns1 = (np.bincount(q1, weights=eqh, minlength=int(q1.max()) + 1)[q1] - eqh).astype(np.float32)
    return np.column_stack([cls, d, nsame, nssame, ns1]).astype(np.float32)


def chn(model: Path, out: Path):
    """(ER_WORK = world()) q1, qx, c_hn p of every pruner-kept France pair -> out. P20's rescore: blocking and the
    context columns stay #9ab's; every record-derived column (features.strings, rule_acc) and the rule-only extra
    pairs are recomputed with R1's cores and the fixed hn. Kept rows wait in a scratch file (3 GB cap) while the full
    list (q1, qx, name_cos) feeds hn_feats. ~4 min, 2.7 GB."""
    import lightgbm as lgb
    import block as B
    import features as F
    import model as M
    import rules
    t0, tmp = time.time(), out.with_suffix(".kept.tmp")
    t1 = pd.read_parquet(B.TXT["test"] / f"{C}_s1.parquet")
    rf1 = pd.read_parquet(WORK / f"rf_test_s1_{C}.parquet").set_index("id").reindex(t1.entity_id.rename("id")).reset_index()
    rf1["core"] = rf1.core.map(strip).astype("str")
    r1 = F.assemble(rf1[F.RF], t1, pd.Series("", index=t1.index))
    del t1, rf1
    m1, counts, ix1 = F.mats(r1), F.pool_counts(r1), pd.Index(r1.id)
    files, offs = F.sx_files("test", C)
    rfs = F.pieces(WORK / f"rf_test_sx_{C}.parquet", np.diff(offs))
    w, full = None, []
    for k, (f, raw) in enumerate(F.raw_sx("test", C)):
        rf = next(rfs)
        rf["core"] = rf.core.map(strip).astype("str")
        rx = F.assemble(rf[F.RF], pd.read_parquet(f), raw.business_name)
        del raw
        mx, off = F.mats(rx), offs[k]
        acc, _ = rules.match(r1[F.RF], rf[F.RF], chunk=100_000, by_state=True, keep_cands=False)
        ak = F._key(ix1.get_indexer(acc.s1), pd.Index(rx.id).get_indexer(acc.sx) + off)
        del acc
        pf = pq.ParquetFile(WORK / "p4/X" / f"test_{f.stem}.parquet")
        cols, seen = pf.schema_arrow.names, []

        def rescored(X):
            S = F.strings(r1, rx, X.q1.to_numpy(), X.qx.to_numpy() - off, counts, m1, mx).round(3)
            S["rule_acc"] = np.isin(F._key(X.q1, X.qx), ak).astype(np.float32)
            X = X.reset_index(drop=True)
            for c in S.columns:
                X[c] = S[c].to_numpy(np.float32)
            return X[cols]

        def sink(X):
            nonlocal w
            keep = M.kept(X)
            full.append((X.q1.to_numpy(np.int32), X.qx.to_numpy(np.int32), X.name_cos.to_numpy(np.float32)))
            t = pa.Table.from_pandas(X[keep].astype(np.float32).assign(q1=X.q1.to_numpy(np.int32)[keep],
                                                                       qx=X.qx.to_numpy(np.int32)[keep]), preserve_index=False)
            w = w or pq.ParquetWriter(tmp, t.schema)
            w.write_table(t)

        for b in pf.iter_batches(SLICE):
            X = b.to_pandas()
            seen.append(F._key(X.q1, X.qx))
            sink(rescored(X))
        extra = np.setdiff1d(ak, np.concatenate(seen))   # pairs only the changed rules propose
        if len(extra):
            X = pd.DataFrame({c: np.full(len(extra), np.nan, np.float32) for c in cols})
            X["q1"], X["qx"], X["blocked"] = (extra >> 32).astype(np.int32), (extra & 0xFFFFFFFF).astype(np.int32), 0.0
            sink(rescored(X))
        print(f"  {f.stem}: {len(rx):,} SX, rule-only extra pairs {len(extra):,}, {time.time() - t0:.0f}s, "
              f"peak {B.peak_gb():.2f} GB", flush=True)
        del rx, mx, rf, ak
        F.trim()
    w.close()
    del r1, m1, counts
    q1, qx, nc = (np.concatenate(a) for a in zip(*full))
    del full
    kf = (q1.astype(np.int64) << 32) | qx.astype(np.int64)
    assert len(np.unique(kf)) == len(kf), "duplicate pairs in the full list"
    H = pd.DataFrame(hn_feats(q1, qx, nc, meta("test", C, *F.ids("test", C))), columns=HN, index=kf)
    del q1, qx, nc, kf
    X = pd.read_parquet(tmp)
    X[HN] = H.reindex((X.q1.to_numpy(np.int64) << 32) | X.qx.to_numpy(np.int64)).to_numpy()
    del H
    b = lgb.Booster(model_file=str(model))
    p = b.predict(X[b.feature_name()], num_threads=os.cpu_count()).astype(np.float32)
    pd.DataFrame({"q1": X.q1.to_numpy(np.int32), "qx": X.qx.to_numpy(np.int32), "p": p}).to_parquet(out, index=False)
    tmp.unlink()
    print(f"{C}: c_hn p on {len(p):,} kept pairs -> {out}, {time.time() - t0:.0f}s, peak {B.peak_gb():.2f} GB", flush=True)


# ---------------------------------------------------------------- post-rules (P29 tier / cap, P28 bxneg, P30 veto)
def keys(path: Path) -> pd.DataFrame:
    """id -> R1 core, hn, city of a rules.feats file."""
    k = pd.read_parquet(path, columns=["id", "core", "hn", "city"])
    k["core"] = k.core.map(strip).astype("str")
    return k.set_index("id")


def tier_mask(k1: pd.DataFrame, kx: pd.DataFrame, chain: np.ndarray, am: np.ndarray) -> np.ndarray:
    """Key tier: the SX's argmax pair with R1 core, hn and city all equal (non-empty) and the core unique among the
    country's S1 (chain size 1). k1 / kx aligned to the pairs."""
    ok = lambda c: (k1[c].to_numpy() == kx[c].to_numpy()) & (k1[c].to_numpy() != "")
    return am & ok("core") & ok("hn") & ok("city") & (chain == 1)


def cap_mask(q1: np.ndarray, s3: np.ndarray, p: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Predicted pairs beyond CAP per S1 and source (lowest p first): the pairs to drop."""
    d = pd.DataFrame({"q1": q1, "s3": s3, "p": p})[pred]
    rank = d.groupby(["q1", "s3"]).p.rank(method="first", ascending=False).to_numpy()
    drop = np.zeros(len(p), bool)
    drop[np.flatnonzero(pred)[rank > np.where(d.s3.to_numpy(), CAP["S3"], CAP["S2"])]] = True
    return drop


def name_rel(c1: str, cx: str, webx: bool) -> tuple:
    """(kind, word) of the SX core vs the S1 core (P20): eq | add1 w | sub1 w | drop1 | typo | addN | subN | dropN |
    disjoint | web."""
    from rapidfuzz.fuzz import ratio
    if webx:
        return "web", ""
    a, b = c1.split(), cx.split()
    s1, sx = set(a), set(b)
    add, drop = [t for t in b if t not in s1], [t for t in a if t not in sx]
    if not add and not drop:
        return "eq", ""
    if not s1 & sx:
        return "disjoint", ""
    if not add:
        return "drop1" if len(drop) == 1 else "dropN", ""
    if not drop:
        return ("add1", add[0]) if len(add) == 1 else ("addN", " ".join(add))
    if len(add) == 1 and len(drop) == 1:
        return ("typo", "") if ratio(add[0], drop[0]) >= 75 else ("sub1", add[0])
    rest = [x for x in add if max(ratio(x, y) for y in drop) < 75]
    if not rest:
        return "typo", ""
    return ("sub1" if len(rest) == 1 else "subN"), " ".join(rest)


_DROP = N.STOP | {"bis", "ter", "quater"}
_NUM = re.compile(r"[a-z]?[-/]?(\d+)([a-z](?![a-z0-9]))?")     # N.house_number's digit run + an attached letter (12b)


def _street(x: str) -> str:
    segs = x.split(",")
    for i, seg in enumerate(segs):                              # the segment N.house_number takes the number from
        s = seg.strip()
        if N._UNIT.match(s):
            continue
        s = N._HN_LABEL.sub("", s.lstrip("#").strip())
        m = _NUM.match(s)
        if m:
            rest = N._basic(s[m.end():])
            if not rest and i + 1 < len(segs):                  # '12, Main St': the number stands alone
                rest = N._basic(segs[i + 1])
            return " ".join(sorted({t for t in N._seg(rest).split() if t not in _DROP}))
    return ""


def street_key(addr: pd.Series) -> pd.Series:
    """Street of the house-number segment, canonical (N._seg), articles / bis / ter dropped, tokens sorted; ''
    when the address has no house number."""
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_street).astype("str")


def _rf(path: Path, ids, cols: list) -> pd.DataFrame:
    t = pq.read_table(path, columns=["id", *cols])
    t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
    return t.to_pandas().set_index("id").reindex(ids)


def _addr(sources: tuple, ids) -> pd.Series:
    out = [pq.read_table(WORK / f"test_source{i}.parquet", columns=["entity_id", "business_address"],
                         filters=[("country", "==", C)]).to_pandas() for i in sources]
    a = pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").business_address
    return a.reindex(ids).fillna("")


def sib_words(pairs: pd.DataFrame, support: int = 20, share: float = 0.5) -> set:
    """Label-free sibling-word table from a candidate list (s1, sx; P3's blocked list): 'kind:word' (the word an
    insert / swap brings in) whose hn-equal-or-SIB3 pairs take a sibling offset >= share of the time, >= support
    pairs ('X Holding' at hn + 5). Branch words sit near 0.95, true-pair noise words near 0.05."""
    r1 = _rf(WORK / f"rf_test_s1_{C}.parquet", pd.unique(pairs.s1), ["core", "hn"])
    rx = _rf(WORK / f"rf_test_sx_{C}.parquet", pd.unique(pairs.sx), ["core", "hn", "is_web"])
    h1, hx = r1.hn.reindex(pairs.s1).fillna("").to_numpy(), rx.hn.reindex(pairs.sx).fillna("").to_numpy()
    both = np.flatnonzero((h1 != "") & (hx != ""))
    dl = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    rel, dl = both[(dl == 0) | np.isin(dl, SIB3)], dl[(dl == 0) | np.isin(dl, SIB3)]
    s1r, sxr = pairs.s1.to_numpy()[rel], pairs.sx.to_numpy()[rel]
    c1, cx = r1.core.reindex(s1r).fillna("").to_numpy(), rx.core.reindex(sxr).fillna("").to_numpy()
    wx = rx.is_web.reindex(sxr).fillna(False).to_numpy()
    g = pd.DataFrame({"k": [f"{k}:{w}" for k, w in (name_rel(a, b, x) for a, b, x in zip(c1, cx, wx))],
                      "s": np.isin(dl, SIB3)})
    g = g[g.k.str.startswith(("add1:", "sub1:"))].groupby("k").s.agg(["size", "mean"])
    return set(g.index[(g["size"] >= support) & (g["mean"] >= share)])


def bxneg(s1: np.ndarray, sx: np.ndarray, qx: np.ndarray, p: np.ndarray, t: float, sibw: set) -> np.ndarray:
    """P28's same-address rule 'bxneg': hn equal (both present), the SX core = the S1 core or one word inserted /
    dropped (not a sibling word of sibw), a shared place, the same street (exact key), and the pair is its SX's argmax
    or the SX is matched nowhere (best p < t). Reads the unfixed rules.feats files, as P31 did."""
    import model as M
    r1 = _rf(WORK / f"rf_test_s1_{C}.parquet", pd.unique(s1), ["core", "hn", "places"])
    rx = _rf(WORK / f"rf_test_sx_{C}.parquet", pd.unique(sx), ["core", "hn", "places", "is_web"])
    h1, hx = r1.hn.reindex(s1).fillna("").to_numpy(), rx.hn.reindex(sx).fillna("").to_numpy()
    am = M.argmax_sx(qx, p)
    best = pd.Series(p).groupby(qx).transform("max").to_numpy()
    both = (h1 != "") & (hx != "")
    dlt = np.full(len(p), np.nan)
    dlt[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    el = np.flatnonzero((dlt == 0) & (am | (best < t)))           # hn equal, and the boost cannot steal
    nr = [name_rel(a, b, w) for a, b, w in zip(r1.core.reindex(s1[el]).fillna("").to_numpy(),
                                               rx.core.reindex(sx[el]).fillna("").to_numpy(),
                                               rx.is_web.reindex(sx[el]).fillna(False).to_numpy())]
    ok = np.array([k in ("add1", "drop1", "eq") and f"{k}:{w}" not in sibw for k, w in nr], bool)
    cand = el[ok] if len(el) else el
    s1c, sxc = s1[cand], sx[cand]
    p1, px = r1.places.reindex(s1c).fillna("").to_numpy(), rx.places.reindex(sxc).fillna("").to_numpy()
    city = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)], bool)
    k1 = street_key(_addr((1,), pd.unique(s1c))).reindex(s1c).fillna("").to_numpy()
    kx = street_key(_addr((2, 3), pd.unique(sxc))).reindex(sxc).fillna("").to_numpy()
    out = np.zeros(len(p), bool)
    out[cand[city & (k1 != "") & (k1 == kx)]] = True
    return out


def post(q1, qx, id1: pa.Array, idx: pa.Array, p: np.ndarray, g: np.ndarray, t: float, v: Path, learn: pd.DataFrame,
         veto: tuple | None = None) -> np.ndarray:
    """Blended France p -> tier -> bxneg -> [veto] -> cap, float64 inside (as P31), float32 out. g = c_hn's p (before
    the blend); v = world(); learn = P3's blocked list (s1, sx) for bxneg's sibling words; veto = (tau, r) with r the
    bge score per pair (NaN = none): a pair with g >= 0.98 and r < tau loses p unless tier / bxneg raised it."""
    import model as M
    s1 = id1.take(pa.array(q1)).to_numpy(zero_copy_only=False)
    sx = idx.take(pa.array(qx)).to_numpy(zero_copy_only=False)
    p = p.astype(np.float64)
    dec = lambda: M.argmax_sx(qx, p) & (p >= t)
    last = dec()
    k1 = keys(WORK / f"rf_test_s1_{C}.parquet")
    chain = k1.core.map(k1.core.value_counts()).reindex(s1).to_numpy()
    kx = keys(v / f"rf_test_sx_{C}.parquet")
    ruled = tier_mask(k1.reindex(s1), kx.reindex(sx).fillna(""), chain, M.argmax_sx(qx, p)) & (p >= FLOOR)
    del k1, kx, chain
    p[ruled] = 1.0
    log = lambda name, n: print(f"{C}: {name} {n:,} pairs, accepted {int(last.sum()):,} -> {int(dec().sum()):,}", flush=True)
    log("key tier", int(ruled.sum()))
    last = dec()
    bx = bxneg(s1, sx, qx, p, t, sib_words(learn))
    p = np.where(bx, np.maximum(p, BOOST), p)
    ruled |= bx
    log("bxneg", int(bx.sum()))
    last = dec()
    if veto:
        tau, r = veto
        vv = (g.astype(np.float64) >= 0.98) & (np.nan_to_num(r.astype(np.float64), nan=1.0) < tau)
        p[vv & ~ruled] = 0.0
        log(f"veto (tau {tau:g}, rule pairs exempt)", int((vv & ~ruled).sum()))
        last = dec()
    s3 = pc.starts_with(idx.take(pa.array(qx)), "S3-").to_numpy(zero_copy_only=False)
    drop = cap_mask(q1, s3, p, last)
    p[np.isin(qx, qx[drop])] = 0.0
    log("cap (their SX lose every pair)", int(drop.sum()))
    return p.astype(np.float32)


def _check():
    assert [strip(c) for c in ["martin fils", "fils", "wallon cie", "dupont freres", ""]] == ["martin", "fils", "wallon",
                                                                                              "dupont", ""]
    assert paren_hn(pd.Series(["(520) RUE X, Lyon", "A, (12) Rue", "12 (3) rue"])).tolist() == \
        ["520 RUE X, Lyon", "A, 12 Rue", "12 (3) rue"]
    k = street_key(pd.Series(["12 Rue de la Paix, Lille", "N°12 R. DE LA PAIX, LILLE", "12, Main Street, Austin, TX",
                              "Lille, Nord"])).tolist()
    assert k[0] == k[1] == "paix r" and k[2] == "main st" and k[3] == "", k
    assert [name_rel(*x, False)[0] for x in [("a b", "a b"), ("a b", "a b c"), ("a b", "a"), ("a b", "a c"),
                                             ("le garage", "le garaje")]] == ["eq", "add1", "drop1", "sub1", "typo"]
    M = {"h1": np.array([10, 10, -1]), "l1": np.array([2, 2, 0], np.int8),
         "hx": np.array([13, 1, 10, 300]), "lx": np.array([2, 1, 2, 3], np.int8)}
    H = hn_feats(np.array([0, 0, 0, 0, 2]), np.array([0, 1, 2, 3, 0]), np.array([.9, .5, .7, .1, .3], np.float32), M)
    assert H[:4, 0].tolist() == [1, 2, 0, 4] and H[:4, 1].tolist() == [3, -9, 0, 100] and np.isnan(H[4, 1]), H
    assert H[:, 4].tolist() == [1, 1, 0, 1, 0], H                       # one other SX of S1 0 carries its hn 10
    s3, pred = np.array([False] * 7 + [True]), np.ones(8, bool)
    assert cap_mask(np.zeros(8, int), s3, np.arange(8) / 10, pred).tolist() == [True, True] + [False] * 6
    print("france self-check ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["chn"]:                        # run.france31's subprocess: france.py chn MODEL OUT
        chn(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        _check()
