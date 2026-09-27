"""The unseen-country chain (#31's France, LB 0.9856; + #34v / #37's decision-time additions, LB 0.986304), label-free
throughout. It runs on every test S1 label absent from train S1 (run.unseen(): France is the only one); C is set by
run.py, no country is named here. Ported from probes P20 (rescore, R1), P26 (house-number features, c_hn), P28 (bxneg),
P29 (paren_hn, key tier, cap), P30 (veto), P31 (the chain), P34v (rescue) and P37 (group completion). run.py drives
it (--recipe sub31* / sub37*):

  world(base)  data/work/p9/fr31 = #9ab's idf144 France world (base) with rf_test_sx_France's house numbers re-parsed
               where an address segment opens with '(n)' ('(520) RUE X': 1.6% of France S2/S3, P6 / P29)
  chn          (own process, ER_WORK = that world) #9ab's X rescored with R1 (the generator's French name fillers NOISE
               leave the core), #6's pruner, then c_hn = m2_big's recipe + 5 house-number features computed over the
               full candidate list (P26) -> p per kept pair
  post(...)    after the bge blend, at France's t: key tier (p := 1), bxneg (p := max(p, 0.99)), optional veto (p := 0),
               cap (an SX over 5 S2 / 6 S3 per S1 loses every pair)
  complete(...) after the decision (argmax, p >= t): empty-S1 rescue (#34v_r90, p >= 0.90), then group completion of
               one-sided groups (#37_gc70, p >= 0.70), then optional hn-equal extras for two-sided groups (hn90), then
               optional synthetic-brand recovery (P41, brand_mask: #41 at p >= 0.30)

    python code/business_entity_resolution/src/france.py        # self-check
"""
import collections
import json
import os
import re
import sys
import time
import unicodedata
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

C = None               # the unseen country label (run.france31 / the chn argv set it)
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
        fixed = paren_hn(a.business_address)
        m = (fixed != a.business_address.fillna("").astype(str)).to_numpy()     # the address has a '(n)' segment
        old = N.house_number(a.business_address[m], a.country[m]).to_numpy()
        assert (old == r.hn.to_numpy()[m]).all(), "rf hn is not normalize.house_number of the raw address"
        r.loc[m, "hn"] = N.house_number(fixed[m], a.country[m]).to_numpy()
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


# ---------------------------------------------------------------- after the decision (P34v rescue, P37 completion)
def _best_per(key: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Mask: the highest-p row of m per key (ties -> the first row)."""
    o = np.flatnonzero(m)
    o = o[np.lexsort((-p[o], key[o]))]
    out = np.zeros(len(p), bool)
    out[o[np.r_[True, key[o][1:] != key[o][:-1]]] if len(o) else o] = True
    return out


def rescue(q1, p, am, acc, sib, t: float) -> np.ndarray:
    """#34v_r90's empty-S1 rescue: an S1 with no accepted pair gets its best argmax, non-sibling pair if p >= t. The SX
    of an unaccepted argmax pair is accepted nowhere, so the one-S1-per-SX rule holds."""
    has = np.bincount(q1[acc], minlength=int(q1.max()) + 1)[q1] > 0
    return _best_per(q1, p, am & ~sib & ~has & ~acc & (p >= t))


def gc_add(q1, qx, is2, p, am, acc, sib, t: float, guard: np.ndarray | None = None, empty: bool = False) -> np.ndarray:
    """P37's group completion: an S1 whose accepted group holds one source only (S2s or S3s) gets every free (SX
    accepted nowhere) argmax, non-sibling pair of the missing source with p >= t, best p first, up to CAP per source.
    guard (per pair, e.g. hn equal): a pair of a two-sided group that passes it joins too (#37_gc70_hn90). In the
    train truth 81% of S1 hold both sources and 14% one; France's predictions were one-sided for 20.7% of its S1.
    empty: guarded pairs may also join an S1 with no members (P40's xf_noaddr)."""
    free = np.bincount(qx[acc], minlength=int(qx.max()) + 1)[qx] == 0
    ns = int(q1.max()) + 1
    n2, n3 = (np.bincount(q1[acc & m], minlength=ns)[q1] for m in (is2, ~is2))
    miss = np.where(is2, (n2 == 0) & (n3 > 0), (n3 == 0) & (n2 > 0))       # this row's source is the missing one
    c = am & ~acc & free & ~sib & (p >= t) & (miss if guard is None else miss | (((n2 + n3 > 0) | empty) & guard))
    o = np.flatnonzero(c)
    k = q1 * 2 + is2
    o = o[np.lexsort((-p[o], k[o]))]
    i = np.arange(len(o))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[o][1:] != k[o][:-1]], i, 0))
    room = np.where(is2, CAP["S2"] - n2, CAP["S3"] - n3)
    out = np.zeros(len(p), bool)
    out[o[rank < room[o]]] = True
    return out


def complete(q1, qx, is2, p, am, acc, h1, hx, rescue_t=None, gc_t=None, hn_t=None, brand=None, noaddr=None,
             swap=None) -> np.ndarray:
    """acc (the decision: am & p >= t) plus, in the staged files' order, the rescue (rescue_t), the group completion
    (gc_t), the hn-equal two-sided extras (hn_t), the brand recovery (brand = (t, brand_mask per pair)) and the
    address-less exact-name variants (noaddr = (t, noaddr_mask per pair), any S1, empty ones too) and the noise-word
    swaps (swap = (s_max, swap_mask per pair), any S1); None = off.
    p is compared in float64, as P37 / P41 built them. Siblings (stack.sib_mask on h1 / hx, the house numbers of q1 /
    qx) never join."""
    import stack as S
    p, sib = np.asarray(p, np.float64), S.sib_mask(h1, hx)

    def add(name, t, m):
        print(f"{C}: {name} p >= {t:g}: +{int(m.sum()):,} pairs to {len(np.unique(q1[m])):,} S1", flush=True)
        return acc | m

    if rescue_t is not None:
        acc = add("empty-S1 rescue", rescue_t, rescue(q1, p, am, acc, sib, rescue_t))
    if gc_t is not None:
        acc = add("group completion", gc_t, gc_add(q1, qx, is2, p, am, acc, sib, gc_t))
    if hn_t is not None:
        acc = add("hn-equal extras", hn_t, gc_add(q1, qx, is2, p, am, acc, sib, hn_t, (h1 != "") & (h1 == hx)))
    if brand is not None:                               # any non-empty group, guarded pairs only (P41)
        acc = add("brand recovery", brand[0], gc_add(q1, qx, is2, p, am, acc, sib | ~brand[1], brand[0],
                                                     np.ones(len(p), bool)))
    if noaddr is not None:                              # any S1, guarded pairs only (P40)
        acc = add("address-less exact names", noaddr[0], gc_add(q1, qx, is2, p, am, acc, sib | ~noaddr[1], noaddr[0],
                                                                np.ones(len(p), bool), empty=True))
    if swap is not None:                                # any S1, guarded pairs only (P46; the mask holds its p floor)
        acc = add("noise-word swaps (s_max)", swap[0], gc_add(q1, qx, is2, p, am, acc, sib | ~swap[1], 0.0,
                                                             np.ones(len(p), bool), empty=True))
    return acc


# ---------------------------------------------------------------- P46: noise-word swap tier (P33's R05 guards + a lens)
SWAP = dict(support=20, e_min=0.8, floor=0.02)       # P33 SUPPORT / P46 E_MIN / P33 FLOOR


def lens_table(s1: np.ndarray, sx: np.ndarray) -> pd.DataFrame:
    """P46's label-free lens over pairs (the country's argmax pairs): per word w that a one-word swap brings in (P20
    name_rel 'sub1'), n = its swap pairs with a house number on both sides, s = their share at a generator sibling
    offset (SIB3), e = their share keeping the house number. Noise words: s ~ 0, e >= 0.8; decoy words s >= 0.08."""
    r1 = _rf(WORK / f"rf_test_s1_{C}.parquet", pd.unique(s1), ["core", "hn"])
    rx = _rf(WORK / f"rf_test_sx_{C}.parquet", pd.unique(sx), ["core", "hn", "is_web"])
    c1, cx = r1.core.reindex(s1).fillna("").to_numpy(), rx.core.reindex(sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(sx).fillna(False).to_numpy()
    kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, wx)]
    sub = np.array([k == "sub1" for k, _ in kw], bool)
    words = np.array([w for _, w in kw], object)[sub]
    del kw, c1, cx, wx
    h1, hx = r1.hn.reindex(s1[sub]).fillna("").to_numpy(), rx.hn.reindex(sx[sub]).fillna("").to_numpy()
    both = (h1 != "") & (hx != "")
    dl = np.full(len(h1), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    g = pd.DataFrame({"w": words, "sib": np.isin(dl, SIB3), "eq": dl == 0, "hn": both}).groupby("w")
    g = g.agg(n=("sib", "size"), s=("sib", "mean"), n_hn=("hn", "sum"), n_eq=("eq", "sum"))
    g["e"] = g.n_eq / g.n_hn.clip(lower=1)
    return g.assign(n_all=g.n, s=g.s * g.n / g.n_hn.clip(lower=1), n=g.n_hn)          # shares over hn-present pairs


def swap_mask(s1: np.ndarray, sx: np.ndarray, p: np.ndarray, t: float, learn: pd.DataFrame, s_max: float) -> np.ndarray:
    """P46's noise-word swap tier per pair (s1 / sx ids of the country's scored list, p = its post-rules p): the SX's
    argmax pair, not accepted at the post-rules t, whose core swaps one S1 word for a word w (name_rel 'sub1') with
    the house number equal, a shared place and the same street (P28), and P33's R05 guards: w is not an R1 filler, not
    a learned sibling word (sib_words on `learn`, P3's blocked list); no other candidate S1 of the SX holds the rest of
    the SX core; the SX core is no S1's core in the S1's city; p >= 0.02. The lens (lens_table on the argmax pairs)
    keeps w with >= 20 hn-present swap pairs, s <= s_max (0.05; P46 'lens_g': 0.09) and e >= 0.8: the generator's
    French noise words (développement, groupe) at the identical address. France +3,185 pairs (P46)."""
    qx = pd.factorize(sx)[0]
    am = argmax(qx, p)
    st = lens_table(s1[am], sx[am])
    r1 = _rf(WORK / f"rf_test_s1_{C}.parquet", pd.unique(s1), ["core", "hn", "places", "city"])
    rx = _rf(WORK / f"rf_test_sx_{C}.parquet", pd.unique(sx), ["core", "hn", "places", "is_web"])
    h1, hx = r1.hn.reindex(s1).fillna("").to_numpy(), rx.hn.reindex(sx).fillna("").to_numpy()
    both = (h1 != "") & (hx != "")
    dl = np.full(len(p), np.nan)
    dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    eq = (dl == 0) & am                                 # P28's numeric hn equality
    ci = np.flatnonzero(eq)
    c1 = r1.core.reindex(s1[ci]).fillna("").to_numpy()
    cx = rx.core.reindex(sx[ci]).fillna("").to_numpy()
    kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, rx.is_web.reindex(sx[ci]).fillna(False).to_numpy())]
    ok = np.array([k == "sub1" for k, _ in kw], bool)
    p1 = r1.places.reindex(s1[ci]).fillna("").to_numpy()
    px = rx.places.reindex(sx[ci]).fillna("").to_numpy()
    ok &= np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)], bool)
    cand, c1, cx = ci[ok], c1[ok], cx[ok]
    w = np.array([k[1] for k, o in zip(kw, ok) if o], object)
    del kw, p1, px
    s1c, sxc = s1[cand], sx[cand]
    k1 = street_key(_addr((1,), pd.unique(s1c))).reindex(s1c).fillna("").to_numpy()
    kx = street_key(_addr((2, 3), pd.unique(sxc))).reindex(sxc).fillna("").to_numpy()
    street = (k1 != "") & (k1 == kx)
    rest = [frozenset(b.split()) - {x} for b, x in zip(cx, w)]
    mine = pd.DataFrame({"sx": sxc, "s1": s1c, "i": np.arange(len(cand))})
    oth = mine.merge(pd.DataFrame({"sx": sx, "o": s1}), on="sx")
    oth = oth[oth.o != oth.s1]
    oc = _rf(WORK / f"rf_test_s1_{C}.parquet", pd.unique(oth.o), ["core"]).core.reindex(oth.o).fillna("").to_numpy()
    hit = np.array([rest[i] <= set(o.split()) for i, o in zip(oth.i.to_numpy(), oc)], bool)
    uniq = ~np.isin(np.arange(len(cand)), oth.i.to_numpy()[hit])
    pool = pd.read_parquet(WORK / f"rf_test_s1_{C}.parquet", columns=["core", "city"])
    keys = set(zip(pool.core.to_numpy(), pool.city.to_numpy()))
    city1 = r1.city.reindex(s1c).fillna("").to_numpy()
    nocore = np.array([(b, cc) not in keys for b, cc in zip(cx, city1)], bool)
    del pool, keys, oth
    sibw = sib_words(learn)
    sib_ok = ~np.isin(np.array([f"sub1:{x}" for x in w], object), list(sibw))
    ws = pd.Series(w)
    lens = (ws.map(st.n).fillna(0).to_numpy() >= SWAP["support"]) & (ws.map(st.s).fillna(1.0).to_numpy() <= s_max) & \
        (ws.map(st.e).fillna(0.0).to_numpy() >= SWAP["e_min"])
    keep = ~np.isin(w, list(NOISE)) & (p[cand] >= SWAP["floor"]) & uniq & nocore & sib_ok & street & lens
    out = np.zeros(len(p), bool)
    out[cand[keep]] = True
    out &= ~(p >= t)                                    # P46: pairs accepted at the post-rules t are not flips
    print(f"{C}: noise-word swap tier: {len(cand):,} hn-equal one-word swaps in a shared place, {int(keep.sum()):,} pass, "
          f"{int(out.sum()):,} not accepted at t {t:g}; lens words "
          f"{sorted(st.index[(st.n >= SWAP['support']) & (st.s <= s_max) & (st.e >= SWAP['e_min'])])[:8]}", flush=True)
    return out


def fkey(s: str) -> str:
    """P40's full normalised name, legal forms kept: accents stripped, lower case, '&' -> 'and', word tokens."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))


def noaddr_mask(s1: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """P40's xf_noaddr per pair (s1 / sx ids): the SX has an empty address and its full normalised name (fkey, legal
    form kept) equals the S1's, a name >= 5 characters that no other S1 of the country carries."""
    raw = lambda k, cols: pq.read_table(WORK / f"test_source{k}.parquet", columns=["entity_id", *cols],
                                        filters=[("country", "==", C)]).to_pandas()
    a = raw(1, ["business_name"])
    k1 = pd.Series([fkey(n) for n in a.business_name], index=a.entity_id.to_numpy())
    n = k1.value_counts()
    k1 = k1[(k1.str.len() >= 5) & (k1.map(n) == 1)]                    # unique, long enough
    x = pd.concat([raw(k, ["business_name", "business_address"]) for k in (2, 3)], ignore_index=True)
    x = x[x.business_address.fillna("").str.strip() == ""]
    kx = pd.Series([fkey(n) for n in x.business_name], index=x.entity_id.to_numpy())
    print(f"{C}: {len(kx):,} address-less SX, {len(k1):,} unique S1 full names", flush=True)
    return (kx.reindex(sx).to_numpy(object) == k1.reindex(s1).to_numpy(object)) & pd.notna(k1.reindex(s1)).to_numpy()


# ---------------------------------------------------------------- P41: synthetic-brand ("gib") recovery (P36's detector)
_DIG = str.maketrans("0158", "olsb")                  # the generator's digit-for-letter typos
_WEBN = re.compile(r"www|\.[a-z]{2,4}\b|@|#|com\b", re.I)


def btoks(s: str) -> list:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().translate(_DIG)
    return re.findall(r"[a-z]+", s)


def _core(oov, min_end: int = 100, k: int = 25) -> list:
    """The k-core of 3-7 letter strings that start AND end >= min_end OOV tokens, linked when two glue into one."""
    pre, suf = collections.Counter(), collections.Counter()
    for t in oov:
        for L in range(3, min(8, len(t) - 2)):
            pre[t[:L]] += 1
            suf[t[-L:]] += 1
    cand = {m for m in pre if pre[m] >= min_end and suf[m] >= min_end}
    edges = collections.defaultdict(set)
    for t in oov:
        for L in range(3, len(t) - 2):
            a, b = t[:L], t[L:]
            if a in cand and b in cand:
                edges[a].add(b)
                edges[b].add(a)
    m = set(cand)
    while drop := {x for x in m if len(edges[x] & m) < k}:
        m -= drop
    return sorted(m)


def syllables(split: str = "test") -> list:
    """P36: the generator's brand syllables, label-free from the split's own names (every country): SX name tokens no
    S1 name uses (non-web names, >= 6 letters) -> _core. 44 syllables on train and on test. Cached (data/work/p36)."""
    path = WORK / "p36" / f"morph_{split}.json"
    if not path.exists():
        names = lambda i: (n for b in pq.ParquetFile(WORK / f"{split}_source{i}.parquet").iter_batches(
            500_000, columns=["business_name"]) for n in b.column(0).to_pylist())
        v1 = {t for n in names(1) for t in btoks(n)}
        oov = {t for i in (2, 3) for n in names(i) if not _WEBN.search(n) for t in btoks(n) if len(t) >= 6 and t not in v1}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_core(oov)))
    return json.loads(path.read_text())


def n_syll(t: str, m: set) -> int:
    """Fewest syllables that spell token t exactly (0 = it cannot be spelled)."""
    best = [0] + [99] * len(t)
    for i in range(len(t)):
        if best[i] < 99:
            for L in range(2, min(8, len(t) - i) + 1):
                if t[i:i + L] in m:
                    best[i + L] = min(best[i + L], best[i] + 1)
    return best[-1] if best[-1] < 99 else 0


def brand_mask(s1: np.ndarray, sx: np.ndarray, v: Path) -> np.ndarray:
    """P41's guard per pair (s1 / sx ids): the SX's name is one bare synthetic brand (a single token spelled by >= 2
    syllables, no word kept from the S1's name) and the pair's S1 is the only S1 of the country whose hn + street + city
    key equals the SX's (hn / city: rules.feats, the SX side from world v; street: street_key; all non-empty). In train
    truth such an SX is that S1's 99.8% (US) / 96.1% (India) of the time."""
    m = set(syllables("test"))
    raw = lambda k, cols: pq.read_table(WORK / f"test_source{k}.parquet", columns=["entity_id", *cols],
                                        filters=[("country", "==", C)]).to_pandas()
    x = pd.concat([raw(k, ["business_name", "business_address"]) for k in (2, 3)], ignore_index=True)
    x = x[np.fromiter((len(t) == 1 and n_syll(t[0], m) >= 2 for t in map(btoks, x.business_name)), bool, len(x))]
    key = ["hn", "street", "city"]
    rx = _rf(v / f"rf_test_sx_{C}.parquet", x.entity_id.to_numpy(), ["hn", "city"])
    kx = pd.DataFrame({"hn": rx.hn.fillna("").to_numpy(), "city": rx.city.fillna("").to_numpy(),
                       "street": street_key(x.business_address).to_numpy()}, index=x.entity_id.to_numpy())
    a = raw(1, ["business_address"])
    r1 = _rf(WORK / f"rf_test_s1_{C}.parquet", a.entity_id.to_numpy(), ["hn", "city"])
    k1 = pd.DataFrame({"id": a.entity_id.to_numpy(), "hn": r1.hn.fillna("").to_numpy(), "city": r1.city.fillna("").to_numpy(),
                       "street": street_key(a.business_address).to_numpy()})
    g = k1[(k1[key] != "").all(axis=1)].groupby(key).id.agg(["first", "size"])
    j = kx.merge(g, left_on=key, right_index=True, how="left")
    one = ((kx[key] != "").all(axis=1) & (j["size"] == 1)).to_numpy()
    key1 = pd.Series(np.where(one, j["first"].fillna("").to_numpy(object), ""), index=kx.index)
    print(f"{C}: {len(kx):,} bare brand SX, {int(one.sum()):,} with a unique hn + street + city S1", flush=True)
    return key1.reindex(sx).fillna("").to_numpy(object) == s1


# ---------------------------------------------------------------- the final chain (#sub_clean, 27 Sep): P42 recipe stages ->
# P41 brand (on the gc70 lists) -> P48 CE veto -> P54 in-band CE rescue -> P47 hn-change class -> P55 combiner. Each stage
# is its probe's rule, on the same France pair rows (the post-rules file order); stage functions are the probes' code.
LO, HI, CAPS = 0.02, 0.98, (5, 6)                  # the CE band (run.py rerank.CFG) and train's max S2 / S3 per S1
W_BLEND = (0.9245, 0.2595, 0.0515)                 # the unseen country's out-of-domain bge blend (= run.W_FRANCE)
HN_OK = (4, 5)                                     # P47: hn_rel classes of the class stage (-1 / -2, other non-sibling)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rank_in(key, p, rows):
    """rows sorted by (key, -p) and each row's 0-based rank within its key (P42)."""
    o = rows[np.lexsort((-p[rows], key[rows]))]
    k = key[o]
    i = np.arange(len(o))
    return o, i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))


def counts(s1, is2, pred, n):
    return np.bincount(s1[pred & is2], minlength=n), np.bincount(s1[pred & ~is2], minlength=n)


def p42_base(s1, is2, p, am, sib, t, cap=CAPS):
    """P42: accept argmax & p >= t, then the cap (an S1's lowest pairs over c2 S2 / c3 S3 leave, and their SX leaves
    every later stage). -> (accepted, free = the pairs a later stage may add)."""
    key = s1 * 2 + is2
    acc = am & (p >= t)
    o, rk = rank_in(key, p, np.flatnonzero(acc))
    over = o[rk >= np.where(is2[o], cap[0], cap[1])]
    acc[over] = False
    free = am & ~sib
    free[over] = False
    return acc, free


def p42_decide(s1, is2, p, am, sib, hn_eq, n, t, t_r, t_gc, t_hn, cap=CAPS) -> dict:
    """P42's recipe stages (cumulative masks base / rescue / gc / hn): rescue = an empty S1's best free pair, gc = a
    one-sided S1's free pairs of the missing source, hn = a non-empty S1's free hn-equal pairs; best p first, caps."""
    out = {}
    out["base"], free = p42_base(s1, is2, p, am, sib, t, cap)
    pred = out["base"]
    for stage, bar in (("rescue", t_r), ("gc", t_gc), ("hn", t_hn)):
        if bar is not None:
            n2, n3 = counts(s1, is2, pred, n)
            m2, m3 = n2[s1], n3[s1]
            ok = {"rescue": m2 + m3 == 0, "gc": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0)),
                  "hn": (m2 + m3 > 0) & hn_eq}[stage]
            c = np.flatnonzero(free & ~pred & ok & (p >= bar))
            if stage == "rescue":
                o, rk = rank_in(s1, p, c)
                take = o[rk == 0]
            else:
                o, rk = rank_in(s1 * 2 + is2, p, c)
                take = o[rk < np.where(is2[o], cap[0] - m2[o], cap[1] - m3[o])]
            pred = pred.copy()
            pred[take] = True
        out[stage] = pred
    return out


def hn_rel(h1, hx) -> np.ndarray:
    """P39 / P47: 0 hn missing on a side (or unparsable), 1 equal, 2 SIB3 offset, 3 hx - h1 in {1, 2}, 4 in {-1, -2},
    5 any other difference."""
    ok = (h1 != "") & (hx != "")
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[ok]), errors="coerce").to_numpy(float)
    out = np.where(~ok | np.isnan(d), 0, np.where(d == 0, 1, np.where(np.isin(d, SIB3), 2,
                   np.where(np.isin(d, [1, 2]), 3, np.where(np.isin(d, [-1, -2]), 4, 5)))))
    return out.astype(np.int8)


def fill_add(s1, is2, key_p, pred, cand, n, cap=CAPS) -> np.ndarray:
    """Candidate rows cand added best key_p first per (S1, source), up to the caps counted on pred (P47 / P54 / P42)."""
    n2, n3 = counts(s1, is2, pred, n)
    o, rk = rank_in(s1 * 2 + is2, key_p, cand)
    take = o[rk < np.where(is2[o], cap[0] - n2[s1[o]], cap[1] - n3[s1[o]])]
    out = np.zeros(len(pred), bool)
    out[take] = True
    return out


# ---- P55: the combiner transferred to the unseen country (LightGBM over country-invariant inputs, fit on both LOCO frames)
def sx_feats(sx, g):
    """Per pair: its SX's candidate count, count with g >= 0.5 and g mass excluding the pair, and the best competitor's g."""
    o = np.lexsort((-g, sx))
    so = sx[o]
    first = np.r_[True, so[1:] != so[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    n = np.diff(np.r_[start, len(o)])
    top1 = g[o][start]
    top2 = np.where(n > 1, g[o][np.minimum(start + 1, len(o) - 1)], 0.0)
    rank = np.arange(len(o)) - start[grp]
    out = np.empty((len(g), 4))
    gs = g[o]
    out[o, 0] = n[grp]
    out[o, 1] = np.bincount(grp, weights=(gs >= 0.5).astype(float))[grp] - (gs >= 0.5)
    out[o, 2] = np.bincount(grp, weights=gs)[grp] - gs
    out[o, 3] = np.where(rank == 0, top2[grp], top1[grp])
    return out


def _dens(v, n):
    q = np.sort(v)
    return (len(q) - np.searchsorted(q, v, side="left")) / n


def p55_feats(s1, is2, g, r, p, hr, ref, stage, sxf, n) -> pd.DataFrame:
    """P55's feature frame on the argmax pairs (densities of p / g, CE r, S1 group ranks / gaps / counts, SX exclusivity)."""
    key = s1 * 2 + is2
    rows = np.arange(len(s1))
    o, rk = rank_in(key, p, rows)
    po = p[o]
    same = np.r_[False, key[o][1:] == key[o][:-1]]
    gap_prev = np.full(len(s1), np.nan)
    gap_next = np.full(len(s1), np.nan)
    gap_prev[o[same]] = po[np.flatnonzero(same) - 1] - po[same]
    nxt = np.r_[same[1:], False]
    gap_next[o[nxt]] = po[nxt] - po[np.flatnonzero(nxt) + 1]
    cnt = np.bincount(key, minlength=2 * n)
    acc = np.bincount(key[ref], minlength=2 * n)
    mx = np.full(2 * n, -1.0)
    np.maximum.at(mx, key, p)
    oth = s1 * 2 + (~is2)
    return pd.DataFrame({
        "dens_p": _dens(p, n), "dens_g": _dens(g, n), "lr": np.where(np.isnan(r), np.nan, _logit(np.nan_to_num(r, nan=0.5))),
        "is2": is2.astype(np.int8), "hn_rel": hr.astype(np.int8), "stage": stage.astype(np.int8),
        "region": np.select([g < LO, g >= HI], [0, 2], 1).astype(np.int8),
        "s1_rank": rk[np.argsort(o)], "gap_prev": gap_prev, "gap_next": gap_next,
        "n_same": cnt[key], "n_oth": cnt[oth], "acc_same": acc[key] - ref, "acc_oth": acc[oth],
        "max_oth": mx[oth], "s1_psum": np.bincount(s1, weights=p, minlength=n)[s1],
        "sx_n": sxf[:, 0], "sx_n05": sxf[:, 1], "sx_psum": sxf[:, 2], "sx_gap": _logit(g) - _logit(np.maximum(sxf[:, 3], 1e-6)),
    }).astype(np.float32)


def p55_score(boosters, X) -> np.ndarray:
    """The bagged combiner: sigmoid of the mean logit of the boosters' probabilities."""
    lg = np.mean([_logit(1 / (1 + np.exp(-b.predict(X, raw_score=True)))) for b in boosters], 0)
    return 1 / (1 + np.exp(-lg))


def p42_frame(f: dict, c: dict, s1, sx, is2, sib, hn_eq, g, r, n) -> dict:
    """P42's France frame as P55 / P54's fr54 read it (p42.fr_p at tau 0.1 on the no-veto post-rules p0: exact but for
    the post-rules cap; P42 decide at t 0.99; ref = recipe - P48's veto, 'unmoved' = p0 on the blend of g and r)."""
    p0 = f["p0"].astype(np.float64)
    rn = np.nan_to_num(r, nan=1.0)
    cond = (g >= HI) & (rn < 0.1)
    q = np.where(cond & ~(cond & (f["p"] > 0)), 0.0, p0)
    amq = argmax(sx, q)
    st = p42_decide(s1, is2, q, amq, sib, hn_eq, n, 0.99, c["rescue"], c["gc"], c["hn"])
    bl = 1 / (1 + np.exp(-(W_BLEND[0] * _logit(g) + W_BLEND[1] * _logit(np.nan_to_num(r, nan=0.5)) + W_BLEND[2])))
    unmoved = np.abs(p0 - bl) <= 1e-4
    added = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
    veto = added & (g > LO) & (g < HI) & unmoved & (rn < c.get("veto", 0.2))    # build48.veto_mask
    _, free = p42_base(s1, is2, q, amq, sib, 0.99)
    return {"q": q, "am": amq, "st": st, "ref": st["hn"] & ~veto, "veto": veto, "free": free, "unmoved": unmoved, "p0": p0}


def p55_frame(f: dict, c: dict, fr42: dict, s1, sx, is2, g, r, n) -> dict:
    """P55's combiner scores on P42's frame (p55.france_frame + bag_score): rows = the frame's argmax pairs, cf, and per
    row ref / own (p not moved by a tier or bxneg rule) / region / P42 stage code."""
    import lightgbm as lgb
    q, amq, ref, stq = fr42["q"], fr42["am"], fr42["ref"], fr42["st"]
    rl = np.where(np.isnan(r), f["r_lo"], r)
    stage_c = np.select([stq["base"], stq["rescue"], stq["gc"], stq["hn"]], [1, 2, 3, 4], 0)
    rows = np.flatnonzero(amq)
    X = p55_feats(s1[rows], is2[rows], g[rows], rl[rows], q[rows], hn_rel(f["h1"], f["hx"])[rows], ref[rows], stage_c[rows],
                  sx_feats(sx, g)[rows], n)
    cf = p55_score([lgb.Booster(model_file=str(x)) for x in c["p55"][2]], X)
    p0f, gf = fr42["p0"].astype(np.float32), g.astype(np.float32)
    own = ~(((p0f == np.float32(1.0)) & (g < 0.99995)) | ((p0f == np.float32(0.99)) & (gf != np.float32(0.99))))[rows]
    return {"rows": rows, "cf": cf, "ref": ref[rows], "own": own, "rg": X.region.to_numpy(), "stage": stage_c[rows]}


def p55_veto(f: dict, c: dict, skip=(2,)) -> np.ndarray:
    """P50's p55n list as a pair mask: P55's band veto (ref & own & c < v & band) minus the P42 stages in skip (2 =
    rescue: those are the LB-proven #34v rescues), on P42's frame of the bge chain (f = its post-rules p / p0 / blend)."""
    import stack as S
    s1, sx, is2 = f["q1"].astype(np.int64), f["qx"].astype(np.int64), f["is2"]
    g, r, n = f["g"].astype(np.float64), f["r"].astype(np.float64), f["n"]
    sib, hn_eq = S.sib_mask(f["h1"], f["hx"]), (f["h1"] != "") & (f["h1"] == f["hx"])
    fr42 = p42_frame(f, c, s1, sx, is2, sib, hn_eq, g, r, n)
    z = p55_frame(f, c, fr42, s1, sx, is2, g, r, n)
    m = z["ref"] & z["own"] & (z["cf"] < c["p55"][0]) & (z["rg"] == 1) & ~np.isin(z["stage"], skip)
    out = np.zeros(len(g), bool)
    out[z["rows"][m]] = True
    print(f"{C}: P55 band veto (P42 stages {tuple(skip)} kept) {int(out.sum()):,} pairs", flush=True)
    return out


ORDER_CLEAN = ("veto", "brand", "u", "cls", "p55")                  # build38.clean + P55 (sub_clean)
ORDER_54 = ("brand", "xf", "swap", "veto54", "cls", "u54")          # the #43 -> #48 -> #52 -> #54 lineage


def chain(f: dict, c: dict) -> np.ndarray:
    """The unseen country's final decision (#sub_clean; #54 with c["order"] = ORDER_54) on its post-rules pairs.
    f (row-aligned arrays): q1 / qx codes, is2, p = post-rules p (veto tau 0.1), p0 = post-rules p without the veto,
    g = c_hn's p, pbl = the bge blend before the post-rules, r = CE r (P30's pool first, then the band cache; NaN = none),
    r_lo = P48's CE r of the other pairs, h1 / hx house numbers (fixed SX side), s1 / sx ids, n = #S1, v = the fr31
    world, learn = P3's blocked list (swap only). c: t (decision), anchor (post-rules t), rescue / gc / hn (P42), then the
    stages of c["order"] (default ORDER_CLEAN), each its probe's rule:
      P42 recipe stages on p (base t, rescue, gc, hn-equal extras) -> lines, always first
      brand  P41 brand pairs at c["brand"], computed on the gc70-style lists (anchor t, rescue 0.90, gc 0.70), unioned
      xf     P40 address-less exact names at c["xf"], computed on the gc70-style lists + brand, unioned
      swap   P46 noise-word swap tier (lens s <= c["swap"], flips at the anchor), merged: free SX, caps
      veto   P48: the recipe's rescue / gc stage pairs in the band whose p no rule moved and whose CE r < c["veto"]
      veto54 the same veto as #48 was built (build48.veto_mask on P42's frame: p0 on the blend of g and r)
      u      P54 in-band CE rescue: free argmax non-sibling band pairs (p unmoved), SX in no line, r >= c["u"], caps
      u54    the same rescue as fr54 built #54: on P42's frame, pairs outside its recipe reference
      cls    P47 class: free argmax pairs with an hn change of class -1/-2 or other non-sibling, p >= c["cls"], caps
      p55    P55 combiner c: drop recipe band pairs (own p) with c < v; add free own argmax pairs with c > u, caps
    Returns the accepted mask (every pair an argmax pair of its SX)."""
    import lightgbm as lgb
    import stack as S
    s1, sx, is2 = f["q1"].astype(np.int64), f["qx"].astype(np.int64), f["is2"]
    p, g, n = f["p"].astype(np.float64), f["g"].astype(np.float64), f["n"]
    h1, hx = f["h1"], f["hx"]
    r = f["r"].astype(np.float64)
    am = argmax(sx, p)
    sib, hn_eq = S.sib_mask(h1, hx), (h1 != "") & (h1 == hx)
    say = lambda name, m: print(f"{C}: {name} {int(m.sum()):,} pairs", flush=True)
    st = p42_decide(s1, is2, p, am, sib, hn_eq, n, c["t"], c["rescue"], c["gc"], c["hn"])
    say("recipe (P42)", st["hn"])
    lines = st["hn"].copy()
    band = (g > LO) & (g < HI)
    unmoved = np.abs(p - f["pbl"].astype(np.float64)) <= 1e-4
    _, free = p42_base(s1, is2, p, am, sib, c["t"])
    taken = lambda: np.bincount(sx[lines], minlength=int(sx.max()) + 1)[sx] > 0
    ones = np.ones(len(p), bool)
    gc70 = gib = fr42 = None
    for stage in c.get("order", ORDER_CLEAN):
        if stage in ("brand", "xf") and gc70 is None:   # the #37_gc70-style lists P41 / P40 were staged on
            gc70 = am & (p >= c["anchor"])
            gc70 |= rescue(s1, p, am, gc70, sib, 0.90)
            gc70 |= gc_add(s1, sx, is2, p, am, gc70, sib, 0.70)
        if stage in ("u54", "p55", "veto54") and fr42 is None:
            fr42 = p42_frame(f, c, s1, sx, is2, sib, hn_eq, g, r, n)
        if stage == "brand":
            gib = gc_add(s1, sx, is2, p, am, gc70, sib | ~brand_mask(f["s1"], f["sx"], f["v"]), c["brand"], ones)
            say("brand (P41) added", gib & ~lines)
            lines |= gib
        elif stage == "xf":
            base = gc70 | (gib if gib is not None else False)
            xf = gc_add(s1, sx, is2, p, am, base, sib | ~noaddr_mask(f["s1"], f["sx"]), c["xf"], ones, empty=True)
            say("address-less exact names (P40) added", xf & ~lines)
            lines |= xf
        elif stage == "swap":
            fl = swap_mask(f["s1"], f["sx"], p, c["anchor"], f["learn"], c["swap"])
            m = gc_add(s1, sx, is2, p, am, lines, sib | ~fl, 0.0, ones, empty=True)
            say("noise-word swaps (P46) added", m)
            lines |= m
        elif stage == "veto":
            added = (st["rescue"] & ~st["base"]) | (st["gc"] & ~st["rescue"])
            veto = added & band & unmoved & (r < c["veto"])
            say("CE veto (P48)", veto & lines)
            lines &= ~veto
        elif stage == "veto54":
            say("CE veto (P48, as #48)", fr42["veto"] & lines)
            lines &= ~fr42["veto"]
        elif stage == "u":
            rr = np.nan_to_num(r, nan=0.0)
            m = fill_add(s1, is2, rr, lines, np.flatnonzero(free & ~lines & band & unmoved & ~taken() & (rr >= c["u"])), n)
            say("in-band CE rescue (P54)", m)
            lines |= m
        elif stage == "u54":
            rr = np.nan_to_num(r, nan=0.0)
            cand = fr42["free"] & ~fr42["ref"] & band & fr42["unmoved"] & ~taken() & (rr >= c["u"])
            m = fill_add(s1, is2, rr, lines, np.flatnonzero(cand), n)
            say("in-band CE rescue (P54, fr54)", m)
            lines |= m
        elif stage == "cls":
            cand = np.flatnonzero(free & ~lines & ~taken() & np.isin(hn_rel(h1, hx), HN_OK) & (p >= c["cls"]))
            m = fill_add(s1, is2, p, lines, cand, n)
            say("hn-change class (P47) added", m)
            lines |= m
        elif stage == "p55":
            v55, u55, _ = c["p55"]
            q, ref = fr42["q"], fr42["ref"]
            z = p55_frame(f, c, fr42, s1, sx, is2, g, r, n)
            rows, cf, own, rg = z["rows"], z["cf"], z["own"], z["rg"]
            rm = np.zeros(len(p), bool)
            rm[rows[ref[rows] & own & (cf < v55) & (rg == 1)]] = True
            used = np.bincount(sx[lines], minlength=int(sx.max()) + 1) > 0      # SX of any line before the veto
            lines &= ~rm
            cnt = np.stack(counts(s1, is2, lines, n), 1)
            sup = ~((rg == 2) & (q[rows] > 0))
            cand = np.flatnonzero(fr42["free"][rows] & ~ref[rows] & own & sup & (cf > u55))
            add = np.zeros(len(p), bool)
            for i in rows[cand[np.argsort(-cf[cand], kind="stable")]]:
                j = 0 if is2[i] else 1
                if used[sx[i]] or cnt[s1[i], j] >= CAPS[j]:
                    continue
                used[sx[i]] = True
                cnt[s1[i], j] += 1
                add[i] = True
            lines |= add
            say("combiner (P55) vetoed", rm)
            say("combiner (P55) added", add)
        else:
            raise ValueError(stage)
    return lines


# ---------------------------------------------------------------- P62: the acronym block + rule (sub_iu54_best)
STREETW = set("rue r avenue av ave bd boulevard blvd chemin ch impasse imp allee place pl quai cours route rte square sq "
              "passage residence res lotissement lot street st road rd drive dr lane ln court ct circle cir way trail trl "
              "nagar marg floor flr no plot near opp".split())      # domain: hand-written street-type words (en / fr / in)
ARTW = set("de la du des le les l d the of and".split())


def acro_pairs(jac: float = 0.6, st_min: float = 1.0) -> pd.DataFrame:
    """P62's acronym block: an SX whose core is one token of <= 8 characters equal to the acronym of the S1's core
    (normalize.acronym, >= 2 tokens), with the same house number and city (rules.feats), is a candidate pair; the rule
    keeps it when the two P3 addresses' token-set Jaccard >= jac and every S1 street token (address minus city, state,
    street-type / article words and numbers) is in the SX address. -> the unambiguous pairs (one S1 per SX): s1, sx.
    Train precision 0.997 (India) / 0.9997 (US)."""
    import glob
    import block as B
    s1 = pq.read_table(WORK / f"rf_test_s1_{C}.parquet", columns=["id", "core", "hn", "city"]).to_pandas()
    s1["acro"] = N.acronym(s1.core)
    s1 = s1[(s1.acro != "") & (s1.hn != "") & (s1.city != "")].drop(columns="core")
    fx, out = pq.ParquetFile(WORK / f"rf_test_sx_{C}.parquet"), []
    for i in range(fx.num_row_groups):
        t = fx.read_row_group(i, columns=["id", "core", "hn", "city"])
        out.append(t.filter(pc.and_(pc.less_equal(pc.utf8_length(t["core"]), 8),
                                    pc.invert(pc.match_substring(t["core"], " ")))).to_pandas())
    sx = pd.concat(out)
    sx = sx[(sx.hn != "") & (sx.city != "")]
    j = s1.merge(sx, left_on=["acro", "hn", "city"], right_on=["core", "hn", "city"])[["id_x", "id_y"]]
    j.columns = ["s1", "sx"]

    def grab(files, ids):
        v = pa.array(sorted(set(ids)), pa.string())
        t = [pq.read_table(x, columns=["entity_id", "addr"]) for x in files]
        t = pa.concat_tables([x.filter(pc.is_in(x["entity_id"], value_set=v)) for x in t]).to_pandas()
        return t.drop_duplicates("entity_id").set_index("entity_id").addr
    a1 = grab([B.TXT["test"] / f"{C}_s1.parquet"], j.s1).reindex(j.s1).fillna("").to_numpy(object)
    ax = grab(sorted(glob.glob(str(B.TXT["test"] / f"{C}_sx_*.parquet"))), j.sx).reindex(j.sx).fillna("").to_numpy(object)
    rf = pq.read_table(WORK / f"rf_test_s1_{C}.parquet", columns=["id", "city", "state"]).to_pandas().set_index("id").reindex(j.s1)
    ja, st = np.zeros(len(j), np.float32), np.full(len(j), np.nan, np.float32)
    for i, (u, w, ci, sa) in enumerate(zip(a1, ax, rf.city.fillna("").to_numpy(object), rf.state.fillna("").to_numpy(object))):
        a, b = set(u.split()), set(w.split())
        ja[i] = len(a & b) / len(a | b) if a | b else 0.0
        k = {x for x in a - set(ci.split()) - set(sa.split()) - STREETW - ARTW if not x.isdigit() and len(x) > 1}
        if k:
            st[i] = len(k & b) / len(k)
    j = j[(ja >= jac) & (st >= st_min)].copy()
    j["n_s1"] = j.groupby("sx").s1.transform("size")
    print(f"{C}: acronym block {len(sx):,} one-token SX, rule pairs {len(j):,}, unambiguous {int((j.n_s1 == 1).sum()):,}",
          flush=True)
    return j[j.n_s1 == 1][["s1", "sx"]].reset_index(drop=True)


def acro_tiers(u: pd.DataFrame, ref: pd.DataFrame, ref_cand: pd.DataFrame, f8: pd.DataFrame) -> pd.DataFrame:
    """P62's change set, referenced as it was staged (acro_fr.py + acro_cs.py): the rule pairs that a reference France
    decision `ref` (s1, sx; P62 used #clean's lines) neither accepts nor gives the SX to another S1; tier T1 = not a
    candidate of `ref_cand`, T2 = #8c's argmax (f8 = #8c's France frame: s1, sx, p), T3 = the rest; dropped when another S1
    of the SX has #8c p >= 0.5, or past the caps on ref's lines (highest #8c p first). -> s1, sx, tier."""
    key = lambda d: pd.MultiIndex.from_arrays([d.s1, d.sx])
    u = u.copy()
    u["matched"] = key(u).isin(key(ref))
    owner = ref.drop_duplicates("sx").set_index("sx").s1
    u["taken"] = u.sx.map(owner).notna() & ~u.matched
    u["in_cand"] = key(u).isin(key(ref_cand))
    b = f8[f8.sx.isin(set(u.sx))]
    am = b.sort_values("p", ascending=False).drop_duplicates("sx").set_index("sx").s1
    u["argmax"] = u.sx.map(am).to_numpy() == u.s1.to_numpy()
    u["p8c"] = pd.Series(b.set_index(["s1", "sx"]).p).reindex(key(u)).to_numpy()
    f = u[~u.matched & ~u.taken].copy()
    f["tier"] = np.select([~f.in_cand, f.argmax], ["T1", "T2"], "T3")
    b = f8[f8.sx.isin(set(f.sx))]
    o = b.merge(f[["s1", "sx"]].assign(rule=True), on=["s1", "sx"], how="left")
    f["comp_p"] = o[o.rule.isna()].groupby("sx").p.max().reindex(f.sx).fillna(0).to_numpy()
    f = f[f.comp_p < 0.5].copy()
    f["s3"] = f.sx.str.startswith("S3-")
    cur = ref.assign(s3=ref.sx.str.startswith("S3-")).groupby(["s1", "s3"]).size()
    f["cur"] = cur.reindex(pd.MultiIndex.from_arrays([f.s1, f.s3])).fillna(0).to_numpy()
    f = f.sort_values("p8c", ascending=False, na_position="last")
    f["k"] = f.groupby(["s1", "s3"]).cumcount() + 1
    f = f[f.cur + f.k <= np.where(f.s3, 6, 5)]
    print(f"{C}: acronym change set {len(f):,} pairs {f.tier.value_counts().to_dict()}", flush=True)
    return f[["s1", "sx", "tier"]].reset_index(drop=True)


def acro_fit(t: pd.DataFrame, lines: pd.DataFrame) -> list:
    """P50's acro_fit on a base decision (lines: s1, sx of the country's accepted pairs): tiers T2, T1, T3 in turn (then
    s1, sx); a pair is kept when its SX is in no line and its S1 stays within 5 S2 / 6 S3. -> [(s1, sx)] in that order."""
    used = set(lines.sx)
    cnt = {}
    for a, x in zip(lines.s1, lines.sx):
        c = cnt.setdefault(a, [0, 0])
        c[0 if x.startswith("S2-") else 1] += 1
    keep = []
    for r in t.assign(o=t.tier.map({"T2": 0, "T1": 1, "T3": 2})).sort_values(["o", "s1", "sx"]).itertuples():
        if r.sx in used:
            continue
        j = 0 if r.sx.startswith("S2-") else 1
        c = cnt.setdefault(r.s1, [0, 0])
        if c[j] >= CAPS[j]:
            continue
        c[j] += 1
        used.add(r.sx)
        keep.append((r.s1, r.sx))
    print(f"{C}: acronym adds kept on the base {len(keep):,}", flush=True)
    return keep


# ---------------------------------------------------------------- P64: legal-form conflict veto (the last France stage)
LF_CANON = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt",
            "etablissement": "ets", "etablissements": "ets"}
LF_LONG = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co"}


def _lf_canon(t: str) -> str:
    from rapidfuzz.fuzz import ratio
    if t in N.LEGAL:
        return LF_CANON.get(t, t)
    return LF_LONG[max(LF_LONG, key=lambda w: ratio(t, w))]      # a typo of a long form ('pirvate', 'Limted')


def _lf_sig(x: str) -> str:
    toks = x.split()
    leg = [t for i, t in enumerate(toks) if not (i == 0 and t == "co" and len(toks) > 1)      # 'Co-op ...', 'Co Name'
           and N._is_legal(t, toks[i + 1] if i + 1 < len(toks) else "")]
    if len(leg) == len(toks):          # only legal tokens: an acronym ('PC' = the initials of a 2-word core) or a bare form
        return ""
    return "+".join(sorted({_lf_canon(t) for t in leg}))


def lf_sig(names) -> np.ndarray:
    """P64: the legal-form signature of names ('Acme Pvt. Limited' -> 'ltd+pvt', 'Bar SARL' -> 'sarl', none -> ''), with
    normalize's own detector (LEGAL + typos of the long forms); computed once per distinct name."""
    u, inv = np.unique(np.asarray(names).astype(str), return_inverse=True)
    return np.array([_lf_sig(x) for x in N.name_norm(pd.Series(u))], object)[inv]


def lf_class(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0 both none, 1 same, 2 S1 only, 3 SX only, 4 both present and disjoint (the conflict), 5 overlap."""
    ea, eb = a == "", b == ""
    ov = np.array([bool(set(u.split("+")) & set(v.split("+"))) for u, v in zip(a, b)], bool)
    return np.select([ea & eb, a == b, eb, ea, ov], [0, 1, 2, 3, 5], 4).astype(np.int8)


def text_names(q1: np.ndarray, qx: np.ndarray) -> tuple:
    """(S1 names at q1, SX names at qx) from P3's cached text tables (the rows q1 / qx index)."""
    import block as B
    import features as F
    n1 = pq.read_table(B.TXT["test"] / f"{C}_s1.parquet", columns=["name"]).column(0).combine_chunks() \
        .take(pa.array(q1)).to_numpy(zero_copy_only=False)
    files, offs = F.sx_files("test", C)
    nx = np.empty(len(qx), object)
    for fx, lo, hi in zip(files, offs[:-1], offs[1:]):
        m = (qx >= lo) & (qx < hi)
        if m.any():
            nx[m] = pq.read_table(fx, columns=["name"]).column(0).combine_chunks().take(pa.array(qx[m] - lo)) \
                .to_numpy(zero_copy_only=False)
    return n1, nx


def lf_veto(q1: np.ndarray, qx: np.ndarray) -> np.ndarray:
    """P64's veto per accepted pair (q1 / qx rows): S1 and SX both carry a legal form and the forms are disjoint
    (SARL vs SAS, SAS vs SA, SCI vs SNC). LOCO (gc70, chain) +0.00300 / +0.00074; the unseen country only (India / US
    class-4 pairs are 0.89 / 0.99 precise: the combiner already prices them)."""
    n1, nx = text_names(q1, qx)
    k = lf_class(lf_sig(n1), lf_sig(nx))
    print(f"{C}: legal-form conflict veto (P64) {int((k == 4).sum()):,} of {len(k):,} accepted pairs", flush=True)
    return k == 4


# ---------------------------------------------------------------- P66: address-less exact full names with a unique core
NULLS = {"", "NULL", "<NULL>"}


def xfc(n1, nx, ax, c1, full_count, core_count) -> np.ndarray:
    """P66 rule66.xfc: the SX's address is empty (or NULL), its full normalised name (fkey, legal form kept) equals the
    S1's, a name >= 5 characters no other S1 of the country carries, and the S1's name core is unique in the country
    (the core guard drops xf_noaddr's false adds: 97% belonged to another S1 with the same core, 'X SARL' vs 'X')."""
    return np.array([(adr or "").strip().upper() in NULLS and a == b and len(a) >= 5 and full_count.get(a, 0) == 1
                     and core_count.get(c, 0) == 1
                     for a, b, adr, c in zip(map(fkey, n1), map(fkey, nx), ax, c1)], bool)


def raw_rows(ids, sources: tuple, cols: list) -> pd.DataFrame:
    """Raw test rows of ids (entity_id index)."""
    v = pa.array(sorted(set(ids)), pa.large_string())
    t = pd.concat([pq.read_table(WORK / f"test_source{i}.parquet", columns=["entity_id", *cols],
                                 filters=[("entity_id", "in", v)]).to_pandas() for i in sources])
    return t.drop_duplicates("entity_id").set_index("entity_id")


def xfc_adds(q1, qx, is2, p, am, base, sib, s1, sx, t: float = 0.3) -> np.ndarray:
    """P66's adds on a #37-style frame (p, argmax am, base = its accepted lines): xfc pairs among the address-less argmax
    pairs, then rule40.add (free SX, non-sibling, p >= t, caps 5 / 6, best p first, any S1). -> mask."""
    raw = lambda k, cols: pq.read_table(WORK / f"test_source{k}.parquet", columns=cols,
                                        filters=[("country", "==", C)]).to_pandas()
    a = pd.concat([raw(k, ["entity_id", "business_address"]) for k in (2, 3)], ignore_index=True)
    na = set(a.entity_id[a.business_address.str.strip().str.upper().isin(NULLS)])
    del a
    w = np.flatnonzero(am & pd.Series(sx).isin(na).to_numpy())       # np.isin loops per value on object arrays
    full = pd.Series([fkey(x) for x in raw(1, ["business_name"]).business_name]).value_counts().to_dict()
    core = pd.read_parquet(WORK / f"rf_test_s1_{C}.parquet", columns=["id", "core"]).set_index("id").core
    n1 = raw_rows(s1[w], (1,), ["business_name"]).business_name.reindex(s1[w]).fillna("").to_numpy(object)
    rx = raw_rows(sx[w], (2, 3), ["business_name", "business_address"]).reindex(sx[w])
    cls = np.zeros(len(p), bool)
    cls[w] = xfc(n1, rx.business_name.fillna("").to_numpy(object), rx.business_address.fillna("").to_numpy(object),
                 core.reindex(s1[w]).fillna("").to_numpy(object), full, core.value_counts().to_dict())
    add = gc_add(q1, qx, is2, p, am, base, sib | ~cls, t, np.ones(len(p), bool), empty=True)
    print(f"{C}: address-less exact names with a unique core (P66): {len(w):,} address-less argmax pairs, "
          f"class {int(cls.sum()):,}, adds {int(add.sum()):,}", flush=True)
    return add


def argmax(qx, p):
    import model as M
    return M.argmax_sx(qx, p)


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
    # rescue / completion toy: S1 0 has an S3 only (row 0), S1 1 has S2 + S3; rows 3..6 are free argmax candidates
    q1, qx, is2 = np.array([0, 1, 1, 0, 0, 1, 0]), np.arange(7), np.array([0, 1, 0, 1, 1, 1, 0], bool)
    p, am, acc = np.array([.99, .99, .99, .92, .95, .91, .93]), np.ones(7, bool), np.r_[[True] * 3, [False] * 4]
    no = np.zeros(7, bool)
    assert np.flatnonzero(gc_add(q1, qx, is2, p, am, acc, no, .9)).tolist() == [3, 4]        # S1 0's missing S2s
    assert np.flatnonzero(gc_add(q1, qx, is2, p, am, acc, no, .94)).tolist() == [4]
    assert np.flatnonzero(gc_add(q1, qx, is2, p, am, acc, no, .9, np.r_[no[:6], True])).tolist() == [3, 4, 6]
    assert np.flatnonzero(gc_add(q1, qx, is2, p, am, acc, p == .95, .9)).tolist() == [3]    # a sibling never joins
    assert not gc_add(q1, qx, is2, p, am, acc | (qx == 3), no, .9)[4]                       # S1 0 is two-sided now
    assert not rescue(q1, p, am, acc, no, .9).any()                                           # no empty S1
    assert np.flatnonzero(rescue(q1, p, am, np.r_[no[:1], acc[1:]], no, .9)).tolist() == [0]  # S1 0 empty: its best
    z, one = np.zeros(8, int), np.r_[True, np.zeros(7, bool)]                                # CAP: 5 S2 per S1
    assert np.flatnonzero(gc_add(z, np.arange(8), ~one, 1 - np.arange(8) / 100, np.ones(8, bool), one, ~np.ones(8, bool),
                                 .9)).tolist() == [1, 2, 3, 4, 5]
    ms = {"zeph", "calo", "nex", "brix", "pyra", "lyra", "kelo", "iri", "quo", "sol"}
    assert [n_syll(t, ms) >= 2 for n in ["Zephcalonex", "QÚOLYRASOL", "Ke1ozeph", "Zeph", "Solar"] for t in btoks(n)] == \
        [True, True, True, False, False]
    assert btoks("M/s Brixpyra Pvt") == ["m", "s", "brixpyra", "pvt"] and n_syll("zephcalobrix", {"zeph", "calo", "brix"}) == 3
    assert fkey("Club des Africa & Fils SARL") == "club des africa and fils sarl" and fkey("Maison Léts") == "maison lets"
    t = lf_sig(["Acme Pvt. Limited", "Foo L.L.C.", "Bar SARL", "Baz", "Qux Pirvate Ltd", "Co-op Stores Inc.",
                "Société Dupont S.A.S.", "Company", "PC", "S A S U Dupont"]).tolist()
    assert t == ["ltd+pvt", "llc", "sarl", "", "ltd+pvt", "inc", "sas", "", "", "sasu"], t
    assert lf_class(np.array(["", "llc", "llc", "", "llc", "ltd+pvt"], object),
                    np.array(["", "llc", "", "inc", "inc", "ltd"], object)).tolist() == [0, 1, 2, 3, 4, 5]
    got = xfc(["Club des Africa SARL", "Club des Africa SARL", "Maison Lets", "Club des Africa SARL", "Dupont"],
              ["CLUB DES AFRICA SARL", "Club des Africa SARL", "Maison Léts", "Club des Africa SARL", "Dupont"],
              ["", "1 Rue X", "NULL", "<NULL>", ""], ["club africa", "club africa", "maison lets", "club africa", "dupont"],
              {"club des africa sarl": 1, "maison lets": 1, "dupont": 1}, {"club africa": 1, "maison lets": 2, "dupont": 1})
    assert list(got) == [True, False, False, True, True], got
    e = np.r_[False, acc[1:]]                                                                 # xf: an empty S1 0 may take it
    assert complete(q1, qx, is2, p, am, e, np.array([""] * 7, object), np.array([""] * 7, object),
                    noaddr=(.9, np.r_[no[:6], True])).tolist() == e.tolist()[:6] + [True]
    g = np.r_[no[:6], True]                                                                   # brand: only row 6 may join
    assert complete(q1, qx, is2, p, am, acc, h := np.array([""] * 7, object), h, brand=(.9, g)).tolist() == \
        acc.tolist()[:6] + [True]
    h = np.array(["1", "", "", "7", "2", "", ""], object)
    assert complete(q1, qx, is2, p, am, acc, h, h, .9, .9, .9).tolist() == [True] * 3 + [True, True, False, False]
    print("france self-check ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["chn"]:                        # run.france31's subprocess: france.py chn MODEL OUT COUNTRY
        C = sys.argv[4]
        chn(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        _check()
