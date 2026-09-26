"""P28 address-anchor rule (the user's idea): two different businesses rarely share an exact address, so when the names
differ by ONE core word but the full address is the same, trust the address and accept the pair.

Rule (label-free, record fields only; rules.feats cores / hn / places + a street key parsed here):
  name   normalize.name_core cores differ by exactly one token: swap (P20 'sub1'), insert ('add1': also the R1 filler
         case '& Fils' / '& Cie'), delete ('drop1')
  hn     house number present on both and EQUAL (so never a generator sibling offset hn + {1..21})
  city   the records share a place segment (rules.addr_score's place overlap `po`)
  street (b, c) the street of the hn segment is the same after normalisation (street_key; fuzzy: ratio >= 85 on the
         sorted-token keys, typos are the generator's main street noise)
  chain  (c) no other S1 of the pool has the same (city_key, hn, street_key): an address shared by S1 = chain / mall, skip
  when   the pair is its SX's argmax, or the SX is matched nowhere (its best p < t) so the boost cannot steal it
  then   p := max(p, BOOST) before the argmax + threshold decision (BOOST > t: accepted regardless of t)
Variants: a = name + hn + city; b = a + street (fuzzy); bx = a + street exact; c = b + chain guard;
bxn / cn = bx / c without the swap kind (insert / delete only), cxn = bxn + chain guard; bxs = bx swaps only. Extensions beyond one word (same
address, name kind as P20's name_rel): bxe = core equal, bxt = typo only, bxnet = insert / delete / equal / typo,
bxne = insert / delete / equal (cxne + chain guard). g = + sibling-word guard (sib_words): bxng, bxsg (swaps),
bxneg, bxnesg (insert / delete / equal / swap).
    .venv/bin/python probes/P28-addr-anchor/anchor.py            # self-check
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
import model as M
import normalize as N
from patterns import name_rel

W = ROOT / "data/work"
SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]           # the generator's sibling hn offsets (P12, P15)
ONE = {"sub1", "add1", "drop1"}
KINDS = ONE | {"eq", "typo"}
BOOST = 0.99
VARIANTS = ("a", "b", "bx", "c", "bxn", "cn", "cxn", "bxs", "bxe", "bxt", "bxnet", "bxne", "cxne", "bxng", "bxsg", "bxneg", "bxnesg", "cxneg")
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
    """Street of the house-number segment, canonical (N._seg: street types, ordinals, filler; articles / bis / ter
    dropped), tokens sorted; '' when the address has no house number."""
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_street).astype("str")


def _rf(path: Path, ids: np.ndarray, cols: list) -> pd.DataFrame:
    t = pq.read_table(path, columns=["id", *cols])
    t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
    return t.to_pandas().set_index("id").reindex(ids)


def _addr(files: list, ids: np.ndarray) -> pd.Series:
    out = []
    for f in files:
        t = pq.read_table(f, columns=["entity_id", "business_address"])
        t = t.filter(pc.is_in(t.column("entity_id"), value_set=pa.array(ids, pa.large_string())))
        out.append(t.to_pandas())
    a = pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").business_address
    return a.reindex(ids).fillna("")


def sib_words(key: np.ndarray, sib: np.ndarray, support: int = 20, share: float = 0.5) -> set:
    """Label-free sibling-word table, learned per country from its own candidate pairs: 'kind:word' (the word a swap /
    insert brings in) whose hn-equal-or-offset pairs take a generator sibling offset (SIB3) at least `share` of the time,
    with >= `support` such pairs ('X Holding' at hn + 5). Pure branch words sit near 0.95 in US-like data (siblings keep
    the hn 3%), ~0.75 in India (26%); true-pair noise words near 0.05."""
    g = pd.DataFrame({"k": key, "s": sib})
    g = g[g.k.str.startswith(("add1:", "sub1:"))].groupby("k").s.agg(["size", "mean"])
    return set(g.index[(g["size"] >= support) & (g["mean"] >= share)])


def sib_table(pairs: pd.DataFrame, xs: str, c: str) -> set:
    """sib_words learned from any candidate pair set (s1, sx ids), e.g. the blocked list before the pruner removes most
    sibling-offset pairs (#8c's pruned India holdout list leaves 1 word to learn; its blocked list many)."""
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", pd.unique(pairs.s1), ["core", "hn"])
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", pd.unique(pairs.sx), ["core", "hn", "is_web"])
    h1, hx = r1.hn.reindex(pairs.s1).fillna("").to_numpy(), rx.hn.reindex(pairs.sx).fillna("").to_numpy()
    both = np.flatnonzero((h1 != "") & (hx != ""))
    dl = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    rel = both[(dl == 0) | np.isin(dl, SIB3)]
    dl = dl[(dl == 0) | np.isin(dl, SIB3)]
    s1r, sxr = pairs.s1.to_numpy()[rel], pairs.sx.to_numpy()[rel]
    c1, cx = r1.core.reindex(s1r).fillna("").to_numpy(), rx.core.reindex(sxr).fillna("").to_numpy()
    wx = rx.is_web.reindex(sxr).fillna(False).to_numpy()
    key = np.array([f"{k}:{w}" for k, w in (name_rel(a, b, x) for a, b, x in zip(c1, cx, wx))], object)
    return sib_words(key, np.isin(dl, SIB3))


def rule_masks(d: pd.DataFrame, split: str, c: str, t: float, am=None, best=None, learn=None) -> dict:
    """d: s1, sx, p (every scored pair of the SX involved; else pass am = argmax flag and best = the SX's max p over
    all its S1); learn: optional (s1, sx) frame to learn sib_words from instead of d. Returns {variant: bool mask of
    rule pairs} + 'kind' / 'info' / 'sibw'."""
    xs = "train" if split == "holdout" else "test"
    u1, ux = pd.unique(d.s1), pd.unique(d.sx)
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", u1, ["core", "hn", "places", "city"])
    h1 = r1.hn.reindex(d.s1).fillna("").to_numpy()
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", ux, ["core", "hn", "places", "is_web"])
    hx = rx.hn.reindex(d.sx).fillna("").to_numpy()
    qx = pd.factorize(d.sx)[0]
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(qx, p) if am is None else am
    best = pd.Series(p).groupby(qx).transform("max").to_numpy() if best is None else best
    both = (h1 != "") & (hx != "")
    dlt = np.full(len(d), np.nan)
    dlt[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    rel = np.flatnonzero((dlt == 0) | np.isin(dlt, SIB3))            # hn equal or a generator sibling offset
    c1 = r1.core.reindex(d.s1.to_numpy()[rel]).fillna("").to_numpy()
    cx = rx.core.reindex(d.sx.to_numpy()[rel]).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx.to_numpy()[rel]).fillna(False).to_numpy()
    nr = [name_rel(a, b, w) for a, b, w in zip(c1, cx, wx)]
    kr = np.array([k for k, _ in nr], object)
    key = np.array([f"{k}:{w}" for k, w in nr], object)
    sibw = sib_words(key, np.isin(dlt[rel], SIB3)) if learn is None else sib_table(learn, xs, c)
    ok = (dlt[rel] == 0) & (am[rel] | (best[rel] < t))                  # hn equal, and the boost cannot steal
    n_elig = int(ok.sum())
    ok &= np.isin(kr, list(KINDS))
    cand, kind, guard = rel[ok], kr[ok], np.isin(key[ok], list(sibw))
    del c1, cx, wx, nr, kr, key
    s1c, sxc = d.s1.to_numpy()[cand], d.sx.to_numpy()[cand]
    p1, px = r1.places.reindex(s1c).fillna("").to_numpy(), rx.places.reindex(sxc).fillna("").to_numpy()
    city = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)])
    del r1, rx
    src = [W / f"{xs}_source{i}.parquet" for i in (1, 2, 3)]
    k1 = street_key(_addr(src[:1], pd.unique(s1c))).reindex(s1c).fillna("").to_numpy()
    kx = street_key(_addr(src[1:], pd.unique(sxc))).reindex(sxc).fillna("").to_numpy()
    st_x = (k1 != "") & (k1 == kx)
    st_f = np.array([a != "" and b != "" and ratio(a, b) >= 85 for a, b in zip(k1, kx)])
    pool = pq.read_table(W / f"rf_{xs}_s1_{c}.parquet", columns=["id", "city", "hn"]).to_pandas()
    pool = pool[(pool.hn != "") & (pool.city != "")]
    ck = pool.set_index("id").reindex(s1c).fillna("")
    ck["street"] = k1
    mates = pool.merge(ck[["city", "hn"]].drop_duplicates(), on=["city", "hn"])   # pool S1 at a candidate's (city, hn)
    mates["street"] = street_key(_addr(src[:1], mates.id.to_numpy())).to_numpy()
    n_addr = mates[mates.street != ""].groupby(["city", "hn", "street"]).size()
    chain = ck.join(n_addr.rename("n"), on=["city", "hn", "street"]).n.fillna(0).to_numpy() >= 2
    del pool, ck, mates, n_addr
    out = {}
    sub, w1 = kind == "sub1", np.isin(kind, list(ONE))
    ins, eq, ty = np.isin(kind, ["add1", "drop1"]), kind == "eq", kind == "typo"
    for v, m in (("a", w1 & city), ("b", w1 & city & st_f), ("bx", w1 & city & st_x), ("c", w1 & city & st_f & ~chain),
                 ("bxn", ins & city & st_x), ("cn", ins & city & st_f & ~chain),
                 ("cxn", ins & city & st_x & ~chain), ("bxs", sub & city & st_x),
                 ("bxe", eq & city & st_x), ("bxt", ty & city & st_x), ("bxnet", (ins | eq | ty) & city & st_x),
                 ("bxne", (ins | eq) & city & st_x), ("cxne", (ins | eq) & city & st_x & ~chain),
                 ("bxng", ins & city & st_x & ~guard), ("bxsg", sub & city & st_x & ~guard),
                 ("bxneg", (ins | eq) & city & st_x & ~guard), ("bxnesg", (ins | eq | sub) & city & st_x & ~guard),
                 ("cxneg", (ins | eq) & city & st_x & ~chain & ~guard)):
        mm = np.zeros(len(d), bool)
        mm[cand[m]] = True
        out[v] = mm
    kk = np.full(len(d), "", object)
    kk[cand] = kind
    out["kind"] = kk
    out["info"] = {"hn_eq_eligible": n_elig, "one_word": int(w1.sum()), "city": int((w1 & city).sum()),
                   "street_fuzzy": int((w1 & city & st_f).sum()), "street_exact": int((w1 & city & st_x).sum()),
                   "chain_guarded": int((w1 & city & st_f & chain).sum()), "eq_typo_addr_exact": int(((eq | ty) & city & st_x).sum()),
                   "sib_words": len(sibw), "guarded": int((city & st_x & guard).sum())}
    out["sibw"] = sibw
    return out


def boosted(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.maximum(p, BOOST), p)


def decide(sx: pd.Series, p: np.ndarray, t: float) -> np.ndarray:
    return M.argmax_sx(pd.factorize(sx)[0], p) & (p >= t)


if __name__ == "__main__":
    f = pd.Series(["12 Rue de la Paix, Lille, Hauts-de-France", "N°12 R. DE LA PAIX, LILLE", "12b rue de la Paix, Lille",
                   "12, Main Street, Austin, TX", "Unit 9, 1108 State Highway, NC", "Lille, Nord", ""])
    k = street_key(f).tolist()
    assert k[0] == k[1] == k[2] == "paix r", k
    assert k[3] == "main st" and k[4] == "hwy state" and k[5] == k[6] == "", k
    assert decide(pd.Series(["x", "x", "y"]), boosted(np.array([.5, .6, .2]), np.array([True, False, True])), .71).tolist() \
        == [True, False, True]
    key = np.array(["add1:holding"] * 30 + ["add1:fils"] * 30 + ["drop1:"] * 30 + ["sub1:rare"] * 5, object)
    sib = np.r_[np.ones(28), np.zeros(2), np.ones(1), np.zeros(29), np.ones(30), np.ones(5)].astype(bool)
    assert sib_words(key, sib) == {"add1:holding"}, sib_words(key, sib)
    print("anchor self-check ok")
