"""P71 A: France noise-word tier on sub66_final, label-free.

The generator copies an S1 into S2/S3 and sometimes inserts a filler word ('& Fils', 'Développement', 'Groupe') or swaps
one S1 word for it, keeping the house number. The India/US-trained CEs read 'group/holdings' as decoy words and reject
these copies. Here:
  1. noise words are learned per country, label-free: SX-vs-S1 document-frequency ratio (test side) plus P46's hn lens
     per kind:word (insert 'ins:w' / swap 'sub:w') over candidate pairs at a shared place: e = share hn-equal, s = share
     at a generator sibling offset (SIB3), mirror = hn-equal count / (r0 * negative-offset count).
  2. add set on a base submission: free SX (no line owns it), identical house number (both present), S1 street tokens in
     the SX street, shared place, name tokens (legal forms dropped) = S1 + noise words (insert) or one S1 word -> one
     noise word (swap); the relation holds for exactly one candidate S1; the SX name is no France S1's own name in that
     city; P64 legal-form conflict veto; caps 5 S2 / 6 S3.

    .venv/bin/python probes/P71-fr-noiseword/p71.py lens            # France lens + add pool -> work/
    .venv/bin/python probes/P71-fr-noiseword/p71.py build OUT WORDS  # sub66_final + adds whose words are in WORDS
"""
import collections
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import france as FR  # noqa: E402
import normalize as N  # noqa: E402

W = ROOT / "data/work"
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
BASE = ROOT / "output/sub66_final"
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
SIB3 = set(FR.SIB3)
CAP = {"S2": 5, "S3": 6}
RATIO_MIN, SUPPORT, E_MIN = 3.0, 20, 0.8
peak = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def toks(names: pd.Series) -> list:
    """name_norm tokens minus legal forms (and their typos), order kept, duplicates dropped."""
    out = []
    for x in N.name_norm(names):
        t = x.split()
        k = [w for i, w in enumerate(t) if not N._is_legal(w, t[i + 1] if i + 1 < len(t) else "")]
        out.append(" ".join(dict.fromkeys(k)))
    return out


def rel(a: str, b: str) -> tuple:
    """(kind, inserted words, dropped words) of SX tokens b vs S1 tokens a (space-joined): eq | ins | sub (one word
    swapped, not a typo) | typo | other | disjoint."""
    from rapidfuzz.fuzz import ratio
    a, b = a.split(), b.split()
    sa, sb = set(a), set(b)
    ins, drop = tuple(w for w in b if w not in sa), tuple(w for w in a if w not in sb)
    if not ins and not drop:
        return "eq", ins, drop
    if not sa & sb:
        return "disjoint", ins, drop
    if not drop:
        return "ins", ins, drop
    if len(ins) == 1 and len(drop) == 1:
        return ("typo" if ratio(ins[0], drop[0]) >= 75 else "sub"), ins, drop
    return "other", ins, drop


def hnum(h: np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(h), errors="coerce").to_numpy(float)


def load(country: str, split: str = "test") -> tuple:
    """S1 / SX frames of one country: id, tok, hn, places, city (S1), is_web (SX), name."""
    rf = lambda k: pd.read_parquet(W / f"rf_{split}_{k}_{country}.parquet",
                                   columns=["id", "hn", "places", "city"] + (["is_web"] if k == "sx" else []))
    raw = lambda i: pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_name"],
                                  filters=[("country", "==", country)]).to_pandas()
    s1 = rf("s1").merge(raw(1).rename(columns={"entity_id": "id", "business_name": "name"}), on="id")
    sx = rf("sx").merge(pd.concat([raw(2), raw(3)]).drop_duplicates("entity_id")
                        .rename(columns={"entity_id": "id", "business_name": "name"}), on="id")
    for d in (s1, sx):
        d["tok"] = toks(d.name)
    return s1.set_index("id"), sx.set_index("id")


def raw(ids, src: tuple, col: str, split: str = "test") -> pd.Series:
    """One raw column of the ids (entity_id index), read with an id filter."""
    v = pa.array(sorted(set(ids)), pa.large_string())
    t = pd.concat([pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", col],
                                 filters=[("entity_id", "in", v)]).to_pandas() for i in src])
    return t.drop_duplicates("entity_id").set_index("entity_id")[col]


def ratios(s1: pd.DataFrame, sx: pd.DataFrame) -> pd.DataFrame:
    """Per token: document frequency in S1 and in S2/S3 and the SX / S1 rate ratio (add-one)."""
    c1 = collections.Counter(w for t in s1.tok for w in t.split())
    cx = collections.Counter(w for t in sx.tok for w in t.split())
    r = pd.DataFrame({"df1": pd.Series(c1), "dfx": pd.Series(cx)}).fillna(0)
    r["ratio"] = ((r.dfx + 1) / len(sx)) / ((r.df1 + 1) / len(s1))
    return r


def base_lines(base: Path, ids: set) -> dict:
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    assert len(sub) == 1_732_544 and sub.source1_entity_id.is_unique
    sub = sub[sub.source1_entity_id.isin(ids)]
    return {s: (m.split(",") if m else []) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)}


def cand_pairs(base: Path, ids: set) -> pd.DataFrame:
    c = pd.read_csv(base / "candidate_pairs.tsv", **RD)
    c = c[c.source1_entity_id.isin(ids) & (c.candidate_entity_ids != "")]
    c = c.assign(sx=c.candidate_entity_ids.str.split(",")).explode("sx")
    return pd.DataFrame({"s1": c.source1_entity_id.to_numpy(), "sx": c.sx.to_numpy()})


def pair_frame(pairs: pd.DataFrame, s1: pd.DataFrame, sx: pd.DataFrame) -> pd.DataFrame:
    """pairs (s1, sx) + kind / ins / drop, D = hx - h1 (NaN when either is missing), shared place."""
    a, b = s1.reindex(pairs.s1), sx.reindex(pairs.sx)
    kw = [rel(x, y) if isinstance(x, str) and isinstance(y, str) else ("none", (), ())
          for x, y in zip(a.tok.to_numpy(), b.tok.to_numpy())]
    d = pairs.reset_index(drop=True).copy()
    d["kind"] = [k for k, _, _ in kw]
    d["ins"] = [" ".join(i) for _, i, _ in kw]
    d["drop"] = [" ".join(j) for _, _, j in kw]
    h1, hx = a.hn.fillna("").to_numpy(), b.hn.fillna("").to_numpy()
    d["h1"], d["hx"] = h1, hx
    d["D"] = np.where((h1 != "") & (hx != ""), hnum(hx) - hnum(h1), np.nan)
    p1, px = a.places.fillna("").to_numpy(), b.places.fillna("").to_numpy()
    d["place"] = [u != "" and bool(set(u.split("|")) & set(v.split("|"))) for u, v in zip(p1, px)]
    d["web"] = b.is_web.fillna(False).to_numpy()
    return d


def lens(d: pd.DataFrame, r0: float) -> pd.DataFrame:
    """Per kind:word over single-word inserts / swaps at a shared place with hn on both sides: n, e (hn equal), s
    (SIB3 offset), neg (negative offset), mirror = r0 * neg / n_eq (true share at D=0 if the word's copies move their hn
    like the base's accepted lines do)."""
    m = d.kind.isin(["ins", "sub"]) & ~d.ins.str.contains(" ") & d.place & d.D.notna() & ~d.web
    g = pd.DataFrame({"k": d.kind[m] + ":" + d.ins[m], "eq": d.D[m] == 0, "sib": d.D[m].isin(SIB3), "neg": d.D[m] < 0})
    t = g.groupby("k").agg(n=("eq", "size"), n_eq=("eq", "sum"), n_sib=("sib", "sum"), n_neg=("neg", "sum"))
    t["e"], t["s"] = t.n_eq / t.n, t.n_sib / t.n
    t["mirror"] = (r0 * t.n_neg / t.n_eq.clip(lower=1)).clip(upper=9.99)
    return t


def pool(d: pd.DataFrame, s1: pd.DataFrame, sx: pd.DataFrame, owned: set, country: str, split: str) -> pd.DataFrame:
    """The add pool before the word list: free SX, ins / sub, hn equal (string, both present), shared place, S1 street in
    the SX street, relation unique among the SX's candidate S1, SX name not a same-city S1's own name; + P64 class."""
    m = d.kind.isin(["ins", "sub"]) & (d.h1 != "") & (d.h1 == d.hx) & d.place & ~d.web & ~d.sx.isin(owned)
    c = d[m].copy()
    c = c[c.groupby("sx").s1.transform("size") == 1]                 # one candidate S1 holds the relation
    c = c[~c.sx.isin(set(d.sx[d.kind.isin(["eq", "typo"])]))]        # and no candidate S1 carries the SX's own name
    srt = lambda t: " ".join(sorted(t.split()))
    mine = set(zip(s1.tok.map(srt), s1.city))
    kx = sx.tok.reindex(c.sx).map(srt).to_numpy()
    c = c[[(k, cc) not in mine for k, cc in zip(kx, s1.city.reindex(c.s1).to_numpy())]]
    FR.C = country
    addr = lambda ids, src: raw(ids, src, "business_address", split).reindex(ids).fillna("")
    k1 = FR.street_key(addr(pd.unique(c.s1), (1,)))
    kx = FR.street_key(addr(pd.unique(c.sx), (2, 3)))
    a, b = k1.reindex(c.s1).fillna("").to_numpy(), kx.reindex(c.sx).fillna("").to_numpy()
    c["street"] = [u != "" and set(u.split()) <= set(v.split()) for u, v in zip(a, b)]
    c["street_eq"] = a == b
    c = c[c.street].copy()
    c["lf"] = FR.lf_class(FR.lf_sig(s1.name.reindex(c.s1).to_numpy()), FR.lf_sig(sx.name.reindex(c.sx).to_numpy()))
    return c.reset_index(drop=True)


def admit(c: pd.DataFrame, words: set) -> np.ndarray:
    """Pool rows whose inserted words are all in `words` (ins) or whose one swapped-in word is (sub), P64 class != 4."""
    ok = np.array([all(f"{k}:{w}" in words for w in i.split()) for k, i in zip(c.kind, c.ins)], bool)
    return ok & (c.lf.to_numpy() != 4)


def merge(pred: dict, add: pd.DataFrame) -> tuple:
    """pred: S1 -> SX list; add rows (s1, sx) in priority order. Free SX only, caps 5 S2 / 6 S3 per S1."""
    owned = {x for v in pred.values() for x in v}
    new, rows = {k: list(v) for k, v in pred.items()}, []
    for s1, sx in add[["s1", "sx"]].itertuples(index=False):
        if sx in owned:
            continue
        cur = new.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]:
            continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx))
    return new, pd.DataFrame(rows, columns=["s1", "sx"])


def write(base: Path, out: Path, new: dict, chg: set):
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in chg:
                g.write(f"{s}\t{','.join(sorted(new[s]))}\n")
                n += 1
            else:
                g.write(line)
    assert n == len(chg)
    if not (out / "candidate_pairs.tsv").exists():
        os.link(base / "candidate_pairs.tsv", out / "candidate_pairs.tsv")


def selftest():
    assert rel("rapiere sportive", "rapiere developpement")[:2] == ("sub", ("developpement",))
    assert rel("martin", "martin et fils") == ("ins", ("et", "fils"), ())
    assert rel("abc club", "abc clubb")[0] == "typo"
    assert toks(pd.Series(["Martin & Fils SARL", "Rapière SAS Développement"])) == ["martin and fils", "rapiere developpement"]
    c = pd.DataFrame({"kind": ["ins", "sub", "ins"], "ins": ["et fils", "groupe", "fils"], "lf": [0, 1, 4]})
    assert admit(c, {"ins:et", "ins:fils", "sub:groupe"}).tolist() == [True, True, False]
    assert admit(c, {"ins:fils"}).tolist() == [False, False, False]
    pred = {"S1-a": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"], "S1-b": []}
    new, rows = merge(pred, pd.DataFrame({"s1": ["S1-a", "S1-a", "S1-b", "S1-b"], "sx": ["S2-6", "S3-7", "S2-5", "S2-8"]}))
    assert new["S1-a"][-1] == "S3-7" and len(new["S1-a"]) == 6 and new["S1-b"] == ["S2-8"] and pred["S1-b"] == []
    print("p71 self-check ok")


if __name__ == "__main__":
    selftest()
