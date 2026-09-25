"""Synthetic S2/S3 records made ONLY from S1 text by code: exact labels by construction (we know which S1 we corrupted).

    rates = estimate("US")                     # op rates measured on US TRUE pairs (+ orphans vs their nearest S1)
    sx, truth = generate(s1_raw, rates, seed)  # corrupt target S1 -> positives, sibling look-alikes, unrelated orphans
    .venv/bin/python probes/P12-synth/synth.py est US|India     # -> probes/P12-synth/rates_<c>.json (aggregates only)
    .venv/bin/python probes/P12-synth/synth.py                  # self-check

Generator structure, read off the train data (P1 taxonomy; the brief's noise list):
- An entity gets (n_S2, n_S3) records drawn from the source country's joint cluster-size distribution.
- Per (entity, source) a BASE address is made once (house-number change / drop, place typo, CITY suffix, dropped or added
  segment, NULL segment, state form): in the data all S3 copies of an entity share e.g. "402 ... Saint Ccloud CITY".
- Per record: name ops (website / handle / alias / fake name / honorific prefix / legal-suffix drop-add-swap / word
  drop-add-reorder / typos / digit-for-letter / accents / brackets / (ID: n) / duplicate token / hyphen / case) and
  light address ops (segment shuffle, street-type abbreviation, empty address, state drop, PO box, case).
- Orphans (26% of real S2/S3 match nothing): ~87% of real false positives are orphans that look like an S1 with a
  different house number -- a hidden SIBLING entity (a word added to the name, house number +delta, same street/city)
  that gets its own copies. The rest are unrelated new businesses built from pool tokens + a pool address.
Vocabulary (legal forms, name tokens, addresses, state table) comes from the TARGET S1 text only; rates and filler
words from the SOURCE country's labelled pairs. No external data, no test text.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.fuzz import ratio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import normalize as N

WORK = ROOT / "data/work"
_TOK = re.compile(r"[a-z0-9]+")
_INDIC = re.compile("[ऀ-෿]")
_D2L = str.maketrans("0134568", "oleasgb")
_L2D = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "g": "6", "b": "8"}
_ACC = {"a": "áàâä", "e": "éèêë", "i": "íîï", "o": "óôö", "u": "úùûü", "c": "ç", "n": "ñ"}
SUFFIX = ["city", "township", "cdp", "twp", "town"]
NULLS = ["NULL", "<NULL>", "null", "N/A"]


def toks(x: str) -> list:
    """Detection tokens: normalize's letter-run merge + digit/I-l fixes, so 'L.L.C' -> llc and 'lnc' -> inc."""
    return [N._tok_fix(t) for t in N._merge_letters(_TOK.findall(anyascii(x).lower().replace(".", "")))]


def _legal_run(t: list) -> int:
    """Length of the trailing run of legal-form tokens."""
    k = 0
    while k < len(t) and t[len(t) - 1 - k] in N.LEGAL:
        k += 1
    return k


# ---------------------------------------------------------------- estimation (source country, labelled)
def _name_ops(a: str, b: str) -> dict:
    """Detected name ops of one true pair (S1 name a, S2/S3 name b)."""
    fb = anyascii(b).lower()
    o = {}
    if b.strip().startswith("@"):
        return {"handle": 1}
    if N._WEB and re.search(N._WEB, fb):
        return {"web": 1, "www": int("www." in fb)}
    if "|" in b:
        return {"pipe": 1}
    if N._ALIAS.search(fb):
        return {"alias": 1}
    o["idtail"] = int(bool(N._ID_TAIL.search(fb)) and not N._ID_TAIL.search(anyascii(a).lower()))
    fb = N._ID_TAIL.sub(" ", fb)
    ta, tb = toks(a), toks(fb)
    ca = [t for t in ta if t not in N.LEGAL and t not in N.HONORIFIC]
    cb = [t for t in tb if t not in N.LEGAL and t not in N.HONORIFIC]
    if ca and cb and max(ratio(x, y) for x in cb for y in ca) < 70:
        return {"fake": 1}
    o["prefix"] = int(bool(tb) and tb[0] in N.HONORIFIC and (not ta or ta[0] != tb[0]))
    if o["prefix"]:
        o["prefix_tok"] = tb[0]
    la, lb = [t for t in ta if t in N.LEGAL], [t for t in tb if t in N.LEGAL]
    o["has_legal"] = int(bool(la))
    o["legal_drop"] = int(bool(la) and not lb)
    o["legal_add"] = int(not la and bool(lb))
    o["legal_swap"] = int(bool(la) and bool(lb) and la != lb)
    used, idx, added, typo, digit, heavy = set(), [], [], 0, 0, 0
    for t in cb:
        if t in ca:
            j = ca.index(t) if ca.index(t) not in used else next((i for i, x in enumerate(ca) if x == t and i not in used), ca.index(t))
            used.add(j); idx.append(j)
            continue
        r = [(ratio(t, y), i) for i, y in enumerate(ca)]
        best, j = max(r) if r else (0, -1)
        if best >= 70:
            used.add(j); idx.append(j)
            if t.translate(_D2L) == ca[j] or (not t.isdigit() and t.translate(_D2L) == ca[j].translate(_D2L)):
                digit += 1
            else:
                typo += 1
                heavy += best < 80
        else:
            added.append(t)
    o["drop"] = int(len(set(range(len(ca))) - used) > 0)
    o["n_drop"] = len(set(range(len(ca))) - used)
    o["drop_last"] = int(bool(ca) and (len(ca) - 1) not in used)
    o["add"] = int(bool(added))
    o["added"] = added
    o["reorder"] = int(any(x > y for x, y in zip(idx, idx[1:])))
    o["typo"] = int(typo > 0)
    o["typo_heavy"] = heavy
    o["typo_n"] = typo
    # raw text: toks() folds digits back to letters, so compare the raw strings (0/1/3/4/5/6/8 inside a word)
    o["digit"] = int(bool(re.search(r"[A-Za-z][0134568]|[0134568][A-Za-z]", b)) and not re.search(r"\d", a))
    o["dup"] = int(any(x == y for x, y in zip(tb, tb[1:])) and not any(x == y for x, y in zip(ta, ta[1:])))
    o["accent"] = int(bool(re.search(r"[^\x00-\x7f]", b)) and not re.search(r"[^\x00-\x7f]", a))
    o["bracket"] = int(bool(re.search(r"[\[\]()]", N._ID_TAIL.sub("", b))) and not re.search(r"[\[\]()]", a))
    o["hyphen"] = int(bool(re.search(r"[A-Za-z]-[A-Za-z]", b)) and not re.search(r"[A-Za-z]-[A-Za-z]", a))
    o["upper"] = int(b.isupper() and not a.isupper())
    o["lower"] = int(b.islower() and not a.islower())
    o["dspace"] = int("  " in b.strip() and "  " not in a)
    return o


def _segs(x: str) -> list:
    return [N._basic(s) for s in anyascii(x).lower().split(",") if N._basic(s)]


def _state_seg(ss: list, country: str) -> tuple:
    tab = N.STATES.get(anyascii(country).lower(), {})
    for i in range(len(ss) - 1, -1, -1):
        if ss[i] in tab:
            return i, ("code" if tab[ss[i]] == ss[i] else "name")
    return -1, ""


def _addr_ops(a: str, b: str, country: str, ha: str = None, hb: str = None) -> dict:
    fb = anyascii(b).lower()
    if not b.strip():
        return {"empty": 1}
    o = {"empty": 0, "null": int(bool(N._PLACEHOLDER.search(fb))),
         "pobox": int(bool(N._POBOX.search(fb)) and not N._POBOX.search(anyascii(a).lower())),
         "upper": int(b.isupper() and not a.isupper())}
    sa = _segs(a)
    sb = [s for s in _segs(N._POBOX.sub(" ", N._PLACEHOLDER.sub(" ", fb)))]
    ia, fa = _state_seg(sa, country)
    ib, fx = _state_seg(sb, country)
    o["state_drop"] = int(ia >= 0 and ib < 0)
    if ia >= 0 and ib >= 0:
        o["state_same_form"] = int(fa == fx)
    if ha is None:
        ha, hb = N.house_number(pd.Series([a])).iloc[0], N.house_number(pd.Series([b])).iloc[0]
    if ha:
        o["hn_drop"] = int(not hb)
        if hb:
            o["hn_change"] = int(ha != hb)
            if ha != hb and len(ha) <= 6 and len(hb) <= 6:
                o["hn_delta"] = int(hb) - int(ha)
            m = re.search(r"(^|[\s,])([#]*)(0*)" + re.escape(hb) + r"(\S*)", b)
            if m:
                o["hn_hash"] = int(bool(m.group(2)))
                o["hn_pad"] = int(bool(m.group(3)))
                tail = m.group(4)
                o["hn_letter"] = int(bool(re.match(r"^[A-Za-z]\b", tail)))
                o["hn_range"] = int(bool(re.match(r"^-\d", tail)))
                o["hn_dash"] = int(bool(re.match(r"^-[A-Za-z]", tail)))
                o["hn_dot"] = int(tail.startswith("."))
    # segments compared with digits removed: the house number is measured above, a place typo is a digit-free change
    nod = lambda s: " ".join(re.sub(r"\d+\w*", " ", s).split())
    ra = [nod(s) for i, s in enumerate(sa) if i != ia]
    rb = [nod(s) for i, s in enumerate(sb) if i != ib]
    ra_seg = [N._seg(x) for x in ra]
    used, idx, typo, suffix, added = set(), [], 0, 0, 0
    for s in rb:
        if not s:
            continue
        if s in ra:
            j = ra.index(s)
        else:
            base = s.rsplit(" ", 1)
            if len(base) == 2 and base[1] in SUFFIX and base[0] in ra:
                j = ra.index(base[0]); suffix += 1
            elif N._seg(s) in ra_seg:
                j = ra_seg.index(N._seg(s))
            else:
                r = [(ratio(N._seg(s), x), i) for i, x in enumerate(ra_seg) if x]
                best, j = max(r) if r else (0, -1)
                if best >= 80:
                    typo += 1
                else:
                    added += 1
                    continue
        used.add(j); idx.append(j)
    wa = [t for t in toks(a) if t in _SHORT or t in _LONG]
    wb = [t for t in toks(b) if t in _SHORT or t in _LONG]
    o["seg_typo"] = int(typo > 0)
    o["city_suffix"] = int(suffix > 0)
    o["abbr"] = int(wa != wb and sorted(_SHORT.get(t, t) for t in wa) == sorted(_SHORT.get(t, t) for t in wb))
    o["seg_add"] = int(added > 0)
    o["seg_drop"] = int(len({i for i, x in enumerate(ra) if x} - used) > 0)
    o["shuffle"] = int(any(x > y for x, y in zip(idx, idx[1:])) or (ia >= 0 and ib >= 0 and (ia == len(sa) - 1) != (ib == len(sb) - 1)))
    return o


FORMS = ["handle", "web", "pipe", "alias", "fake"]
NAME_KEYS = ["handle", "web", "pipe", "alias", "fake", "idtail", "prefix", "drop", "add", "reorder", "typo", "digit",
             "dup", "accent", "bracket", "hyphen", "upper", "lower", "dspace"]
ADDR_KEYS = ["empty", "null", "pobox", "upper", "state_drop", "hn_drop", "hn_change", "hn_hash", "hn_pad", "hn_letter",
             "hn_range", "hn_dash", "hn_dot", "seg_typo", "city_suffix", "abbr", "seg_add", "seg_drop", "shuffle"]
BASE_KEYS = ["hn_change", "hn_drop", "seg_typo", "city_suffix", "seg_add", "seg_drop", "null"]


def _load_country(country: str, n: int, seed: int):
    s1 = pd.read_parquet(WORK / "train_source1.parquet", filters=[("country", "==", country)])
    fo = pd.read_parquet(WORK / "s1_folds.parquet").set_index("entity_id").fold
    s1 = s1[fo.reindex(s1.entity_id).to_numpy() >= 2]             # never look at eval folds 0/1
    gt = pd.read_parquet(WORK / "gt_pairs.parquet")
    gt = gt[gt.s1.isin(s1.entity_id)]
    return s1, gt


def estimate(country: str, n: int = 15000, n_orph: int = 15000, seed: int = 0) -> dict:
    """Rates of every op on `country` TRUE pairs of folds >= 2 (per source S2 / S3), cluster sizes, orphan shares,
    and the sibling shape of orphans (vs their top-1 blocked S1). Aggregates only."""
    s1, gt = _load_country(country, n, seed)
    cnt = gt.assign(src=gt.sx.str[:2]).groupby(["s1", "src"]).size().unstack(fill_value=0).reindex(s1.entity_id, fill_value=0)
    sizes = Counter(zip(cnt.get("S2", 0), cnt.get("S3", 0)))
    samp = s1.sample(n=min(n, len(s1)), random_state=seed)
    g = gt[gt.s1.isin(samp.entity_id)]
    a1 = samp.set_index("entity_id")
    out = {"country": country, "n_s1_pool": len(s1), "cluster_sizes": {f"{k[0]},{k[1]}": v for k, v in sizes.items()}}
    rows = pair_rows(country, a1, g)
    import pickle
    (WORK / "p12").mkdir(parents=True, exist_ok=True)
    (WORK / f"p12/est_rows_{country}.pkl").write_bytes(pickle.dumps(rows))     # data/ is gitignored
    out |= _summarise(rows)
    out |= _orphans(country, s1, gt, n_orph, seed)
    return out


def pair_rows(country: str, a1: pd.DataFrame, g: pd.DataFrame) -> list:
    """Detected ops for every (s1, sx) pair of g; a1 = raw S1 rows indexed by entity_id (must hold every g.s1)."""
    sx = pd.concat([pd.read_parquet(WORK / f"train_source{i}.parquet", filters=[("entity_id", "in", g.sx.tolist())])
                    for i in (2, 3)]).set_index("entity_id")
    A, Bx = a1.business_address.reindex(g.s1), sx.business_address.reindex(g.sx)
    H1, HX = N.house_number(A.reset_index(drop=True)), N.house_number(Bx.reset_index(drop=True))
    rows = []
    for s, x, ha, hb in zip(g.s1, g.sx, H1, HX):
        an, aa, bn, ba = a1.business_name[s], a1.business_address[s], sx.business_name[x], sx.business_address[x]
        ind = bool(_INDIC.search(bn))
        rows.append({"s1": s, "src": x[:2], "indic": int(ind), "name": {} if ind else _name_ops(an, bn),
                     "addr": _addr_ops(aa, ba, country, ha, hb)})
    return rows


def _mean(rows, part, k, cond=None):
    v = [r[part][k] for r in rows if k in r[part] and (cond is None or cond(r))]
    return round(float(np.mean(v)), 4) if v else 0.0


def _summarise(rows: list) -> dict:
    out = {}
    for src in ("S2", "S3"):
        R = [r for r in rows if r["src"] == src]
        lat = [r for r in R if not r["indic"]]
        nm = {k: round(float(np.mean([r["name"].get(k, 0) for r in lat])), 4) if lat else 0.0 for k in FORMS}
        norm = [r for r in lat if not any(r["name"].get(k) for k in FORMS)]
        nm |= {k: round(float(np.mean([r["name"].get(k, 0) for r in norm])), 4) if norm else 0.0
               for k in NAME_KEYS if k not in FORMS}
        nm["www"] = _mean(R, "name", "www", lambda r: r["name"].get("web") == 1)
        leg = [r for r in norm if r["name"].get("has_legal") == 1]
        nol = [r for r in norm if r["name"].get("has_legal") == 0]
        nm |= {"legal_drop": _mean(leg, "name", "legal_drop"), "legal_swap": _mean(leg, "name", "legal_swap"),
               "legal_add": _mean(nol, "name", "legal_add"),
               "n_drop2": _mean([r for r in R if r["name"].get("drop")], "name", "n_drop") - 1,
               "drop_last": _mean([r for r in R if r["name"].get("drop")], "name", "drop_last"),
               "typo_heavy": round(sum(r["name"].get("typo_heavy", 0) for r in R) /
                                   max(1, sum(r["name"].get("typo_n", 0) for r in R)), 4),
               "indic": round(float(np.mean([r["indic"] for r in R])), 4) if R else 0.0}
        nm["prefix_tok"] = dict(Counter(r["name"]["prefix_tok"] for r in R if "prefix_tok" in r["name"]).most_common(10))
        nm["added"] = dict(Counter(t for r in norm for t in r["name"].get("added", []) if len(t) > 2).most_common(60))
        ad = {k: _mean(R, "addr", k) for k in ADDR_KEYS}
        ad["state_same_form"] = _mean(R, "addr", "state_same_form")
        d = np.array([r["addr"]["hn_delta"] for r in R if "hn_delta" in r["addr"]])
        ad["hn_delta"] = dict(Counter(int(x) for x in d if abs(x) < 50).most_common(40))
        ad["hn_big_share"] = round(float((np.abs(d) >= 50).mean()), 4) if len(d) else 0.0      # digit dropped / added
        ad["hn_big_neg"] = round(float((d <= -50).sum() / max((np.abs(d) >= 50).sum(), 1)), 4)
        # base share: among entities with >= 2 records of this source, P(both | one) -> share decided per entity
        by = {}
        for r in R:
            by.setdefault(r["s1"], []).append(r["addr"])
        base = {}
        for k in BASE_KEYS:
            one = both = 0
            for recs in by.values():
                v = [x.get(k, 0) for x in recs if not x.get("empty")]
                for i in range(len(v)):
                    for j in range(i + 1, len(v)):
                        if v[i] or v[j]:
                            one += 1
                            both += v[i] and v[j]
            r_ = ad.get(k, 0)
            pb = both / one if one else 0
            base[k] = round(float(np.clip((pb - r_) / max(1 - r_, 1e-6), 0, 1)), 3)
        ad["base_share"] = base
        out[src] = {"n": len(R), "name": nm, "addr": ad}
    return out


def _orphans(country: str, s1: pd.DataFrame, gt: pd.DataFrame, n: int, seed: int) -> dict:
    """Orphan share per source; every orphan's top-1 blocked S1 (P4's full-train candidates) as its 'parent':
    orphans per parent (family size), and for a sample: look-alike share, added words, house-number offset."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files = sorted((WORK / "p3/txt").glob(f"{country}_sx_*.parquet"))
    ids = pa.concat_arrays([pq.read_table(f, columns=["entity_id"]).column(0).combine_chunks() for f in files])
    orph = ~pc.is_in(ids, value_set=pq.read_table(WORK / "gt_pairs.parquet", columns=["sx"]).column(0).combine_chunks()).to_numpy(zero_copy_only=False)
    is2 = pc.starts_with(ids, "S2").to_numpy(zero_copy_only=False)
    out = {"orphan_share": {"S2": round(float(orph[is2].mean()), 4), "S3": round(float(orph[~is2].mean()), 4)},
           "sx_per_s1": round(len(ids) / pq.ParquetFile(WORK / f"p3/txt/{country}_s1.parquet").metadata.num_rows, 4)}
    c = pd.read_parquet(WORK / f"p4/cands_train_full_{country}.parquet", columns=["q1", "qx", "block_score", "fold"])
    c = c[orph[c.qx.to_numpy()]]
    c = c.sort_values(["qx", "block_score"], ascending=[True, False], kind="stable").drop_duplicates("qx")
    c = c[c.fold.to_numpy() >= 2]                                   # parents from folds >= 2 only
    fam = c.groupby("q1").size()
    out["orph_family"] = {str(k): int(v) for k, v in fam.clip(upper=8).value_counts().sort_index().items()}
    rng = np.random.default_rng(seed)
    c = c.iloc[rng.choice(len(c), min(n, len(c)), replace=False)]
    ids1 = pd.read_parquet(WORK / f"p3/txt/{country}_s1.parquet", columns=["entity_id"]).entity_id
    par, oid = ids1.iloc[c.q1.to_numpy()].to_numpy(), ids.take(pa.array(c.qx.to_numpy())).to_pylist()
    del ids, orph, is2
    a1 = s1.set_index("entity_id")
    sx = pd.concat([pd.read_parquet(WORK / f"train_source{i}.parquet", filters=[("entity_id", "in", oid)])
                    for i in (2, 3)]).set_index("entity_id")
    H1 = N.house_number(a1.business_address.reindex(par).reset_index(drop=True))
    HX = N.house_number(sx.business_address.reindex(oid).reset_index(drop=True))
    look, deltas, added, same_name, tot = 0, [], Counter(), 0, 0
    for p, x, ha, hb in zip(par, oid, H1, HX):
        an, aa, bn, ba = a1.business_name[p], a1.business_address[p], sx.business_name[x], sx.business_address[x]
        if _INDIC.search(bn):
            continue
        tot += 1
        ca = [t for t in toks(an) if t not in N.LEGAL and t not in N.HONORIFIC]
        cb = [t for t in toks(N._ID_TAIL.sub(" ", anyascii(bn).lower())) if t not in N.LEGAL and t not in N.HONORIFIC]
        if not ca or not cb:
            continue
        shared = sum(max(ratio(t, y) for y in ca) >= 70 for t in cb)
        if shared / max(len(ca), 1) < 0.5:
            continue
        look += 1
        ext = [t for t in cb if max(ratio(t, y) for y in ca) < 70 and len(t) > 2]
        added.update(ext)
        same_name += not ext
        if ha and hb and len(ha) <= 6 and len(hb) <= 6:
            deltas.append(int(hb) - int(ha))
    d = np.array(deltas)
    near = d[np.abs(d) < 60]
    return out | {"orph_lookalike": round(look / max(tot, 1), 4), "orph_sib_added_word": round(1 - same_name / max(look, 1), 4),
                  "orph_sib_added": dict(added.most_common(60)),
                  "orph_hn_delta": {str(k): int(v) for k, v in Counter(near.tolist()).most_common(40)},
                  "orph_hn_far": round(float((np.abs(d) >= 60).mean()), 4) if len(d) else 0.0}


# ---------------------------------------------------------------- generation (target S1 text only)
class Vocab:
    """Everything the generator draws from the TARGET S1 pool (label-free)."""

    def __init__(self, s1: pd.DataFrame, rng):
        self.rng = rng
        leg, words = Counter(), Counter()
        for nm in s1.business_name.head(200_000):
            w = nm.split()
            t = [x.lower().strip(".,()") for x in w]
            k = _legal_run([re.sub(r"[^a-z]", "", x) for x in t])
            if k:
                leg[" ".join(w[len(w) - k:])] += 1
            words.update(x for x in w[:len(w) - k] if x.isalpha() and len(x) > 2)
        self.legal = list(leg)
        self.legal_p = np.array(list(leg.values()), float) / max(sum(leg.values()), 1)
        top = [w for w, _ in words.most_common(20000)]
        self.words = top or ["Alpha"]
        self.addrs = s1.business_address.to_numpy()
        self.country = s1.country.iloc[0] if len(s1) else ""
        tab = N.STATES.get(anyascii(self.country).lower(), {})
        self.code2name, self.name2code = {}, {}
        for k, v in tab.items():
            if k == v:
                continue
            self.code2name.setdefault(v, k)
            self.name2code.setdefault(k, v)

    def legal_form(self) -> str:
        return self.legal[self.rng.choice(len(self.legal), p=self.legal_p)] if self.legal else "Ltd"

    def fake_word(self) -> str:
        """A pronounceable non-word glued from pieces of pool words ('Nylaquolyra'-style fake names)."""
        r = self.rng
        parts = [self.words[r.integers(len(self.words))] for _ in range(r.integers(2, 4))]
        s = "".join(p[:r.integers(2, 5)] if i == 0 else p[r.integers(0, max(1, len(p) - 3)):][:4] for i, p in enumerate(parts))
        return s.capitalize()


def _typo(w: str, rng, heavy: float) -> str:
    if len(w) < 3:
        return w
    if rng.random() < heavy:                                       # scramble the inside ('NELWKSB', 'Mdnamglemt')
        mid = list(w[1:-1]); rng.shuffle(mid)
        return w[0] + "".join(mid) + w[-1]
    i = int(rng.integers(1, len(w)))
    op = rng.integers(5)
    c = "abcdefghijklmnopqrstuvwxyz"[rng.integers(26)]
    if op == 0:
        return w[:i] + w[i + 1:]                                   # deletion
    if op == 1:
        return w[:i] + c + w[i:]                                   # insertion
    if op == 2:
        return w[:i] + c + w[i + 1:]                               # substitution
    if op == 3 and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]                 # transposition
    return w[:i] + w[i] + w[i:]                                    # doubled letter ('Ccloud')


def _digitise(w: str, rng) -> str:
    ix = [i for i, ch in enumerate(w) if ch.lower() in _L2D]
    if not ix:
        return w
    i = ix[rng.integers(len(ix))]
    return w[:i] + _L2D[w[i].lower()] + w[i + 1:]


def _accent(w: str, rng) -> str:
    ix = [i for i, ch in enumerate(w) if ch.lower() in _ACC]
    if not ix:
        return w
    i = ix[rng.integers(len(ix))]
    a = _ACC[w[i].lower()]
    ch = a[rng.integers(len(a))]
    return w[:i] + (ch.upper() if w[i].isupper() else ch) + w[i + 1:]


def _draw(counter: dict, rng, default):
    if not counter:
        return default
    k = list(counter)
    p = np.array(list(counter.values()), float)
    return k[rng.choice(len(k), p=p / p.sum())]


def make_name(name: str, R: dict, V: Vocab, rng) -> str:
    """One S2/S3 name from an entity name; R = rates[source]["name"]."""
    r = rng.random
    w = name.split()
    k = _legal_run([re.sub(r"[^a-z]", "", x.lower()) for x in w])
    core, legal = w[:len(w) - k] or w, w[len(w) - k:]
    u = r()
    if u < R["handle"]:
        return "@" + "".join(core).lower()
    u -= R["handle"]
    if u < R["web"]:
        c = list(core)
        if r() < 0.25 and len(c) > 1:
            rng.shuffle(c)
        s = re.sub(r"[^a-z0-9]", "", "".join(c).lower())
        return ("www." if r() < R.get("www", 0) else "") + s + [".com", ".net", ".org", ".co"][rng.choice(4, p=[.7, .12, .12, .06])]
    u -= R["web"]
    if u < R["alias"]:
        f = V.fake_word() + ("" if r() < 0.5 else " " + ["Co", "One", "Sys"][rng.integers(3)])
        m = ["doing business as", "dba", "d/b/a", "f/k/a", "formerly known as", "aka"][rng.integers(6)]
        return f"{f} {m} {name}" if r() < 0.8 else f"{name} {m} {f}"
    u -= R["alias"]
    if u < R["fake"]:
        return V.fake_word()
    u -= R["fake"]
    if u < R["pipe"]:
        return f"{name} | www.{re.sub(r'[^a-z0-9]', '', ''.join(core).lower())}.com"
    core = list(core)
    if legal:
        if r() < R["legal_drop"]:
            legal = legal[:-1] if len(legal) > 1 and r() < 0.5 else []
        elif r() < R["legal_swap"] / max(1 - R["legal_drop"], 1e-6):
            legal = V.legal_form().split()
    elif r() < R["legal_add"]:
        legal = V.legal_form().split()
    if len(core) > 1 and r() < R["drop"]:
        n = 1 + int(r() < R.get("n_drop2", 0))
        for _ in range(min(n, len(core) - 1)):
            core.pop(len(core) - 1 if r() < R.get("drop_last", 0.5) else int(rng.integers(len(core))))
    if r() < R["add"]:
        core.append(str(_draw(R["added"], rng, "services")).title())
    if len(core) > 1 and r() < R["reorder"]:
        i = int(rng.integers(len(core) - 1))
        if r() < 0.5:
            core[i], core[i + 1] = core[i + 1], core[i]
        else:
            core = core[-1:] + core[:-1]
    for fn, key in ((lambda x: _typo(x, rng, R.get("typo_heavy", 0.2)), "typo"), (lambda x: _digitise(x, rng), "digit"),
                    (lambda x: _accent(x, rng), "accent")):
        if r() < R[key] and core:
            i = int(rng.integers(len(core)))
            core[i] = fn(core[i])
    if r() < R["dup"] and core:
        i = int(rng.integers(len(core)))
        core.insert(i, core[i])
    if r() < R["bracket"] and (core or legal):
        if legal and r() < 0.5:
            legal = ["(" + " ".join(legal) + ")"] if r() < 0.5 else ["[" + " ".join(legal) + "]"]
        else:
            i = int(rng.integers(len(core)))
            core[i] = "[" + core[i] + "]" if r() < 0.5 else "(" + core[i] + ")"
    if r() < R["prefix"]:
        core.insert(0, str(_draw(R["prefix_tok"], rng, "the")).title())
    out = core + legal
    if r() < R["hyphen"] and len(out) > 1:
        i = int(rng.integers(len(out) - 1))
        out[i:i + 2] = [out[i] + "-" + out[i + 1]]
    s = (" " if r() >= R["dspace"] else "  ").join(out)
    if r() < R["idtail"]:
        s += f" (ID: {rng.integers(1000, 99999)})"
    if r() < R["upper"]:
        s = s.upper()
    elif r() < R["lower"]:
        s = s.lower()
    return s


_HN = re.compile(r"^(\D*?)(\d+)(.*)$")


def _hn_seg(segs: list) -> int:
    for i, s in enumerate(segs):
        if re.match(r"^\s*(?:#|no\.?\s*|h\.?\s*no\.?\s*|plot\s*no\.?\s*|flat\s*no\.?\s*)?\d", s, re.I):
            return i
    return -1


def _set_hn(seg: str, fn) -> str:
    m = _HN.match(seg)
    return m.group(1) + fn(m.group(2)) + m.group(3) if m else seg


def base_address(addr: str, A: dict, V: Vocab, rng, p=None) -> list:
    """Per (entity, source) address: a list of segments with the entity-level noise applied once."""
    r = rng.random
    p = p or {}
    segs = [s.strip() for s in addr.split(",") if s.strip()]
    h = _hn_seg(segs)
    st = next((i for i in range(len(segs) - 1, -1, -1) if N._basic(anyascii(segs[i]).lower()) in V.name2code
               or N._basic(anyascii(segs[i]).lower()) in V.code2name), -1)
    bs = lambda k: A["base_share"].get(k, 0)
    q = lambda k: A[k] * (bs(k) if p.get("base") else 1 - bs(k))    # entity-level share once, the rest per record
    if h >= 0 and r() < q("hn_change"):
        if r() < A.get("hn_big_share", 0):                         # a digit dropped (5402 -> 402) or added
            fn = (lambda x: x[1:] or x) if r() < A.get("hn_big_neg", 0.8) else (lambda x: x + str(rng.integers(10)))
        else:
            d = int(_draw(A["hn_delta"], rng, 1))
            fn = lambda x: str(max(1, int(x) + d))
        segs[h] = _set_hn(segs[h], fn)
    if h >= 0 and r() < q("hn_drop"):
        segs[h] = re.sub(r"^\s*\S*\d\S*\s*", "", segs[h]).strip() or segs[h]
    places = [i for i in range(len(segs)) if i not in (h, st) and not re.search(r"\d", segs[i])]
    if places and r() < q("seg_typo"):
        i = places[rng.integers(len(places))]
        w = segs[i].split()
        j = int(rng.integers(len(w)))
        w[j] = _typo(w[j], rng, 0.0)
        segs[i] = " ".join(w)
    if places and r() < q("city_suffix"):
        i = places[rng.integers(len(places))]
        segs[i] += " " + SUFFIX[rng.choice(len(SUFFIX), p=[.5, .2, .15, .1, .05])].upper()
    # per-segment drop: the source's P(>= 1 dropped) spread over its mean segment count, so long (India) addresses
    # lose more segments than short (US) ones at the same per-segment rate
    ps = 1 - (1 - q("seg_drop")) ** (1 / max(A.get("mean_segs", 2.0), 1.0))
    cand = [i for i in range(len(segs)) if i != st]
    drop = {i for i in cand if r() < ps}
    if drop and len(drop) < len(cand):
        segs = [x for i, x in enumerate(segs) if i not in drop]
    if r() < q("seg_add"):
        o = V.addrs[rng.integers(len(V.addrs))].split(",")
        extra = ["Unit " + "ABCDF"[rng.integers(5)], f"PMB {rng.integers(100, 999)}", o[0].strip()]
        segs.insert(int(rng.integers(len(segs) + 1)), extra[rng.choice(3, p=[.4, .2, .4])])
    if r() < q("null"):
        segs.insert(int(rng.integers(len(segs) + 1)), NULLS[rng.integers(len(NULLS))])
    return segs


def make_address(segs: list, A: dict, V: Vocab, rng) -> str:
    """Per-record address from the entity/source base segments."""
    r = rng.random
    if r() < A["empty"] or not segs:
        return ""
    segs = list(segs)
    st = next((i for i in range(len(segs) - 1, -1, -1) if N._basic(anyascii(segs[i]).lower()) in V.name2code
               or N._basic(anyascii(segs[i]).lower()) in V.code2name), -1)
    if st >= 0:
        key = N._basic(anyascii(segs[st]).lower())
        is_code = key in V.code2name
        if r() >= A["state_same_form"]:                            # swap code <-> name
            segs[st] = V.code2name[key].title() if is_code else V.name2code[key].upper()
        if r() < A["state_drop"]:
            segs.pop(st)
    h = _hn_seg(segs)
    if h >= 0:
        if r() < A["hn_hash"]:
            segs[h] = ("##" if r() < 0.5 else "#") + segs[h]
        elif r() < A["hn_pad"]:
            segs[h] = _set_hn(segs[h], lambda x: "0" * int(rng.integers(1, 3)) + x)
        if r() < A["hn_letter"]:
            segs[h] = _set_hn(segs[h], lambda x: x + "abcd"[rng.integers(4)])
        elif r() < A["hn_range"]:
            segs[h] = _set_hn(segs[h], lambda x: f"{x}-{int(x) + int(rng.integers(1, 9))}")
        elif r() < A["hn_dash"]:
            segs[h] = _set_hn(segs[h], lambda x: x + "-" + "ABC"[rng.integers(3)])
        elif r() < A["hn_dot"]:
            segs[h] = _set_hn(segs[h], lambda x: x + ".")
    if r() < A["abbr"]:
        segs = [" ".join(_abbr(t) for t in s.split()) for s in segs]
    if r() < A["pobox"]:
        segs.insert(int(rng.integers(len(segs) + 1)), f"PO BOX {rng.integers(100, 9999)}")
    if len(segs) > 1 and r() < A["shuffle"]:
        rng.shuffle(segs)
    s = ", ".join(segs)
    return s.upper() if r() < A["upper"] else s


_SHORT = {k: v for k, v in N.STREET.items() if k.isalpha() and v.isalpha() and len(v) < len(k)
          and k in {"street", "road", "drive", "avenue", "lane", "court", "circle", "place", "boulevard", "trail",
                    "terrace", "highway", "parkway", "cove", "fort", "mount", "north", "south", "east", "west"}}
_LONG = {v: k for k, v in _SHORT.items()}


def _abbr(t: str) -> str:
    lo = t.lower().strip(".")
    if lo in _SHORT:
        v = _SHORT[lo]
        return v.upper() if t.isupper() else v.title()
    if lo in _LONG:
        v = _LONG[lo]
        return v.upper() if t.isupper() else v.title()
    return t


def sibling(name: str, addr: str, rates: dict, V: Vocab, rng) -> tuple:
    """A hidden look-alike entity: same street / city, house number + delta, often an extra name word."""
    w = name.split()
    k = _legal_run([re.sub(r"[^a-z]", "", x.lower()) for x in w])
    core, legal = w[:len(w) - k] or w, w[len(w) - k:]
    if rng.random() < rates["orph_sib_added_word"]:
        word = str(_draw(rates["orph_sib_added"], rng, "group")).title()
        if len(core) > 1 and rng.random() < 0.35:
            core = core[:-1] + [word]                              # replace the last content word
        else:
            core = core + [word]
        if legal and rng.random() < 0.3:
            legal = V.legal_form().split()
    segs = [s.strip() for s in addr.split(",")]
    h = _hn_seg(segs)
    if h >= 0 and rng.random() < rates.get("orph_hn_far", 0):     # a far number on the same street
        segs[h] = _set_hn(segs[h], lambda x: str(int(rng.integers(1, 10 ** max(2, min(len(x), 5))))))
    elif h >= 0:                                                   # the generator's offsets: +1..+21
        d = int(_draw(rates["orph_hn_delta"], rng, 3))
        segs[h] = _set_hn(segs[h], lambda x: str(max(1, int(x) + d)))
    else:                                                          # no house number: a different street, same city
        o = [s.strip() for s in V.addrs[rng.integers(len(V.addrs))].split(",")]
        segs = [o[0]] + segs[1:] if len(segs) > 1 else [o[0]] + segs
    return " ".join(core + legal), ", ".join(segs)


def unrelated(V: Vocab, rng) -> tuple:
    """A new business: pool words + maybe a legal form, at another pool record's street with a fresh house number."""
    n = int(rng.integers(1, 4))
    name = " ".join(V.words[rng.integers(min(len(V.words), 3000))] for _ in range(n))
    if rng.random() < 0.6:
        name += " " + V.legal_form()
    segs = [s.strip() for s in V.addrs[rng.integers(len(V.addrs))].split(",")]
    h = _hn_seg(segs)
    if h >= 0:
        segs[h] = _set_hn(segs[h], lambda x: str(int(rng.integers(1, 10 ** max(1, min(len(x), 5))))))
    return name, ", ".join(segs)


def _copies(name, addr, n2, n3, rates, V, rng, owner, out):
    for src, n in (("S2", n2), ("S3", n3)):
        if not n:
            continue
        R = rates[src]
        base = base_address(addr, R["addr"], V, rng, {"base": True})
        for _ in range(n):
            segs = base_address(", ".join(base), R["addr"], V, rng, {"base": False})
            out.append((src, make_name(name, R["name"], V, rng), make_address(segs, R["addr"], V, rng), owner))


def generate(s1: pd.DataFrame, rates: dict, seed: int = 0, prefix: str = "p12") -> tuple:
    """s1: raw target S1 rows (entity_id, business_name, business_address, country), all of ONE country.
    Returns (sx raw DataFrame [entity_id, business_name, business_address, country], truth DataFrame [s1, sx]):
    every S1 gets a cluster, then orphans (siblings + unrelated) until the source country's orphan share is met."""
    rng = np.random.default_rng(seed)
    V = Vocab(s1, rng)
    sizes = [tuple(map(int, k.split(","))) for k in rates["cluster_sizes"]]
    p = np.array(list(rates["cluster_sizes"].values()), float)
    pick = rng.choice(len(sizes), size=len(s1), p=p / p.sum())
    out = []
    for (e, nm, ad), k in zip(s1[["entity_id", "business_name", "business_address"]].itertuples(index=False), pick):
        n2, n3 = sizes[k]
        _copies(nm, ad, n2, n3, rates, V, rng, e, out)
    n_true = len(out)
    osh = np.mean(list(rates["orphan_share"].values()))
    n_orph = int(n_true * osh / (1 - osh))
    fam = {int(k): v for k, v in rates["orph_family"].items()}
    o2 = rates["orphan_share"]["S2"] / sum(rates["orphan_share"].values())
    look = rates["orph_lookalike"]
    names, addrs = s1.business_name.to_numpy(), s1.business_address.to_numpy()
    made = 0
    while made < n_orph:
        k = int(_draw(fam, rng, 1))
        n2 = int(rng.binomial(k, o2))
        if rng.random() < look:
            i = int(rng.integers(len(s1)))
            nm, ad = sibling(names[i], addrs[i], rates, V, rng)
        else:
            nm, ad = unrelated(V, rng)
        _copies(nm, ad, n2, k - n2, rates, V, rng, "", out)
        made += k
    sx = pd.DataFrame(out, columns=["src", "business_name", "business_address", "owner"])
    sx["entity_id"] = sx.src + f"-{prefix}" + pd.Series(np.arange(len(sx)), dtype=str).str.zfill(8)
    sx["country"] = s1.country.iloc[0]
    truth = sx.loc[sx.owner != "", ["owner", "entity_id"]].set_axis(["s1", "sx"], axis=1).reset_index(drop=True)
    return sx[["entity_id", "business_name", "business_address", "country"]], truth


def _check():
    rates = json.loads((HERE / "rates_US.json").read_text()) if (HERE / "rates_US.json").exists() else None
    assert _legal_run(["sunrise", "traders", "pvt", "ltd"]) == 2 and _legal_run(["blue", "alliance"]) == 0
    o = _name_ops("First Networks Inc", "firstnetworks.com")
    assert o.get("web") == 1, o
    o = _name_ops("Gigi's Heartland Tax", "Heartland Gigi's Tax")
    assert o["reorder"] == 1 and o["drop"] == 0, o
    o = _name_ops("Urban Auto Body", "Urban  Auto")
    assert o["drop"] == 1 and o["drop_last"] == 1 and o["dspace"] == 1, o
    o = _name_ops("Enum", "Enum [Corp] [Corp]")
    assert o["legal_add"] == 1 and o["bracket"] == 1, o
    a = _addr_ops("1791 Harris Road, Penfield, NY", "Pennfield, New York, 1786 Harris Rd", "US")
    assert a["hn_change"] == 1 and a["hn_delta"] == -5 and a["seg_typo"] == 1 and a["shuffle"] == 1 \
        and a["state_same_form"] == 0 and a["abbr"] == 1, a
    a = _addr_ops("238 Kekoskee Street, City Of Mayville, WI", "Kekoskee St, Mayvlile CITY, Wisconsin", "US")
    assert a["hn_drop"] == 1, a
    if rates:
        s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "business_name": ["Twisted Deli LLC", "Metro Holding Co"],
                           "business_address": ["4 Dix Road, Woburn, MA", "3860 Pin Oak Road, Raleigh, NC"],
                           "country": "US"})
        sx, tr = generate(s1, rates, seed=1)
        assert len(tr) <= len(sx) and set(tr.s1) <= {"S1-1", "S1-2"} and sx.entity_id.is_unique
        assert sx.entity_id.str[:3].isin(["S2-", "S3-"]).all()
    print("synth self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "est":
        r = estimate(sys.argv[2])
        (HERE / f"rates_{sys.argv[2]}.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k not in ("cluster_sizes",)}, indent=1)[:6000])
    else:
        _check()
