"""P67 generator: synthetic (S1, SX) pairs from S1 records alone (organiser ruling: "synthetic pairs from the provided
records" are allowed). No labels are read for the target country.
  pos  apply the generator-op catalogue (P40 op rates on India/US train true pairs) to an S1 -> synthetic SX, y = 1
  neg  (S1 a, synthetic SX of a DIFFERENT S1 b), b among a's TF-IDF near neighbours in the same country (same street,
       shared name words, same city), y = 0. S1 is deduplicated, so b is always another business.
Texts go through P3's cleaning (block.texts, the CE's input space): ta = S1 "name | addr", tb = SX "name | addr".

  gen C N      C in France (test S1) | India | US (train fold-0 S1 = the LOCO frames' S1) -> data/work/p67/sm/syn_<C>.parquet
  sanity C     synthetic vs real (train) true / band-false pairs: name, addr, full-text similarity quantiles
    .venv/bin/python probes/P67-fr-synth/gen67.py gen France 70000
"""
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P3-block")]
W, OUT = ROOT / "data/work", ROOT / "data/work/p67/sm"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

# P40 catalogue_rates.csv, mean of S2/S3 (India, US); France = mean of India and US (Latin script, no Indic ops).
RATES = {
    "India": dict(legal_drop=.197, legal_swap=.091, legal_add=.035, generic=.076, tokdrop=.109, reorder=.081, dup=.027,
                  insert=.073, digit=.022, t_del=.007, t_ins=.010, t_sub=.011, t_tr=.003, t_multi=.026, web=.037,
                  gib=.017, honor=.080, bracket=.029, empty=.039, nullseg=.021, hn_drop=.074, hn_small=.008,
                  hn_other=.027, hn_label=.149, abbr=.227, seg_drop=.609, seg_reorder=.349, seg_typo=.186, state=.351,
                  seg_add=.070),
    "US": dict(legal_drop=.152, legal_swap=.015, legal_add=.085, generic=.072, tokdrop=.197, reorder=.083, dup=.042,
               insert=.108, digit=.029, t_del=.010, t_ins=.011, t_sub=.013, t_tr=.004, t_multi=.034, web=.051, gib=.014,
               honor=.013, bracket=.033, empty=.047, nullseg=.028, hn_drop=.081, hn_small=.009, hn_other=.054,
               hn_label=.073, abbr=.519, seg_drop=.240, seg_reorder=.268, seg_typo=.279, state=.446, seg_add=.109),
}
RATES["France"] = {k: (RATES["India"][k] + RATES["US"][k]) / 2 for k in RATES["US"]} | {"honor": 0.0}

# Small hand-written dictionaries (allowed: legal suffixes, street types). Nothing geographic.
LEGAL = {"France": ["SARL", "SAS", "SASU", "SA", "EURL", "SCI", "SNC", "EI", "SELARL"],
         "US": ["LLC", "Inc", "Corp", "Co", "Ltd", "LP", "PC", "PLLC"],
         "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "Limited", "Co"]}
LEGAL_TOK = {t.lower() for v in LEGAL.values() for x in v for t in x.split()} | {
    "inc", "incorporated", "corporation", "company", "limited", "private", "pvt", "llp", "opc", "eirl", "scop", "scp",
    "gie", "ets", "etablissement", "etablissements", "établissements"}
GENERIC = {"France": ["Services", "Compagnie", "Groupe", "Participations", "Développement", "Conseil", "Centre",
                      "(France)", "Holding", "Distribution"],
           "US": ["Services", "Center", "Group", "Enterprises", "Solutions", "Holdings", "International", "Partners"],
           "India": ["Services", "Enterprises", "Centre", "Solutions", "Traders", "Group", "Industries", "Associates"]}
HONOR = {"India": ["M/s", "Sri", "Shri"], "US": ["The"], "France": ["The"]}
STREET = {"France": {"rue": "R", "avenue": "Av", "boulevard": "Bd", "place": "Pl", "chemin": "Ch", "impasse": "Imp",
                     "allée": "All", "allee": "All", "route": "Rte", "cours": "Crs", "quai": "Q", "square": "Sq",
                     "faubourg": "Fg", "résidence": "Res", "residence": "Res", "saint": "St", "sainte": "Ste"},
          "US": {"street": "St", "road": "Rd", "drive": "Dr", "avenue": "Ave", "lane": "Ln", "court": "Ct",
                 "boulevard": "Blvd", "place": "Pl", "highway": "Hwy", "parkway": "Pkwy", "circle": "Cir",
                 "north": "N", "south": "S", "east": "E", "west": "W", "suite": "Ste", "terrace": "Ter"},
          "India": {"road": "Rd", "street": "St", "nagar": "Ngr", "colony": "Col", "sector": "Sec", "main": "Mn",
                    "cross": "Crs", "near": "Nr", "opposite": "Opp", "building": "Bldg"}}
HN_LABEL = {"France": ["N° ", "NO ", "No. ", "#"], "US": ["#", "No. "], "India": ["No. ", "#", "H.No. ", "Plot No. "]}
SEG_ADD = {"France": "France", "US": "USA", "India": "India"}
DIG = str.maketrans("olesabg", "0135486")
SYLL = json.loads((W / "p36/morph_train.json").read_text())
_HN = re.compile(r"^(\s*)(\d+)(\s*[A-Za-z]?\b)")
_LET = "abcdefghijklmnopqrstuvwxyz"


def _typo(w: str, rng, kind: str) -> str:
    if len(w) < 3:
        return w
    i = int(rng.integers(1, len(w) - 1))
    if kind == "del":
        return w[:i] + w[i + 1:]
    if kind == "ins":
        return w[:i] + _LET[rng.integers(26)] + w[i:]
    if kind == "sub":
        return w[:i] + _LET[rng.integers(26)] + w[i + 1:]
    if kind == "tr":
        return w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
    j = [k for k, ch in enumerate(w) if ch.lower() in "olesabg"]
    return w if not j else w[:j[0]] + w[j[0]].lower().translate(DIG) + w[j[0] + 1:]


def _pick(toks, rng, ok=lambda t: True):
    j = [k for k, t in enumerate(toks) if ok(t)]
    return None if not j else j[int(rng.integers(len(j)))]


def gen_name(n: str, c: str, R: dict, rng, vocab) -> str:
    r = lambda k: rng.random() < R[k]
    if r("gib"):                                     # brand name built from the 44 learned syllables
        return "".join(SYLL[int(i)] for i in rng.integers(len(SYLL), size=int(rng.integers(2, 4)))).capitalize()
    t = n.split()
    is_leg = lambda x: x.lower().strip("().,") in LEGAL_TOK
    if r("legal_drop") and any(map(is_leg, t)) and not all(map(is_leg, t)):
        t = [x for x in t if not is_leg(x)]
    elif r("legal_swap") and (k := _pick(t, rng, is_leg)) is not None:
        t[k] = LEGAL[c][int(rng.integers(len(LEGAL[c])))]
    elif r("legal_add") and not any(map(is_leg, t)):
        t.append(LEGAL[c][int(rng.integers(len(LEGAL[c])))])
    core = lambda x: not is_leg(x)
    if r("tokdrop") and sum(map(core, t)) >= 2:
        del t[_pick(t, rng, core)]
    if r("generic"):
        g = GENERIC[c][int(rng.integers(len(GENERIC[c])))]
        k = _pick(t, rng, is_leg)
        t.insert(len(t) if k is None else k, g)
    if r("insert") and vocab is not None:
        t.insert(int(rng.integers(len(t) + 1)), vocab[int(rng.integers(len(vocab)))])
    if r("reorder") and len(t) >= 2:
        k = int(rng.integers(len(t) - 1))
        t[k], t[k + 1] = t[k + 1], t[k]
    if r("dup") and t:
        k = int(rng.integers(len(t)))
        t.insert(k, t[k])
    if r("honor"):
        t.insert(0, HONOR[c][int(rng.integers(len(HONOR[c])))])
    if r("bracket") and t:
        t = [x.strip("()") for x in t] if any("(" in x for x in t) else t[:1] + [f"({t[1]})"] + t[2:] if len(t) > 2 else t
    for kind, key in (("digit", "digit"), ("del", "t_del"), ("ins", "t_ins"), ("sub", "t_sub"), ("tr", "t_tr")):
        if r(key) and (k := _pick(t, rng, lambda x: len(x) >= 3)) is not None:
            t[k] = _typo(t[k], rng, kind)
    if r("t_multi"):
        for _ in range(int(rng.integers(2, 4))):
            if (k := _pick(t, rng, lambda x: len(x) >= 3)) is not None:
                t[k] = _typo(t[k], rng, ("del", "ins", "sub", "tr")[int(rng.integers(4))])
    out = " ".join(t)
    if r("web"):
        out = "www." + "".join(x.lower() for x in t if core(x)) + (".fr" if c == "France" else ".com")
    return out


def gen_addr(a: str, c: str, R: dict, rng, dept: dict, shift: bool = False) -> str:
    """shift: decoy = the house number moved by 3-40 (a neighbouring business on the same street), never emptied."""
    r = lambda k: rng.random() < R[k]
    if shift:
        R = R | {"empty": 0.0, "nullseg": 0.0, "hn_drop": 0.0, "hn_small": 0.0, "hn_other": 0.0}
    if r("empty"):
        return ""
    if r("nullseg"):
        return "NULL"
    segs = [s.strip() for s in a.split(",") if s.strip()]
    if not segs:
        return a
    si = next((k for k, s in enumerate(segs) if _HN.match(s)), 0)
    st = segs[si]
    u = rng.random()
    m = _HN.match(st)
    if m:
        hn = int(m.group(2))
        if shift:
            st = f"{max(1, hn + int(rng.integers(3, 41)) * int(rng.choice([-1, 1])))}{st[m.end(2):]}"
            u = rng.random()
            if u < R["hn_label"]:
                st = HN_LABEL[c][int(rng.integers(len(HN_LABEL[c])))] + st.lstrip()
        elif u < R["hn_drop"]:
            st = st[m.end():].strip()
        elif u < R["hn_drop"] + R["hn_small"]:
            st = f"{max(1, hn + int(rng.choice([-2, -1, 1, 2])))}{st[m.end(2):]}"
        elif u < R["hn_drop"] + R["hn_small"] + R["hn_other"]:
            st = f"{int(rng.integers(1, 300))}{st[m.end(2):]}"
        elif u < R["hn_drop"] + R["hn_small"] + R["hn_other"] + R["hn_label"]:
            st = HN_LABEL[c][int(rng.integers(len(HN_LABEL[c])))] + st.lstrip()
    if r("abbr"):
        ab = STREET[c]
        st = " ".join(ab.get(w.lower(), w) for w in st.split())
    segs[si] = st
    if r("seg_typo"):
        k = int(rng.integers(len(segs)))
        w = segs[k].split()
        if (j := _pick(w, rng, lambda x: len(x) >= 4)) is not None:
            w[j] = _typo(w[j], rng, ("del", "ins", "sub", "tr")[int(rng.integers(4))])
            segs[k] = " ".join(w)
    if r("state") and len(segs) >= 3:              # the last non-street segment is the state / région slot
        k = max(k for k in range(len(segs)) if k != si)
        city = next((segs[j] for j in range(len(segs)) if j not in (si, k)), "")
        segs[k] = dept.get(city.lower(), segs[k]) if c == "France" else dept.get(segs[k].lower(), segs[k])
    if r("seg_drop") and len(segs) >= 2:
        del segs[_pick(segs, rng, lambda s: s is not segs[si]) or len(segs) - 1]
    if r("seg_add"):
        segs.append(SEG_ADD[c])
    if r("seg_reorder") and len(segs) >= 2:
        segs = [segs[i] for i in rng.permutation(len(segs))]
    return ", ".join(segs)


def state_map(c: str) -> dict:
    """France: city -> the département text S2/S3 most often put in the région slot for that city, learned from the
    TEST SX rows (label-free, test-side stats: allowed). US/India: the hand-written state name <-> code tables."""
    import normalize as NZ
    if c == "US":
        return {**{v: k.upper() for k, v in NZ.US_STATES.items()}, **{k: v.title() for k, v in NZ.US_STATES.items()}}
    if c == "India":
        return {vs[0]: k.upper() for k, vs in NZ.IN_STATES.items()} | {k: vs[0].title() for k, vs in NZ.IN_STATES.items()}
    regions = set(NZ.FR_REGIONS)
    rows = []
    for f in ("test_source2.parquet", "test_source3.parquet"):
        t = pq.read_table(W / f, columns=["business_address", "country"], filters=[("country", "=", "France")])
        rows.append(t.column("business_address").to_pandas())
    s = pd.concat(rows).str.split(",")
    s = s[s.str.len() == 3]
    last, city = s.str[-1].str.strip(), s.str[-2].str.strip()
    b = lambda x: x.astype("string[pyarrow]").str.lower().str.replace(r"[\p{P}\s]+", " ", regex=True).str.strip()
    ok = ~b(last).isin(regions) & ~last.str.contains(r"\d") & (last.str.len() > 2)
    d = pd.DataFrame({"city": city[ok].str.lower(), "dep": last[ok]})
    d = d.groupby(["city", "dep"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    d = d[d.n >= 5].drop_duplicates("city")
    return dict(zip(d.city, d.dep))


def s1_records(c: str) -> pd.DataFrame:
    """Raw S1 rows: France = every test S1; India / US = train fold-0 S1 (the LOCO frames' S1)."""
    if c == "France":
        return pq.read_table(W / "test_source1.parquet", filters=[("country", "=", "France")]).to_pandas()
    fo = pd.read_parquet(W / "s1_folds.parquet")
    keep = set(fo.entity_id[(fo.country == c) & (fo.fold == 0)])
    d = pq.read_table(W / "train_source1.parquet", filters=[("country", "=", c)]).to_pandas()
    return d[d.entity_id.isin(keep)].reset_index(drop=True)


def clean(names, addrs, c) -> np.ndarray:
    import block as B
    from full import FULL_CFG
    t = B.texts(pd.DataFrame({"business_name": names, "business_address": addrs, "country": c}), FULL_CFG)
    return (t["name"].astype(str) + " | " + t["addr"].astype(str)).to_numpy(dtype=object)


def neighbours(txt: np.ndarray, anchors: np.ndarray, rng, k=10) -> np.ndarray:
    """For each anchor, one other S1 among its top-k neighbours in one of three views, drawn per anchor: name char
    3-grams (name twins / siblings, w.p. .5), address words (same street / building, .3), full text words (.2)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    t = pd.Series(txt, dtype=str).str.partition(" | ")
    views = [(t[0], dict(analyzer="char_wb", ngram_range=(3, 3))), (t[2], dict(token_pattern=r"\S+")),
             (pd.Series(txt, dtype=str), dict(token_pattern=r"[^\s|]+"))]
    pick = rng.choice(3, size=len(anchors), p=[0.5, 0.3, 0.2])
    out = np.full(len(anchors), -1)
    for v, (col, kw) in enumerate(views):
        X = TfidfVectorizer(sublinear_tf=True, min_df=2, dtype=np.float32, **kw).fit_transform(col)
        XT = X.T.tocsr()
        idx = np.flatnonzero(pick == v)
        for s in range(0, len(idx), 2000):
            a = anchors[idx[s:s + 2000]]
            M = sp_matmul_topn(X[a], XT, top_n=k + 1, n_threads=4).tocsr()
            for i in range(len(a)):
                cols = M.indices[M.indptr[i]:M.indptr[i + 1]]
                cols = cols[cols != a[i]]
                if len(cols):
                    out[idx[s + i]] = cols[rng.integers(len(cols))]
        del X, XT
    return out


KIND_P = {"v1": [0.4, 0.25, 0.35], "v2": [0.1, 0.15, 0.75], "v1h": [0.4, 0.25, 0.35]}
# v2: fewer decoys (v1 lowers the standalone AUC). v1h = v1 with the OTHER country's op rates (strict LOCO: the target's
# rates were learned from its labels); France always gets the India/US mean.
OTHER = {"India": "US", "US": "India", "France": "France"}


def gen(c, n, ver="v2", seed=67):
    t0 = time.time()
    n = int(n)
    rng = np.random.default_rng(seed)
    d = s1_records(c)
    R, dept = RATES[OTHER[c] if ver.endswith("h") else c], state_map(c)
    vocab = pd.Series(" ".join(d.business_name.sample(min(len(d), 50_000), random_state=1)).split())
    vocab = vocab[(vocab.str.len() >= 3) & ~vocab.str.lower().isin(LEGAL_TOK)].to_numpy()
    s1_txt = clean(d.business_name.to_numpy(), d.business_address.to_numpy(), c)
    print(f"{c}: {len(d):,} S1, {len(dept):,} state-map keys; {time.time() - t0:.0f}s", flush=True)
    anc = rng.choice(len(d), size=min(n, len(d)), replace=False)
    pos = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]          # 1.5 synthetic copies per anchor
    nb = neighbours(s1_txt, anc, rng)
    neg_a = np.r_[anc, anc[rng.random(len(anc)) < 0.5]]
    hashn = d.business_address.str.contains(r"(?:^|,)\s*\d+", regex=True).to_numpy()
    kind = rng.choice(3, size=len(neg_a), p=KIND_P[ver])                # 0 hn-shift decoy, 1 name-swap decoy, 2 nbr
    kind[(kind == 0) & ~hashn[neg_a]] = 2
    neg_b = np.where(kind == 2, pd.Series(nb, index=anc).reindex(neg_a).to_numpy(), neg_a)
    ok = neg_b >= 0
    neg_a, neg_b, kind = neg_a[ok], neg_b[ok], kind[ok]
    src = np.r_[pos, neg_b]
    knd = np.r_[np.full(len(pos), -1), kind]
    nm = [gen_name(d.business_name.iat[i], c, R, rng, vocab) for i in src]
    for j in np.flatnonzero(knd == 1):                                  # one core word -> another S1's word
        t = nm[j].split()
        k = _pick(t, rng, lambda x: x.lower().strip("().,") not in LEGAL_TOK)
        if k is not None:
            t[k] = vocab[int(rng.integers(len(vocab)))]
        nm[j] = " ".join(t)
    ad = [gen_addr(d.business_address.iat[i], c, R, rng, dept, shift=k == 0) for i, k in zip(src, knd)]
    tb = clean(np.array(nm, dtype=object), np.array(ad, dtype=object), c)
    a1 = np.r_[pos, neg_a]
    y = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg_a), np.int8)]
    out = pd.DataFrame({"ta": s1_txt[a1], "tb": tb, "y": y, "p": np.float32(0.5),
                        "s1": d.entity_id.to_numpy()[a1], "sxsrc": d.entity_id.to_numpy()[src], "kind": knd.astype(np.int8)})
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"syn_{c}_{ver}.parquet", index=False)
    print(f"syn_{c}_{ver}: {len(out):,} pairs, pos {y.mean():.3f}; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    return out


def _sims(ta, tb):
    from rapidfuzz import fuzz, process
    p = lambda s: pd.Series(s, dtype=str).str.partition(" | ")
    a, b = p(ta), p(tb)
    cp = lambda x, y, sc: process.cpdist(x.to_numpy(), y.to_numpy(), scorer=sc, workers=4) / 100
    return pd.DataFrame({"name": cp(a[0], b[0], fuzz.ratio), "addr": cp(a[2], b[2], fuzz.ratio),
                         "tset": cp(pd.Series(ta, dtype=str), pd.Series(tb, dtype=str), fuzz.token_set_ratio),
                         "addr_empty": (b[2].str.len() == 0).astype(float)})


def sanity(c):
    """Real true pairs of country c (P65's 400k random train true pairs, cleaned) vs synthetic positives; real band
    false pairs (P13 train band file, y = 0) vs synthetic negatives. Quantiles of name / addr / token-set sims."""
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    s = pd.read_parquet(OUT / f"syn_{c}_v1.parquet")
    res = {}
    rt = pd.read_parquet(W / "p65/sm" / f"train_{c}.parquet").sample(20_000, random_state=0)
    sp = lambda x: pd.Series(x, dtype=str).str.rpartition(" | ")
    a, b = sp(rt.tgt), sp(rt.src)
    real_t = _sims(clean(a[0].to_numpy(), a[2].to_numpy(), c), clean(b[0].to_numpy(), b[2].to_numpy(), c))
    band = pd.read_parquet(OUT / f"train_{c}.parquet")
    real_f = _sims(*band[band.y == 0].sample(20_000, random_state=0)[["ta", "tb"]].to_numpy().T)
    real_bt = _sims(*band[band.y == 1].sample(20_000, random_state=0)[["ta", "tb"]].to_numpy().T)
    syn_t = _sims(*s[s.y == 1].head(20_000)[["ta", "tb"]].to_numpy().T)
    syn_f = _sims(*s[s.y == 0].head(20_000)[["ta", "tb"]].to_numpy().T)
    for k, v in {"real_true": real_t, "syn_pos": syn_t, "real_band_true": real_bt, "real_band_false": real_f,
                 "syn_neg": syn_f}.items():
        res[k] = {col: [round(x, 3) for x in v[col].quantile(q)] for col in ("name", "addr", "tset")} | \
                 {"addr_empty": round(float(v.addr_empty.mean()), 3)}
        print(c, k, res[k], flush=True)
    (Path(__file__).parent / f"sanity_{c}.json").write_text(json.dumps(res, indent=1))
    return res


def _selfcheck():
    rng = np.random.default_rng(0)
    R0 = {k: 0.0 for k in RATES["US"]}
    assert gen_name("Maison Ligue SARL", "France", R0, rng, None) == "Maison Ligue SARL"
    assert gen_addr("5 Rue Lafayette, Bordeaux, Nouvelle-Aquitaine", "France", R0, rng, {}) == "5 Rue Lafayette, Bordeaux, Nouvelle-Aquitaine"
    R1 = R0 | {"legal_drop": 1.0, "abbr": 1.0}
    assert gen_name("Maison Ligue SARL", "France", R1, rng, None) == "Maison Ligue"
    assert gen_addr("5 Rue Lafayette, Bordeaux, Nouvelle-Aquitaine", "France", R1, rng, {}).startswith("5 R Lafayette")
    assert gen_addr("x", "US", R0 | {"empty": 1.0}, rng, {}) == ""
    print("selfcheck ok")


if __name__ == "__main__":
    {"gen": gen, "sanity": sanity, "self": _selfcheck}[sys.argv[1]](*sys.argv[2:])
