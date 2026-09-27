"""Touch counts of candidate France normalisation rules on test argmax pairs, per p band. Aggregates only."""
import re, sys, json, collections
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from anyascii import anyascii
W = "/home/harsh1/github-repos/amc-26/data/work/"
OUT = sys.argv[1]
def load(f, c="France", cols=("entity_id", "business_name", "business_address")):
    t = ds.dataset(W + f).to_table(filter=pc.field("country") == c, columns=list(cols))
    return t.to_pandas()
NA = re.compile(r"[^a-z0-9]+")
def asc(s):
    s = s.fillna("").astype(object)
    return s.map(lambda x: x if x.isascii() else anyascii(x)).str.lower()
def clean(s):   # == block.clean(anyascii) token list
    return asc(s).str.replace(NA.pattern, " ", regex=True).str.split()
res = {}
# ---- other countries: rates of the France patterns (safety of country-agnostic rules)
for c in ("US", "India"):
    d = load("test_source2.parquet", c, ("business_name", "business_address")).sample(300_000, random_state=0)
    a, n = d.business_address.fillna(""), d.business_name.fillna("")
    res[c] = {"deg_fused": float(a.str.contains(r"[°º№]\d").mean()), "deg_any": float(a.str.contains(r"[°º№]").mean()),
              "numletter_fused": float(asc(a).str.contains(r"\b\d+[a-z]\b").mean()),
              "dotted_name": float(clean(n).map(lambda t: any(len(x) == 1 and len(y) == 1 and x.isalpha() and y.isalpha()
                                                                for x, y in zip(t, t[1:]))).mean())}
    del d
print(res, flush=True)
s1 = load("test_source1.parquet").set_index("entity_id")
sx = pd.concat([load("test_source2.parquet"), load("test_source3.parquet")]).set_index("entity_id")
c1 = pd.read_parquet(W + "rf_test_s1_France.parquet", columns=["id", "core", "hn"]).set_index("id")
cx = pd.read_parquet(W + "rf_test_sx_France.parquet", columns=["id", "core", "hn"]).set_index("id")
p = pd.read_parquet(W + "p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"])
p = p.sort_values("p", ascending=False).drop_duplicates("sx")
q = p.join(s1, on="s1").join(sx, on="sx", rsuffix="_x").join(c1, on="s1").join(cx, on="sx", rsuffix="_x")
del s1, sx, c1, cx, p
q = q.dropna(subset=["business_name", "business_name_x"]).reset_index(drop=True)
bands = {"acc": q.p >= 0.985, "near": (q.p >= 0.5) & (q.p < 0.985), "low": (q.p >= 0.1) & (q.p < 0.5), "all": q.p >= 0}
B = {k: m.to_numpy() for k, m in bands.items()}
# ---- address canonical map (La Poste / AFNOR abbreviations seen in the data), applied to clean tokens
ADDR = {"r": "rue", "av": "avenue", "ave": "avenue", "aven": "avenue", "bd": "boulevard", "blvd": "boulevard",
        "boul": "boulevard", "bld": "boulevard", "all": "allee", "imp": "impasse", "rte": "route", "ch": "chemin",
        "chem": "chemin", "che": "chemin", "pl": "place", "crs": "cours", "qu": "quai", "res": "residence",
        "resid": "residence", "psg": "passage", "pass": "passage", "pas_": "passage", "sq": "square", "fg": "faubourg",
        "fbg": "faubourg", "faub": "faubourg", "st": "saint", "ste": "sainte", "apt": "appartement",
        "app": "appartement", "appt": "appartement", "bat": "batiment", "bt": "batiment", "esc": "escalier",
        "etg": "etage", "lot": "lotissement", "ham": "hameau", "rpt": "rond point", "prom": "promenade",
        "pro": "promenade", "esp": "esplanade", "trav": "traverse", "tra": "traverse", "mte": "montee",
        "chs": "chaussee", "ld": "lieu dit", "za": "zone artisanale", "zi": "zone industrielle", "gal": "general",
        "gen": "general", "mal": "marechal", "pdt": "president", "dr": "docteur", "pr": "professeur", "nd": "notre dame",
        "gd": "grand", "gde": "grande", "pt": "petit", "pte": "petite", "cte": "cite", "vla": "villa", "dom": "domaine"}
POSTNUM = {"b": "bis", "t": "ter", "q": "quater"}   # only right after a house number
QSEG = {"q": "quai"}                                  # 'q' opening a segment
def canon_addr(raw):
    out = []
    for x in asc(raw):
        x = re.sub(r"[°º№]|\bndeg", " ", x)                       # 'n°16' -> 'n 16'
        x = re.sub(r"\b(\d+)(bis|ter|quater|[btq])\b", r"\1 \2", x)  # '12b' -> '12 b'
        toks = []
        for seg in x.split(","):
            t = NA.sub(" ", seg).split()
            t = [w for i, w in enumerate(t) if not (w in ("n", "no") and i + 1 < len(t) and t[i + 1].isdigit())]
            for i, w in enumerate(t):
                if i and t[i - 1].isdigit() and w in POSTNUM:
                    toks.append(POSTNUM[w])
                elif i == 0 and w in QSEG:
                    toks.append(QSEG[w])
                else:
                    toks.extend(ADDR.get(w, w).split())
        out.append(toks)
    return out
a0, b0 = clean(q.business_address), clean(q.business_address_x)
a1, b1 = canon_addr(q.business_address), canon_addr(q.business_address_x)
def jac(u, v):
    su, sv = set(u), set(v)
    return len(su & sv) / max(len(su | sv), 1)
j0 = np.array([jac(u, v) for u, v in zip(a0, b0)]); j1 = np.array([jac(u, v) for u, v in zip(a1, b1)])
eq0 = np.array([set(u) == set(v) for u, v in zip(a0, b0)]); eq1 = np.array([set(u) == set(v) for u, v in zip(a1, b1)])
rawx = q.business_address_x.fillna("")
flags = {"addr_changed": j1 != j0, "addr_jacc_up": j1 > j0 + 1e-9, "addr_becomes_equal": eq1 & ~eq0,
         "deg_fused_sx": rawx.str.contains(r"[°º№]\d").to_numpy(),
         "numletter_fused_sx": asc(rawx).str.contains(r"\b\d+(?:bis|ter|quater|[btq])\b").to_numpy(),
         "paren_hn_sx": rawx.str.contains(r"(?:^|,\s*)\(\d+\)").to_numpy(),
         "hn_missing_sx_only": ((q.hn.fillna("") != "") & (q.hn_x.fillna("") == "")).to_numpy()}
# ---- names
n0, m0 = clean(q.business_name), clean(q.business_name_x)
def merge1(t):
    o, run = [], []
    for w in t + [""]:
        if len(w) == 1 and w.isalpha():
            run.append(w); continue
        if run:
            o.append("".join(run)); run = []
        if w:
            o.append(w)
    return o
LEG = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "eirl", "scop", "scp", "selarl", "gie", "sca", "scs",
       "selas", "selafa", "selca", "scm", "scpi", "earl", "gaec", "scea", "sccv", "sem", "spl", "scic", "sep", "sasp"}
dotted = np.array([any(len(x) == 1 and len(y) == 1 and x.isalpha() and y.isalpha() for x, y in zip(t, t[1:])) for t in m0])
core_eq = (q.core.fillna("") == q.core_x.fillna("")).to_numpy() & (q.core.fillna("") != "").to_numpy()
L1 = [set(merge1(t)) & LEG for t in n0]; Lx = [set(merge1(t)) & LEG for t in m0]
lswap = np.array([bool(u) and bool(v) and not (u & v) for u, v in zip(L1, Lx)])
ldrop = np.array([bool(u) != bool(v) for u, v in zip(L1, Lx)])
first = lambda t: t[0] if t else ""
ste1 = np.array([first(t) in ("ste", "societe", "soc", "sté") for t in n0]); stex = np.array([first(t) in ("ste", "societe", "soc") for t in m0])
asso = lambda t: bool({"association", "asso", "assoc", "ass"} & set(t))
names = {"dotted_legal_sx": dotted, "core_eq": core_eq, "legal_swap": lswap, "legal_swap&core_eq": lswap & core_eq,
         "legal_dropadd": ldrop, "legal_dropadd&core_eq": ldrop & core_eq,
         "first_ste_or_societe_s1": ste1, "first_ste_or_societe_sx": stex,
         "ste_mismatch": ste1 != stex, "asso_any": np.array([asso(u) or asso(v) for u, v in zip(n0, m0)]),
         "asso_one_side": np.array([asso(u) != asso(v) for u, v in zip(n0, m0)])}
LEGAL_NOT_IN_SRC = {"selas", "selafa", "selca", "scm", "scpi", "earl", "gaec", "scea", "sccv", "sem", "spl", "scic", "sep",
                    "sasp", "societe", "ste", "association", "asso", "sel", "selurl", "saem", "sica", "cuma"}
tokcount = collections.Counter()
for t in list(n0[B["all"]]) if False else []:
    pass
allnames = pd.concat([pd.Series(n0), pd.Series(m0)])
for t in allnames:
    tokcount.update(set(merge1(t)) & LEGAL_NOT_IN_SRC)
res["legal_not_in_src_record_counts"] = dict(tokcount.most_common())
for k, m in B.items():
    r = {"n": int(m.sum())}
    for f, v in {**flags, **names}.items():
        r[f] = int(v[m].sum())
    r["mean_addr_jacc_before"], r["mean_addr_jacc_after"] = float(j0[m].mean()), float(j1[m].mean())
    res[k] = r
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
