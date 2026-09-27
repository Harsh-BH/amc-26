"""Label-free checks of generator operators vs our normaliser, test split, per country. Aggregates only.
A website generator: accents folded or deleted?   B alias markers the _ALIAS regex misses
C French elision tokens   D house-number formats house_number() cannot parse   E honorific-like first tokens"""
import re, sys
from collections import Counter
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
from anyascii import anyascii

ROOT = Path("/home/harsh1/github-repos/amc-26")
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import normalize as N

W = ROOT / "data/work"
C = sys.argv[1]


def load(src, cols, cap=None):
    t = ds.dataset(W / f"test_{src}.parquet").to_table(columns=cols, filter=ds.field("country") == C)
    df = t.to_pandas()
    return df.head(cap) if cap else df


s1 = load("source1", ["business_name", "business_address"])
sx = pd.concat([load(s, ["business_name", "business_address"], 400_000) for s in ("source2", "source3")], ignore_index=True)
print(f"== {C}: S1 {len(s1):,}  SX sample {len(sx):,}")

# A: website core vs squashed S1 core, accents folded vs deleted
core = N.name_core(s1.business_name, pd.Series(C, index=s1.index))
fold_sq = set(core.str.replace(" ", "", regex=False))
nm = s1.business_name.str.lower()
deleted = nm.map(lambda x: re.sub(r"[^a-z0-9 ]", "", x))          # non-ASCII DELETED, not transliterated
del_sq = set(N.name_core(deleted, pd.Series(C, index=s1.index)).str.replace(" ", "", regex=False))
web = sx.business_name[N.is_website(sx.business_name)]
wc = N.website_core(web)
wc = wc[wc != ""]
hit_f, hit_d = wc.isin(fold_sq), wc.isin(del_sq)
print(f"A websites {len(wc):,}: exact core hit folded {hit_f.mean():.3f}  deleted {hit_d.mean():.3f}  "
      f"deleted-only {(hit_d & ~hit_f).mean():.3f}  either {(hit_d | hit_f).mean():.3f}")

# B: alias markers
fx = N.fold(sx.business_name)
pats = {"d.b.a": r"\bd\.b\.a\b", "f.k.a": r"\bf\.k\.a\b", "a.k.a": r"\ba\.k\.a\b", "t.a": r"\bt\.a\.?(?=\s)",
        "dba": r"\bdba\b", "f/k/a": r"f/k/a", "formerly known as": r"formerly known as", "doing business as": r"doing business as",
        "trading as": r"trading as", "anciennement": r"\banciennement\b", "dite": r"\bdite?\b", "ex-": r"\bex-",
        "sous le nom": r"sous le nom", "enseigne": r"\benseigne\b", "nee": r"\bnee\b", "sys": r"\bsys\b"}
print("B alias markers per 10k SX:", {k: round(fx.str.contains(p, regex=True).mean() * 1e4, 1) for k, p in pats.items()})
missed = fx.str.contains(r"\bd\.b\.a\b|\bf\.k\.a\b|\ba\.k\.a\b", regex=True)
print("   dotted alias not resolved by name_norm (share of SX):", round(missed.mean(), 5))

# C: elision
el = fx.str.contains(r"\b[ldjmnst]'[a-z]", regex=True)
el1 = N.fold(s1.business_name).str.contains(r"\b[ldjmnst]'[a-z]", regex=True)
v1 = Counter(t for x in N.name_norm(s1.business_name) for t in x.split())
toks = [t for x in N.name_norm(sx.business_name) for t in x.split()]
glued = sum(1 for t in toks if len(t) > 4 and t[0] in "ld" and t not in v1 and v1.get(t[1:], 0) >= 20)
print(f"C elision apostrophe: S1 {el1.mean():.4f} SX {el.mean():.4f}; SX tokens = d/l + S1 word, token unseen in S1: "
      f"{glued:,} of {len(toks):,} ({glued / max(len(toks),1):.5f})")

# D: house-number formats
ax = N.fold(sx.business_address)
hn = N.house_number(sx.business_address, pd.Series(C, index=sx.index))
lead_num = ax.str.contains(r"(?:^|,)\s*\(?\s*(?:#|no\.?|n|h\.?\s*no\.?|d\.?\s*no\.?|door no\.?|pl|plt|plot)?\s*\d", regex=True)
forms = {"(n)": r"(?:^|,)\s*\(\d+\)", "h.no": r"\bh\.\s*no\b", "d.no": r"\bd\.\s*no\b", "plt/pl": r"\b(?:plt|pl)\b\.?\s*(?:no)?\.?\s*\d",
         "zero-pad": r"(?:^|[\s,#])0\d+\b", "#": r"#\s*\d", "n deg": r"\bn\s+\d", "bis/ter": r"\b\d+\s*(?:bis|ter|quater)\b"}
for k, p in forms.items():
    m = ax.str.contains(p, regex=True)
    print(f"D {k:9s}: {m.mean() * 1e4:7.1f} per 10k SX, house_number empty among them {(hn[m] == '').mean() if m.any() else 0:.3f}")

# E: first-token asymmetry of names (honorifics / prefixes the generator adds)
f1 = Counter(x.split()[0] for x in N.name_norm(s1.business_name) if x)
fxx = Counter(x.split()[0] for x in N.name_norm(sx.business_name) if x)
n1, nx = len(s1), len(sx)
rows = sorted(((t, fxx[t] / nx, f1[t] / n1) for t in fxx if fxx[t] >= 100), key=lambda r: -(r[1] + 1e-5) / (r[2] + 1e-5))
print("E first-token SX-over:", ", ".join(f"{t} {a*100:.2f}%/{b*100:.2f}%" for t, a, b in rows[:25]))
print("   in HONORIFIC:", [t for t, _, _ in rows[:25] if t in N.HONORIFIC])
