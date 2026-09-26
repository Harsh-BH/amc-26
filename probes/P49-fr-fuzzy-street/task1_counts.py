import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from rapidfuzz.fuzz import ratio

ROOT = Path("../../")
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P28-addr-anchor")]
import normalize as N
from anchor import street_key, _NUM

def raw_street_key(addr: pd.Series) -> pd.Series:
    # Just basic extraction without fold/N._seg to see if case/accents matter
    pass

df = pd.read_parquet("../../data/work/p31/pairs_France.parquet")
df = df[df.argmax == True]
df = df[(df.core_1 == df.core_x) & (df.core_1 != "")]
df_same_city = df[df.city_1 == df.city_x]

# 1. Equal HN, unequal normalised street
df_hn = df_same_city[(df_same_city.hn_1 == df_same_city.hn_x) & (df_same_city.hn_1 != "")]
s1 = street_key(df_hn.addr_1)
sx = street_key(df_hn.addr_x)
unequal = s1 != sx

df_u = df_hn[unequal].copy()
df_u["s1"] = s1[unequal]
df_u["sx"] = sx[unequal]

# Let's test the fixes independently.
def test_fix(addr, use_glue=False, use_fr_street=False, use_no_fold=False):
    FR_STREET = N.STREET.copy()
    if use_fr_street:
        FR_STREET.update({"ch": "chem", "che": "chem", "gen": "gen", "general": "gen"})
    
    _DROP_FR = N.STOP | {"bis", "ter", "quater"}
    
    def _str(x):
        if use_glue:
            x = re.sub(r"\b([ld])['’\s]+", r"\1", x)
        segs = x.split(",")
        for i, seg in enumerate(segs):
            s = seg.strip()
            if N._UNIT.match(s): continue
            s = N._HN_LABEL.sub("", s.lstrip("#").strip())
            m = _NUM.match(s)
            if m:
                rest = N._basic(s[m.end():])
                if not rest and i + 1 < len(segs):
                    rest = N._basic(segs[i + 1])
                toks = [FR_STREET.get(t, t) for t in rest.split() if t not in N.FILLER]
                toks = [t[:-2] if N._ORD.match(t) else t for t in toks]
                rest_seg = " ".join(toks[1:] if toks[:1] == ["of"] else toks)
                return " ".join(sorted({t for t in rest_seg.split() if t not in _DROP_FR}))
        return ""
    
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_str).astype("str")

s1_glue = test_fix(df_u.addr_1, use_glue=True)
sx_glue = test_fix(df_u.addr_x, use_glue=True)
fixed_a = (s1_glue == sx_glue).sum()

s1_abbr = test_fix(df_u.addr_1, use_fr_street=True)
sx_abbr = test_fix(df_u.addr_x, use_fr_street=True)
fixed_b = (s1_abbr == sx_abbr).sum()

# For (c) accents/case, we can't really fix it because it's ALREADY fixed by normalize!
# Wait, if it's already fixed by normalize, then they don't have unequal normalised street!
# But let's check if there are cases that only differ by accents but have ratio >= 85
s1_both = test_fix(df_u.addr_1, use_glue=True, use_fr_street=True)
sx_both = test_fix(df_u.addr_x, use_glue=True, use_fr_street=True)
fixed_both = (s1_both == sx_both).sum()

# Compute fuzzy match on the fixed keys
fuzzy = np.array([ratio(a, b) >= 85 for a, b in zip(s1_both, sx_both)])
fixed_d = fuzzy.sum() - fixed_both

print("Total unequal:", len(df_u))
print("(a) Fixed by elision/apostrophe:", fixed_a)
print("(b) Fixed by abbreviations:", fixed_b)
print("(a)+(b) Fixed by both:", fixed_both)
print("(d) Generator typo (ratio >= 85):", fixed_d)

# SX with street but NO house number while S1 has one
df_no_hn = df_same_city[(df_same_city.hn_1 != "") & (df_same_city.hn_x == "")]
# With identical name
s1_no = test_fix(df_no_hn.addr_1, use_glue=True, use_fr_street=True)
sx_no = test_fix(df_no_hn.addr_x, use_glue=True, use_fr_street=True)

# how many of them have fuzzy street equal?
fuzzy_no = np.array([a != "" and b != "" and ratio(a, b) >= 85 for a, b in zip(s1_no, sx_no)])
print("SX no HN, identical name, fuzzy street equal:", fuzzy_no.sum())
