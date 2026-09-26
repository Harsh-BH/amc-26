import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path("../../")
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P28-addr-anchor")]
import normalize as N
from anchor import _NUM

FR_STREET = N.STREET.copy()
FR_STREET.update({
    "ch": "chem",
    "che": "chem",
    "gen": "gen",
    "general": "gen",
})

_DROP_FR = N.STOP | {"bis", "ter", "quater"}

def _street_fr(x: str) -> str:
    # 1. Lowercase and fold
    # x is already folded by N.fold
    
    # We apply the glue BEFORE splitting.
    # Wait, fold() removes some stuff but keeps spaces and apostrophes (except degrees etc).
    # anyascii is applied.
    # Let's glue \b([ld])['’\s]+ -> \1
    x = re.sub(r"\b([ld])['’\s]+", r"\1", x)
    
    segs = x.split(",")
    for i, seg in enumerate(segs):
        s = seg.strip()
        if N._UNIT.match(s):
            continue
        s = N._HN_LABEL.sub("", s.lstrip("#").strip())
        m = _NUM.match(s)
        if m:
            rest = N._basic(s[m.end():])
            if not rest and i + 1 < len(segs):
                rest = N._basic(segs[i + 1])
            
            # Apply N._seg logic but with FR_STREET
            toks = [FR_STREET.get(t, t) for t in rest.split() if t not in N.FILLER]
            toks = [t[:-2] if N._ORD.match(t) else t for t in toks]
            rest_seg = " ".join(toks[1:] if toks[:1] == ["of"] else toks)
            
            return " ".join(sorted({t for t in rest_seg.split() if t not in _DROP_FR}))
    return ""

def french_street_key(addr: pd.Series) -> pd.Series:
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_street_fr).astype("str")

df = pd.read_csv("diff.csv").fillna("")
print("Total initial mismatches:", len(df))

s1_new = french_street_key(df.addr_1)
sx_new = french_street_key(df.addr_x)

match = s1_new == sx_new
print("Fixed by new key:", match.sum())

df["s1_new"] = s1_new
df["sx_new"] = sx_new
df["fixed"] = match

df[~match].to_csv("diff_new.csv", index=False)
