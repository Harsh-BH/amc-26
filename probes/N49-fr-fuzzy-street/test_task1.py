import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio

sys.path[:0] = ["../../code/business_entity_resolution/src", "../../probes/P28-addr-anchor"]
import normalize as N
from anchor import street_key, _street

df = pd.read_parquet("../../data/work/p31/pairs_France.parquet", columns=["addr_1", "addr_x", "hn_1", "hn_x", "core_1", "core_x", "city_1", "city_x", "argmax"])
df = df[df.argmax == True]
df = df[(df.core_1 == df.core_x) & (df.core_1 != "")]
df = df[(df.hn_1 == df.hn_x) & (df.hn_1 != "")]
df = df[df.city_1 == df.city_x]

s1 = street_key(df.addr_1)
sx = street_key(df.addr_x)
unequal = s1 != sx
df_u = df[unequal].copy()
print("Total unequal normalised street:", len(df_u))

# Now, implement a fixed street key.
FR_STREET = N.STREET.copy()
FR_STREET.update({
    "ch": "chem", "che": "chem", "chemin": "chem",
    "gen": "gen", "general": "gen",
    "av": "ave", "avenue": "ave",
    "r": "r", "rue": "r",
    "bd": "blvd", "boulevard": "blvd",
    "all": "all", "allee": "all",
    "imp": "imp", "impasse": "imp",
    "crs": "crs", "cours": "crs",
    "pl": "pl", "place": "pl",
    "rte": "rte", "route": "rte",
    "st": "st", "saint": "st", "sainte": "ste"
})
_DROP_FR = N.STOP | {"bis", "ter", "quater", "l", "d"}
import re

def _street_fr(x: str) -> str:
    x = re.sub(r"\b([ld])['’\s]+", r"\1", x)
    segs = x.split(",")
    for i, seg in enumerate(segs):
        s = seg.strip()
        if N._UNIT.match(s): continue
        s = N._HN_LABEL.sub("", s.lstrip("#").strip())
        from anchor import _NUM
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

def french_street_key(addr: pd.Series) -> pd.Series:
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_street_fr).astype("str")

f1 = french_street_key(df_u.addr_1)
fx = french_street_key(df_u.addr_x)

both_eq = (f1 == fx)
print("(a)/(b)/(c) Fixed by French street key:", both_eq.sum())

fuz = np.array([ratio(a, b) >= 85 for a, b in zip(f1, fx)])
print("(d) Fixed by fuzzy match:", fuz.sum())

