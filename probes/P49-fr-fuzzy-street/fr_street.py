import sys
from pathlib import Path
import pandas as pd
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
    "av": "ave"
})

_DROP_FR = N.STOP | {"bis", "ter", "quater"}

def _street_fr(x: str) -> str:
    # Glue elisions so l'argonne -> largonne and d'agen -> dagen
    x = re.sub(r"\b([ld])['’\s]+", r"\1", x)
    
    segs = x.split(",")
    # 1. Try to find the segment with the house number (anchor.py logic)
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
            
    # 2. Fallback: if no house number, find the segment with a street type!
    # N._STREET_TYPES has "st", "rd", "r", "ave", "blvd", "chem", "imp", "rte", "all" etc.
    # We should normalize segments and see if they contain a street type.
    best = ""
    for i, seg in enumerate(segs):
        s = seg.strip()
        rest = N._basic(s)
        toks = [FR_STREET.get(t, t) for t in rest.split() if t not in N.FILLER]
        toks = [t[:-2] if N._ORD.match(t) else t for t in toks]
        rest_seg = " ".join(toks[1:] if toks[:1] == ["of"] else toks)
        cleaned = " ".join(sorted({t for t in rest_seg.split() if t not in _DROP_FR}))
        if cleaned:
            # Does it have a street type?
            if any(t in N._STREET_TYPES for t in cleaned.split()):
                return cleaned
            if not best:
                best = cleaned
    return best

def french_street_key(addr: pd.Series) -> pd.Series:
    f = N.fold(addr).str.replace(N._PLACEHOLDER.pattern, " ", regex=True).str.replace(N._POBOX.pattern, " ", regex=True)
    return f.map(_street_fr).astype("str")

