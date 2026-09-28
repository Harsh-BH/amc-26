"""P64: legal-form signature of a business name, and S1-vs-SX legal-form agreement features.

sig(name) = sorted '+'-joined canonical legal tokens found by normalize's own detector (_is_legal: LEGAL + typos of the
long forms), e.g. 'Acme Pvt. Limited' -> 'ltd+pvt', 'Foo L.L.C.' -> 'llc', 'Bar SARL' -> 'sarl', no form -> ''.
    .venv/bin/python probes/P64-legal-form-feat/lf.py        # self-check
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
import normalize as N  # noqa: E402
from rapidfuzz.fuzz import ratio  # noqa: E402

CANON = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt",
         "etablissement": "ets", "etablissements": "ets"}
LONG = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co"}


def _canon(t: str) -> str:
    if t in N.LEGAL:
        return CANON.get(t, t)
    return LONG[max(LONG, key=lambda w: ratio(t, w))]          # a typo of a long form ('pirvate', 'Limted')


def _sig(x: str) -> str:
    toks = x.split()
    leg = [t for i, t in enumerate(toks) if not (i == 0 and t == "co" and len(toks) > 1)      # 'Co-op ...', 'Co Name'
           and N._is_legal(t, toks[i + 1] if i + 1 < len(toks) else "")]
    if len(leg) == len(toks):          # the whole name is 'legal' tokens: an acronym ('PC' = the initials of a 2-word core) or a bare form
        return ""
    return "+".join(sorted({_canon(t) for t in leg}))


def sig(names: pd.Series) -> pd.Series:
    return pd.Series([_sig(x) for x in N.name_norm(names)], index=names.index, dtype="str")


def pair_class(s1: pd.Series, sx: pd.Series) -> np.ndarray:
    """0 both none, 1 same, 2 S1 only, 3 SX only, 4 both present and different, 5 overlap (share a token, differ)."""
    a, b = s1.to_numpy(object), sx.to_numpy(object)
    ea, eb = a == "", b == ""
    ov = np.array([bool(set(u.split("+")) & set(v.split("+"))) for u, v in zip(a, b)])
    return np.select([ea & eb, a == b, eb, ea, ov], [0, 1, 2, 3, 5], 4).astype(np.int8)


if __name__ == "__main__":
    t = sig(pd.Series(["Acme Pvt. Limited", "Foo L.L.C.", "Bar SARL", "Baz", "Qux Pirvate Ltd", "Co-op Stores Inc.",
                       "Société Dupont S.A.S.", "Company", "PC", "S A S U Dupont"]))
    assert t.tolist() == ["ltd+pvt", "llc", "sarl", "", "ltd+pvt", "inc", "sas", "", "", "sasu"], t.tolist()
    k = pair_class(pd.Series(["", "llc", "llc", "", "llc", "ltd+pvt"]), pd.Series(["", "llc", "", "inc", "inc", "ltd"]))
    assert k.tolist() == [0, 1, 2, 3, 4, 5], k.tolist()
    print("lf self-check ok")
