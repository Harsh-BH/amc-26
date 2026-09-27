"""P51: Ditto-style structured serialization of one record side for the bge cross-encoder.

serialize(df, M) -> one string per row, built only from the repo's parsers (no country branching; the country label
only picks normalize's state table, as everywhere else):
    name: <name_core> | legal: <canonical legal forms> | hn: <house no> | street: <street_key> | city: <city>
    | area: <other address segments> | state: <state code> | pin: <postcode> | brand | noaddr | raw: <P3 name>
Empty fields are omitted; 'brand' (P36 gib synthetic brand name) and 'noaddr' (no address left after placeholder
removal) appear only when true. 'raw' goes last: longest_first truncation eats it first, and it only repeats the name.
France's '(11) RUE X' is fixed by france.paren_hn before parsing; 'N°' / 'R.' / 'Sàrl' fold to the same canonical
tokens India/US use (normalize.fold / STREET / LEGAL).
    .venv/bin/python probes/P51-ce-serialize/serialize.py      # self-check
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P36-iu-residual")]
import normalize as N
import france as FR
import gib

CANON = {"private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp", "company": "co",
         "etablissement": "ets", "etablissements": "ets"}


def _legal(norm: str) -> str:
    toks = norm.split()
    out = []
    for i, t in enumerate(toks):
        if N._is_legal(t, toks[i + 1] if i + 1 < len(toks) else ""):
            if t not in N.LEGAL:                     # a typo of a long form ('pirvate', 'limted')
                t = max(N._LONG_LEGAL, key=lambda w: N.ratio(t, w))
            t = CANON.get(t, t)
            if t not in out:
                out.append(t)
    return " ".join(out)


def serialize(df: pd.DataFrame, M) -> np.ndarray:
    """df: name (P3's Latin name), business_name, business_address (raw), country. M: gib syllables of the split."""
    name = df["name"].astype(str)
    core = N.name_core(name, df["country"])
    legal = N.name_norm(name).map(_legal)
    addr = FR.paren_hn(df["business_address"])
    af = N.address_fields(addr, df["country"])
    street = FR.street_key(addr)
    brand = gib.is_gib(df["business_name"].astype(str).tolist(), M)
    out = np.empty(len(df), dtype=object)
    for i, (c, lg, hn, st, pl, state, pin, an, b, raw) in enumerate(zip(
            core, legal, af.house_number, street, af.places, af.state, af.postcode, af.addr_norm, brand, name)):
        city = pl[-1] if pl else ""
        area = [s for s in an.split(", ") if s and s != state and s != city and not (hn and s.split()[0] == hn)
                and " ".join(sorted(set(s.split()) - FR._DROP)) != st]
        f = [("name", c), ("legal", lg), ("hn", hn), ("street", st), ("city", city), ("area", " , ".join(area)),
             ("state", state), ("pin", pin)]
        s = " | ".join(f"{k}: {v}" for k, v in f if v)
        s += " | brand" * bool(b) + " | noaddr" * (not an) + f" | raw: {raw}"
        out[i] = s.removeprefix(" | ")
    return out


if __name__ == "__main__":
    df = pd.DataFrame({
        "name": ["camera amicale sarl", "sharma traders pvt ltd", "acme holdings llc", "zephcalonex"],
        "business_name": ["Camera Amicale SARL", "M/s Sharma Traders Pvt Ltd", "Acme Holdings, LLC", "Zephcalonex"],
        "business_address": ["(8) R. Konrad Adenauer, Nantes, Pays de la Loire", "12, MG Road, Pune, Maharashtra",
                             "1810-A Abbott Street, Springfield, IL", "NULL"],
        "country": ["France", "India", "US", "US"]})
    got = serialize(df, ["zeph", "calo", "nex"])
    for g in got:
        print(g)
    assert got[0].startswith("name: camera amicale | legal: sarl | hn: 8 | street: adenauer konrad r"), got[0]
    assert "state: pays de la loire" in got[0]
    assert "area" not in got[1] and "legal: pvt ltd" in got[1] and "hn: 12" in got[1] and "state: mh" in got[1] and "city: pune" in got[1]
    assert "hn: 1810" in got[2] and "state: il" in got[2] and "legal: llc" in got[2]
    assert got[3].endswith("| brand | noaddr | raw: zephcalonex"), got[3]
    print("serialize self-check ok")
