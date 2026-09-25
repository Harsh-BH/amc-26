"""France layer over src/normalize.py (P1). normalize.py already folds accents (anyascii), drops the French legal
forms (SARL/SAS/SASU/SA/EURL/SCI/SNC/EI/EIRL/Ets/Établissements), the articles de/des/du/la/le/les/l/d, maps the
French street types (rue->r, av/avenue->ave, bd->blvd, impasse->imp, allée->all, chemin->chem, route->rte,
faubourg->fg, quai->qu, cours->crs, sainte->ste, saint->st) and régions/départements -> région. Verified below.

What it misses (label-free diagnostics on the France test rows, probes/P6-france/diag_norm.py; nothing fitted):
  1. names: St/Ste vs Saint/Sainte in names (the address side already maps them)  -> FR_NAME_CANON.
     The French suffix constructs '& Fils', 'Frères'/'Frs', '& Cie'/'Compagnie', 'et Associés' are added / dropped
     between sources like 'Services'/'Center' in US, but France S1 names also carry them, so dropping them merges
     distinct S1 into rule ties (rule stage: +31.6k / -18.4k pairs, S1 coverage down) -> FR_FILLER, opt-in only.
  2. house numbers written '(520) RUE DE CASSEL' (1.6% of France S2/S3, 0% in US/India) were not parsed.
  3. French segments start with the street type ('R. VICTOR HUGO', 'PLACE DE LA NATION'): 3% of France S2/S3 have
     one without a house number, and normalize took it as a place -- and, with train place counts all 0 for
     France, as the city (alphabetical max: 'r lyderic' > 'lille').  -> FR_NONPLACE_FIRST.
Everything is keyed on the folded country label 'france', so US / India output is unchanged (asserted below).

    .venv/bin/python probes/P6-france/france.py        # self-check on handwritten examples
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
import normalize as N

# domain: suffix constructs of French trade names ('Martin & Fils', 'Dupont Frères', 'Durand & Cie', 'X et Associés')
FR_FILLER = {"fils", "freres", "frs", "cie", "compagnie", "associes"}
FR_NAME_CANON = {"st": "saint", "ste": "sainte"}          # names keep the long form, as websites do
# domain: first word of a French street / building segment (post normalize._seg canonical forms)
FR_NONPLACE_FIRST = {"r", "ave", "blvd", "imp", "all", "chem", "rte", "fg", "qu", "crs", "pl", "sq", "cite", "rpt",
                     "rond", "residence", "res", "passage", "sentier", "voie", "venelle", "ruelle", "parvis",
                     "esplanade", "promenade", "mail", "lotissement", "lot", "hameau", "lieu", "ld", "za", "zi", "zac",
                     "zone", "parc", "domaine", "square", "maison", "immeuble", "batiment", "bat", "espace", "centre",
                     "center", "rdc", "rez", "bp", "cs", "tsa"}
_FR = "france"


def _fr_core(core: str, filler: bool) -> str:
    toks = [FR_NAME_CANON.get(t, t) for t in core.split()]
    keep = [t for t in toks if not (filler and t in FR_FILLER)]
    return " ".join(keep or toks)


def name_core(s: pd.Series, country: pd.Series, filler: bool = False) -> pd.Series:
    """normalize.name_core, then for France rows: St/Ste -> Saint/Sainte; filler=True also drops FR_FILLER."""
    core = N.name_core(s, country)
    fr = (N.fold(country) == _FR).to_numpy()
    if fr.any():
        core = core.copy()
        core[fr] = core[fr].map(lambda x: _fr_core(x, filler))
    return core


def _paren_hn(addr: pd.Series) -> pd.Series:
    """'(520) RUE X' -> '520 RUE X' (a parenthesised number opening a segment). US / India: ~0 rows."""
    return addr.fillna("").astype(str).str.replace(r"(^|,\s*)\((\d+)\)", r"\1\2", regex=True)


def address_fields(addr: pd.Series, country: pd.Series, freq: dict | None = None) -> pd.DataFrame:
    """normalize.address_fields with '(n)' house numbers parsed, and for France rows street / building segments
    dropped from places before the city is chosen (same choice rule as normalize: train freq, else last place)."""
    f = N.address_fields(_paren_hn(addr), country, freq)
    fr = (N.fold(country) == _FR).to_numpy()
    if fr.any():
        pl = f.places[fr].map(lambda r: [x for x in r if x.split(" ", 1)[0] not in FR_NONPLACE_FIRST])
        if freq is None:
            ck = pl.map(lambda r: r[-1] if r else "")
        else:
            ck = pl.map(lambda r: max(r, key=lambda x: (freq.get(x, 0), x)) if r else "")
        f = f.copy()
        f.loc[fr, "places"] = pl
        f.loc[fr, "city_key"] = ck
    return f


def feats(d: pd.DataFrame, freq: dict | None = None) -> pd.DataFrame:
    """rules._feats with the France layer (same columns): id, country, core, web, is_web, state, city, hn, places."""
    import rules
    import translit as T
    name, addr = T.to_latin(d.business_name), T.to_latin(d.business_address)
    a = address_fields(addr, d.country, rules.freq() if freq is None else freq)
    return pd.DataFrame({"id": d.entity_id.to_numpy(), "country": d.country.to_numpy(),
                         "core": name_core(name, d.country).to_numpy(), "web": N.website_core(name).to_numpy(),
                         "is_web": N.is_website(name).to_numpy(), "state": a.state.to_numpy(),
                         "city": a.city_key.to_numpy(), "hn": a.house_number.to_numpy(),
                         "places": a.places.map("|".join).to_numpy()})


def _check():
    S = lambda *v: pd.Series(list(v), dtype="str")
    fr = lambda n: S(*["France"] * n)
    # --- names: handwritten S1 form vs generator-style variants (case, accents, punctuation, legal form, filler)
    names = S("Boulangerie Dupont SARL", "SARL BOULANGERIE DUPONT", "boulangerie dupont s.a.r.l.",
              "Martin & Fils SAS", "MARTIN ET FILS", "Ets Martin", "Établissements Martin & Cie",
              "Pharmacie St Michel", "Pharmacie Saint-Michel", "PHARMACIE STE ANNE EURL", "Pharmacie Sainte Anne",
              "Cabinet Durand et Associés SCP", "Cabinet Durand", "Garage Moreau Frères", "Garage Moreau Frs",
              "Crêperie de l'Océan", "CREPERIE DE L'OCEAN", "Café du Théâtre (France) SA", "Fils")
    got = list(name_core(names, fr(len(names)), filler=True))
    assert got == ["boulangerie dupont"] * 3 + ["martin"] * 4 + ["pharmacie saint michel"] * 2 + \
        ["pharmacie sainte anne"] * 2 + ["cabinet durand"] * 2 + ["garage moreau"] * 2 + \
        ["creperie locean"] * 2 + ["cafe theatre", "fils"], got
    assert list(name_core(S("Martin & Fils SAS", "Pharmacie St Michel"), fr(2))) == ["martin fils", "pharmacie saint michel"]
    # the France layer never touches other countries
    us = S("Saint Mary Hospital", "Johnson & Sons Co", "St Louis Bread Fils")
    assert list(name_core(us, S("US", "US", "US"))) == list(N.name_core(us, S("US", "US", "US")))
    # --- addresses
    a = S("23 Boulevard de la Renaissance, Saint-Nazaire, Pays de la Loire",
          "(23) BD DE LA RENAISSANCE, ST NAZAIRE, Loire-Atlantique",
          "Loire-Atlantique, ST-NAZAIRE, 23 bd. de la Renaissance",
          "R. VICTOR HUGO, LILLE, Nord", "Lille, Place de la République",
          "Maison des Associations, 12 Rue Colbert, Lille, Hauts-de-France",
          "N°16 Allée du Port, Pessac, Gironde", "4 IMP. DES LILAS, MERIGNAC")
    f = address_fields(a, fr(len(a)))
    assert list(f.house_number) == ["23", "23", "23", "", "", "12", "16", "4"], list(f.house_number)
    assert list(f.state) == ["pays de la loire"] * 3 + ["hauts de france", "", "hauts de france",
                                                         "nouvelle aquitaine", ""], list(f.state)
    assert list(f.city_key) == ["st nazaire"] * 3 + ["lille"] * 3 + ["pessac", "merignac"], list(f.city_key)
    base = N.address_fields(a, fr(len(a)), {})
    assert base.city_key[3] == "r victor hugo" and base.house_number[1] == "", "the gaps this layer fixes"
    # train place counts know no French city: the France fallback must still not pick the street
    assert address_fields(a[3:5], fr(2), {"lille": 0}).city_key.tolist() == ["lille", "lille"]
    u = S("1822 Perkins Road, Unit # 533, Stillwater, OK", "Rue Street, Saint Paul, MN")
    assert address_fields(u, S("US", "US")).equals(N.address_fields(u, S("US", "US")))
    print("france self-check ok")


if __name__ == "__main__":
    _check()
