"""France free pairs with identical name core, same city, same hn but a different street string (sub77): what the street
difference is. type-only (street-type word / abbreviation, 'de la' fillers), typo (edit distance <= 2 on the rest), other."""
import csv, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles")]
import p70
from rapidfuzz.distance import Levenshtein
W = ROOT / "data/work"
TYPES = set("rue r av ave avenue bd bld blvd boulevard ch chem chemin q quai imp impasse all allee allees rte route pl place pass psg passage "
            "crs cours sq square res residence lot lotissement ham hameau cite voie sente sentier rpt rond point parvis prom promenade esplanade "
            "fg faubourg mail zi za zac zone port quartier qua clos chaussee montee cour villa venelle traverse".split())
FILL = set("de du des la le les l d a au aux et".split())
strip = lambda s: " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in TYPES and t not in FILL)


def classify(s1, sx):
    a, b = s1.split("|", 1)[-1], sx.split("|", 1)[-1]
    if a == b:
        return "same"
    ra, rb = strip(a), strip(b)
    if ra and ra == rb:
        return "type_only"
    if ra and rb and Levenshtein.distance(ra, rb) <= 2:
        return "typo"
    return "other"


if __name__ == "__main__":
    base = sys.argv[1]
    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
    r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
    rx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
    for c in ("core", "hn", "city"):
        d[c + "1"] = r1[c].reindex(d.s1).fillna("").to_numpy(); d[c + "x"] = rx[c].reindex(d.sx).fillna("").to_numpy()
    m = pd.read_csv(ROOT / "output" / base / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(r1.index)]
    owned = {x for v in m.matched_entity_ids if v for x in v.split(",")}
    g = d[(d.core1 == d.corex) & (d.core1 != "") & (d.city1 == d.cityx) & (d.city1 != "") & (d.hn1 == d.hnx) & (d.hn1 != "") & ~d.sx.isin(owned)].copy()
    a1 = p70._raw("test", set(g.s1), [1], ["business_address", "country"]).business_address
    ax = p70._raw("test", set(g.sx), [2, 3], ["business_address", "country"]).business_address
    fr_ = lambda s: pd.Series(["France"] * len(s), index=s.index)
    k1 = p70._street(a1.reindex(g.s1).fillna("").reset_index(drop=True), fr_(g.reset_index(drop=True)))
    kx = p70._street(ax.reindex(g.sx).fillna("").reset_index(drop=True), fr_(g.reset_index(drop=True)))
    g["cls"] = [classify(a, b) if a and b else ("no_street_s1" if not a else "no_street_sx") for a, b in zip(k1, kx)]
    g.to_parquet(W / f"p70/street_pool_{base}.parquet", index=False)
    print("free identical-core same-city same-hn pairs:", len(g))
    print(g.groupby("cls").agg(n=("sx", "size"), p_med=("p", "median")).round(3).to_string())
    for c in ("type_only", "typo", "other", "same"):
        e = g[g.cls == c].head(4)
        print(c, [(a1.get(s, ""), ax.get(x, "")) for s, x in zip(e.s1, e.sx)])
