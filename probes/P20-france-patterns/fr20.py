"""P20 France-pattern rules: record-level changes to rules.feats fields (id, country, core, web, is_web, state, city, hn,
places), general code keyed on nothing but the text. Each rule takes and returns a rules.feats frame.

R1 noise_core: the generator's French name-suffix fillers {fils, cie, associes, freres, frs} ('& Fils', '& Cie',
   'et Associés', 'Frères' / 'Frs') leave the name core, like normalize.GENERIC's 'services' / 'center' in US/India.
   Label-free evidence (P20 RESULT): SX cores that add / swap in one of these words keep the S1's house number
   81-88% of the time and take a sibling offset (hn + {3,4,5,7,9,11,13,21}) only 2-5%, vs 71-74% for the branch words
   (distribution / participations / holding / international). They are noise on true pairs, not sibling markers.
   'compagnie' / 'groupe' / 'developpement' are NOT in the list: they are also S1 org words or sibling branch words
   (sibling-offset share 17% / 51% / 64% when added). US / India: 10 S1 + 132 S2/S3 of 11M test records carry a listed word.
    .venv/bin/python probes/P20-france-patterns/fr20.py       # self-check
"""
import pandas as pd

NOISE = {"fils", "cie", "associes", "freres", "frs"}


def _strip(core: str) -> str:
    keep = [t for t in core.split() if t not in NOISE]
    return " ".join(keep) if keep else core


def noise_core(rf: pd.DataFrame) -> pd.DataFrame:
    rf = rf.copy()
    rf["core"] = rf.core.map(_strip).astype("str")
    return rf


RULES = {"noise": noise_core}


if __name__ == "__main__":
    rf = pd.DataFrame({"core": ["martin fils", "fils", "wallon cie", "green associes", "dupont freres", "garage", ""]})
    assert noise_core(rf).core.tolist() == ["martin", "fils", "wallon", "green", "dupont", "garage", ""]
    print("fr20 self-check ok")


# R2 (pair filter, P15's sib_rule.py on any France probability file): p := 0 when hn_x - hn_1 is a generator sibling
# offset {3,4,5,7,9,11,13,21}; France siblings keep / offset the hn like US ones (P20: 2-6% same hn), so the rule's
# US-holdout sign (+0.00097 on m2_big p) is the relevant one, not India's.
SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]


def sib_zero(d: pd.DataFrame, hn: pd.Series) -> pd.DataFrame:
    """d: s1, sx, p; hn: id -> house number ('' = none). Returns d with p = 0 on sibling-offset pairs."""
    h1, hx = hn.reindex(d.s1).fillna("").to_numpy(), hn.reindex(d.sx).fillna("").to_numpy()
    ok = (h1 != "") & (hx != "")
    delta = pd.to_numeric(pd.Series(hx[ok])).to_numpy() - pd.to_numeric(pd.Series(h1[ok])).to_numpy()
    sib = pd.Series(False, index=d.index)
    sib[ok] = pd.Series(delta).isin(SIB3).to_numpy()
    d = d.copy()
    d.loc[sib.to_numpy(), "p"] = 0.0
    return d


if __name__ == "__main__" and len(__import__("sys").argv) > 2:          # fr20.py sib IN.parquet OUT.parquet
    import sys
    from pathlib import Path
    W = Path(__file__).resolve().parents[2] / "data/work"
    hn = pd.concat([pd.read_parquet(W / f"rf_test_{s}_France.parquet", columns=["id", "hn"]) for s in ("s1", "sx")]
                   ).set_index("id").hn
    d = pd.read_parquet(sys.argv[2], columns=["s1", "sx", "p"])
    o = sib_zero(d, hn)
    assert len(o) == len(d) and (o.p <= d.p).all()
    o.to_parquet(sys.argv[3], index=False)
    print(f"sib: {int((o.p < d.p).sum()):,} of {len(d):,} pairs zeroed -> {sys.argv[3]}")


# R3 (feature de-shift): France's sx_margin (block-score margin of the SX's best S1 over its next S1) is compressed ~2x vs
# US/India (90th pct 0.24 vs 0.49 / 0.45) because France's common tokens sit at max train IDF (P14). Quantile-map it onto
# the US+India test distribution (over all blocked pairs), a label-free per-country standardisation; US/India map ~onto
# themselves. Tables: data/work/p20/sx_margin_quantiles.json (1001 quantiles per country, from p4/X test files).
def qm_tables(path, src="France", dst=("US", "India")):
    import json
    import numpy as np
    q = json.loads(open(path).read())
    return np.asarray(q[src]), np.mean([np.asarray(q[c]) for c in dst], axis=0)


def qmap(x, qsrc, qdst):
    import numpy as np
    y = np.interp(x, qsrc, qdst).astype(np.float32)
    return np.where(np.isnan(x), np.nan, y).astype(np.float32)
