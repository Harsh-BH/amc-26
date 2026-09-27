"""P40: are name traits decided per (entity, source)? Same-entity same-source copy pairs vs random same-source pairs.
Aggregates only."""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import world
import normalize as NZ

s1, sx, t = world()
t = t.assign(src=t.sx.str[:2], c=s1.country.reindex(t.s1).to_numpy())
nm = sx.business_name.reindex(t.sx).fillna("").to_numpy()
ad = sx.business_address.reindex(t.sx).fillna("").to_numpy()
n1 = s1.business_name.reindex(t.s1).fillna("").to_numpy()
LEG = NZ.LEGAL
def legal(s):
    return " ".join(w for w in re.findall(r"[a-z]+", s.lower()) if w in LEG)
tr = pd.DataFrame({
    "upper": [x == x.upper() and any(ch.isalpha() for ch in x) for x in nm],
    "lower": [x == x.lower() and any(ch.isalpha() for ch in x) for x in nm],
    "legal": [legal(x) for x in nm],
    "legal_changed": [legal(x) != legal(y) for x, y in zip(nm, n1)],
    "dspace": ["  " in x for x in nm],
    "accent": [any(ord(ch) > 127 and ch.isalpha() for ch in x) and not any(ord(ch) > 0x24F for ch in x) for x in nm],
    "addr_upper": [x == x.upper() and any(ch.isalpha() for ch in x) for x in ad],
    "addr_empty": [not x.strip() for x in ad],
    "exact_name": [x == y for x, y in zip(nm, n1)],
})
tr["s1"], tr["src"], tr["c"] = t.s1.to_numpy(), t.src.to_numpy(), t.c.to_numpy()
tr = tr.sort_values(["s1", "src"]).reset_index(drop=True)
same = (tr.s1.to_numpy()[1:] == tr.s1.to_numpy()[:-1]) & (tr.src.to_numpy()[1:] == tr.src.to_numpy()[:-1])
rng = np.random.default_rng(0)
rows = []
for c in ("India", "US"):
    for src in ("S2", "S3"):
        idx = np.flatnonzero(same & (tr.c.to_numpy()[:-1] == c) & (tr.src.to_numpy()[:-1] == src))
        pool = np.flatnonzero((tr.c.to_numpy() == c) & (tr.src.to_numpy() == src))
        ra, rb = rng.choice(pool, len(idx)), rng.choice(pool, len(idx))
        for k in ["upper", "lower", "legal", "legal_changed", "dspace", "accent", "addr_upper", "addr_empty", "exact_name"]:
            v = tr[k].to_numpy()
            rows.append(dict(c=c, src=src, trait=k, rate=(v[pool] != (v[pool][0] if v.dtype == object else False)).mean() if v.dtype != object else np.nan,
                             p_equal_same_entity=(v[idx] == v[idx + 1]).mean(), p_equal_random=(v[ra] == v[rb]).mean()))
print(pd.DataFrame(rows).round(4).to_string())
