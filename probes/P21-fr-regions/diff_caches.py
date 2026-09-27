"""P21 check B detail: every France cache rebuilt under data/work/p21/w vs the shared data/work one -- byte equality,
then the rows / pairs that differ (texts, rules.feats, P3 candidates, X files, M2 p).

    .venv/bin/python probes/P21-fr-regions/diff_caches.py [variant]     # variant: '' (base) | fr_idf144 | fr_idfdom
"""
import filecmp
import sys
from pathlib import Path

import pandas as pd

A, B = Path("data/work"), Path("data/work/p21/w")
v = sys.argv[1] if len(sys.argv) > 1 else ""
if v:
    A, B = A / "p9" / v, B / "p9" / v
rels = ["p3/txt_test/France_s1.parquet", *[f"p3/txt_test/France_sx_00{i}.parquet" for i in range(3)],
        "rf_test_s1_France.parquet", "rf_test_sx_France.parquet", "p3/cands_test_France.parquet",
        *[f"p4/X/test_France_sx_00{i}.parquet" for i in range(3)], *[f"p4/p1/test_France_sx_00{i}.parquet" for i in range(3)]]
rels += sorted(str(p.relative_to(B)) for p in (B / "p9").glob("m2_test_France_*.parquet"))
rels += sorted(str(p.relative_to(B)) for p in (B / "p9").glob("r_test_France_*.parquet"))
for r in rels:
    a, b = A / r, B / r
    if not (a.exists() and b.exists()) or b.is_symlink():
        print(f"{r}: missing or not rebuilt ({a.exists()}, {b.exists()}, link={b.is_symlink()})")
        continue
    if filecmp.cmp(a, b, shallow=False):
        print(f"{r}: byte-identical")
        continue
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    key = [c for c in ("s1", "sx", "q1", "qx", "k") if c in x.columns] or \
          (["entity_id"] if "entity_id" in x.columns else ["id"] if "id" in x.columns else None)
    if key is None or len(x) != len(y) and not key:
        print(f"{r}: differs, shapes {x.shape} vs {y.shape}")
        continue
    m = x.merge(y, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    only_a, only_b = (m._merge == "left_only").sum(), (m._merge == "right_only").sum()
    both = m[m._merge == "both"]
    cols = [c[:-2] for c in m.columns if c.endswith("_a")]
    ne = pd.Series(False, index=both.index)
    for c in cols:
        ne |= ~((both[c + "_a"] == both[c + "_b"]) | (both[c + "_a"].isna() & both[c + "_b"].isna()))
    chg = [c for c in cols if (~((both[c + "_a"] == both[c + "_b"]) | (both[c + "_a"].isna() & both[c + "_b"].isna()))).any()]
    print(f"{r}: DIFFERS rows {len(x):,} vs {len(y):,}; only old {only_a}, only new {only_b}, changed {int(ne.sum())} "
          f"(cols {chg[:12]}); keys: {m.loc[(m._merge != 'both') | ne.reindex(m.index, fill_value=False), key].head(12).values.tolist()}")
