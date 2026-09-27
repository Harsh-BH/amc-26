"""splice.py FRANCE_SUB BASE OUT [VETO.parquet]: BASE's matching_results.tsv with every France S1 line taken from FRANCE_SUB; BASE's
candidate_pairs.tsv hard-linked. Asserts same S1 order, only France lines differ, France matches within BASE's
candidate lists. VETO: (s1, sx) pairs removed from the France lines afterwards (veto51.py). Prints lines changed and
pairs added / removed."""
import csv, json, os, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
fr_sub, base, out = map(Path, sys.argv[1:4])
veto = {}
if len(sys.argv) > 4:
    for a, b in pd.read_parquet(sys.argv[4]).itertuples(index=False):
        veto.setdefault(a, set()).add(b)
fr = set(pd.read_parquet(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"])
         .query("country == 'France'").entity_id)
cand = {}
with open(base / "candidate_pairs.tsv", encoding="utf-8") as f:
    next(f)
    for ln in f:
        a, b = ln.rstrip("\n").split("\t")
        if a in fr:
            cand[a] = set(b.split(",")) if b else set()
out.mkdir(parents=True, exist_ok=True)
chg = add = rem = nv = 0
with open(base / "matching_results.tsv", encoding="utf-8") as f, open(fr_sub / "matching_results.tsv", encoding="utf-8") as g, \
        open(out / "matching_results.tsv", "w", encoding="utf-8") as h:
    h.write(next(f)); next(g)
    for lb, lf in zip(f, g):
        s1, xb = lb.rstrip("\n").split("\t")
        s1f, xf = lf.rstrip("\n").split("\t")
        assert s1 == s1f, (s1, s1f)
        if s1 in fr:
            A, B = set(xb.split(",")) - {""}, set(xf.split(",")) - {""}
            v = B & veto.get(s1, set())
            nv += len(v)
            B -= v
            assert B <= cand[s1], s1
            chg += A != B; add += len(B - A); rem += len(A - B)
            h.write(f"{s1}\t{','.join(x for x in xf.split(',') if x and x not in v)}\n" if v else lf)
        else:
            h.write(lb)
    assert next(f, None) is None and next(g, None) is None
for n in ("candidate_pairs.tsv",):
    if not (out / n).exists():
        os.link(base / n, out / n)
info = {"france_sub": str(fr_sub), "base": str(base), "france_lines_changed": chg, "pairs_added": add, "pairs_removed": rem, "vetoed": nv}
(out / "p51_splice.json").write_text(json.dumps(info))
print(json.dumps(info))
