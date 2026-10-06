"""Independent check of a staged P71 file vs its base: only France lines change, only additions (A) or only removals (B),
no SX in two lines, every match in the candidate file.  .venv/bin/python probes/P71-fr-noiseword/check71.py BASE OUT"""
import csv
import json
import sys

import pandas as pd

RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
base, out = sys.argv[1], sys.argv[2]
a = pd.read_csv(f"{base}/matching_results.tsv", **RD).set_index("source1_entity_id").matched_entity_ids
b = pd.read_csv(f"{out}/matching_results.tsv", **RD).set_index("source1_entity_id").matched_entity_ids
assert len(b) == 1_732_544 and b.index.is_unique and set(a.index) == set(b.index)
b = b.reindex(a.index)
diff = a.index[a.to_numpy() != b.to_numpy()]
s1 = pd.read_parquet("data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
assert (s1.reindex(diff) == "France").all(), "a non-France line changed"
sp = lambda v: set(v.split(",")) if v else set()
added = sum(len(sp(b[s]) - sp(a[s])) for s in diff)
removed = sum(len(sp(a[s]) - sp(b[s])) for s in diff)
ids = [x for v in b.to_numpy() if v for x in v.split(",")]
assert len(ids) == len(set(ids)), "an SX sits in two lines"
c = pd.read_csv(f"{out}/candidate_pairs.tsv", **RD).set_index("source1_entity_id").candidate_entity_ids.reindex(diff)
miss = sum(len(sp(b[s]) - sp(c[s])) for s in diff)
assert miss == 0, f"{miss} matches outside the candidate file"
print(json.dumps({"lines_changed": len(diff), "added": added, "removed": removed, "sx_unique": True, "in_candidates": True}))
