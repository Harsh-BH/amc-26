"""P36 label-free test profile: gib SX per country and the share the current best file (#34v_rescue90) assigns.
    PYTHONPATH=probes/P36-iu-residual probes/P36-iu-residual/py.sh probes/P36-iu-residual/te_gib.py [sub_dir]
"""
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from base import OUT, ROOT, W
import gib as G

sub = ROOT / "output" / (sys.argv[1] if len(sys.argv) > 1 else "sub34v_rescue90") / "matching_results.tsv"
t = pcsv.read_csv(sub, parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
                  convert_options=pcsv.ConvertOptions(column_types={"source1_entity_id": pa.large_string(),
                                                                    "matched_entity_ids": pa.large_string()}))
acc = pc.list_flatten(pc.split_pattern(t["matched_entity_ids"], ",")).combine_chunks()
acc = acc.filter(pc.greater(pc.utf8_length(acc), 0))
del t
M = G.learn("test")
rows = []
for i in (2, 3):
    f = pq.ParquetFile(W / f"test_source{i}.parquet")
    for rg in range(f.num_row_groups):
        b = f.read_row_group(rg, columns=["entity_id", "business_name", "country"])
        g = G.is_gib(b.column("business_name").to_pylist(), M)
        a = pc.is_in(b["entity_id"], value_set=acc).to_numpy(zero_copy_only=False)
        rows.append(pd.DataFrame({"c": b.column("country").to_numpy(zero_copy_only=False), "gib": g, "acc": a}))
d = pd.concat(rows, ignore_index=True)
r = d.groupby(["c", "gib"]).acc.agg(["size", "mean"]).unstack(1)
print(sub.parent.name, "\n", r.round(4).to_string())
r.to_csv(OUT / "te_gib.csv")
