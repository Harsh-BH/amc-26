"""Check: stack.frame("test", c)'s p at t 0.71 reproduces output/sub7/matching_results.tsv for India / US exactly
(same pruned rows, same blend, same argmax). Prints counts only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402

sub = Path(sys.argv[1] if len(sys.argv) > 1 else "output/sub7") / "matching_results.tsv"
mr = pd.read_csv(sub, sep="\t", dtype=str, keep_default_na=False)
for c in ("India", "US"):
    d = S.frame("test", c)
    a = S.M.argmax_sx(d["qx"], d["p"]) & (d["p"] >= S.T7)
    mine = np.sort((d["q1"][a] << 32) | d["qx"][a])
    q1 = pc.index_in(pa.array(mr.source1_entity_id.to_numpy()), value_set=d["id1"]).to_numpy(zero_copy_only=False)
    m = mr[~pd.isna(q1)]
    m = m[m.matched_entity_ids != ""]
    e = m.assign(sx=m.matched_entity_ids.str.split(",")).explode("sx")
    k1 = pc.index_in(pa.array(e.source1_entity_id.to_numpy()), value_set=d["id1"]).to_numpy(zero_copy_only=False)
    kx = pc.index_in(pa.array(e.sx.to_numpy()), value_set=d["idx"]).to_numpy(zero_copy_only=False)
    theirs = np.sort((k1.astype(np.int64) << 32) | kx.astype(np.int64))
    print(f"{c}: reconstruction {len(mine):,} matches, {sub.parent.name} {len(theirs):,}; identical "
          f"{len(mine) == len(theirs) and bool((mine == theirs).all())}; only mine {len(np.setdiff1d(mine, theirs)):,}, "
          f"only theirs {len(np.setdiff1d(theirs, mine)):,}", flush=True)
    del d, e, m
