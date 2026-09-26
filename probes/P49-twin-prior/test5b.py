"""P49 test 5b: an ambiguous address-less SX x and the OTHER true members of its twins. If some twin j holds a member
with x's identical raw name (with or without an address), is j the owner of x? Oracle membership (full train truth);
addressed members are what a ~99%-precise decision holds, so the 'addressed sibling' row is the label-free-able one.
Also how often a Levenshtein-closest member (raw name) sits at the owner. Aggregates only.
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/test5b.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame import W, sx_text, noaddr  # noqa: E402


def se(p, n):
    return np.sqrt(p * (1 - p) / max(n, 1))


for c in ("India", "US"):
    tw = pd.read_parquet(W / f"p49/tw_{c}.parquet", columns=["sx", "s1", "holdout", "y_full", "nset", "sx_name", "src"])
    tw["own"] = tw.groupby("sx").y_full.transform("any")
    tw["anyho"] = tw.groupby("sx").holdout.transform("any")
    q = tw[tw.own & tw.anyho]
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("s1", "in", list(set(q.s1)))]).to_pandas()
    tx = sx_text(set(gt.sx)).set_index("entity_id")
    gt["n"] = tx.business_name.reindex(gt.sx).to_numpy()
    gt["na"] = noaddr(tx.business_address.reindex(gt.sx))
    del tx
    m = q[["sx", "s1", "sx_name", "nset", "y_full", "src"]].merge(gt.rename(columns={"sx": "sy"}), on="s1")
    m = m[m.sx != m.sy]
    m = m[m.n == m.sx_name]
    for nm, sub in (("any sibling", m), ("addressed sibling", m[~m.na]), ("address-less sibling", m[m.na]),
                    ("addressed, same source", m[~m.na & (m.sy.str[:2] == m.src)]),
                    ("addressed, other source", m[~m.na & (m.sy.str[:2] != m.src)])):
        holders = sub.groupby("sx").s1.nunique()
        one = sub[sub.sx.isin(holders[holders == 1].index)].drop_duplicates("sx")
        r = one.y_full.mean() if len(one) else np.nan
        print(f"{c}: {nm:26s}: x with an identical-raw-name member at exactly one twin: {len(one):5,} of {q.sx.nunique():,} "
              f"({len(one) / q.sx.nunique():.3f}); that twin is the owner {r:.3f} ± {se(r, len(one)):.3f} "
              f"(uniform {(1 / one.nset).mean() if len(one) else np.nan:.3f}); >1 twin holds one: {(holders > 1).sum():,}", flush=True)
