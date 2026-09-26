"""P40: label-free decoy check. Gib-named / address-less SX per S1 by country (train with owned share, test incl.
France). If France's gib SX per S1 matches India (no gib decoys) its rejected gib pairs are mostly true. Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "P36-iu-residual")]
from cat import W
import gib as G

import re
M = G.learn("train")
STEP = 10
DT = re.compile(r"(?i)(?<![0-9])[a-z]+[0158][a-z]+|\b[0158][a-z]{3,}")   # digit-for-letter inside a word (5abrisoft, denta1s)
gt = pd.read_parquet(W / "gt_pairs.parquet", columns=["sx"])
owned = set(gt.sx)
del gt
for split in ("train", "test"):
    s1 = pq.read_table(W / f"{split}_source1.parquet", columns=["country"]).column(0).to_pandas().value_counts()
    rows = []
    for i in (2, 3):
        f = pq.ParquetFile(W / f"{split}_source{i}.parquet")
        for rg in range(0, f.num_row_groups, STEP):          # row order is random: every STEP-th group is a fair sample
            t = f.read_row_group(rg).to_pandas()
            web = t.business_name.str.contains(r"www|\.com|\.in\b|\.fr\b|@", regex=True)
            t["gib"] = G.is_gib(t.business_name.tolist(), M) & ~web.to_numpy()
            t["noaddr"] = t.business_address.str.strip() == ""
            nm = t.business_name
            t["digit_typo"] = [bool(DT.search(x)) for x in nm]
            t["dspace"] = nm.str.contains("  ", regex=False)
            t["upper"] = [x == x.upper() and any(ch.isalpha() for ch in x) for x in nm]
            t["null_seg"] = t.business_address.str.contains("NULL", regex=False)
            t["na_seg"] = t.business_address.str.contains("N/A", regex=False)
            t["own"] = t.entity_id.isin(owned) if split == "train" else False
            rows.append(t.groupby("country").agg(n=("gib", "size"), gib=("gib", "sum"), noaddr=("noaddr", "sum"),
                                                 digit_typo=("digit_typo", "sum"), dspace=("dspace", "sum"), upper=("upper", "sum"),
                                                 null_seg=("null_seg", "sum"), na_seg=("na_seg", "sum"),
                                                 gib_own=("own", lambda x: (x & t.loc[x.index, "gib"]).sum()),
                                                 noaddr_own=("own", lambda x: (x & t.loc[x.index, "noaddr"]).sum())))
    r = pd.concat(rows).groupby(level=0).sum()
    r["S1"] = s1.reindex(r.index) / STEP
    for k in ("digit_typo", "dspace", "upper", "null_seg", "na_seg"):
        r[k] = r[k] / r.n
    r["sx_per_s1"] = r.n / r.S1
    r["gib_per_s1"] = r.gib / r.S1
    r["noaddr_per_s1"] = r.noaddr / r.S1
    r["gib_share_sx"] = r.gib / r.n
    r["noaddr_share_sx"] = r.noaddr / r.n
    if split == "train":
        r["gib_owned"] = r.gib_own / r.gib
        r["noaddr_owned"] = r.noaddr_own / r.noaddr
    print(split)
    print(r.drop(columns=["gib_own", "noaddr_own"]).round(4).to_string())
