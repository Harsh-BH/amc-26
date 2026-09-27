"""Twin tie-breaker test (train truth): does an address-less SX share its owner with an addressed SX of the exact same raw
name (case, typos, punctuation kept)? Restricted to owners whose name core is shared by >= 2 S1 (twins), where the record has
no other tie-breaker. Baseline = 1 / (number of twin S1 with that core)."""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
tr = pd.read_csv(ROOT / "data/student_resource/dataset/train/train_ground_truth.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
own = tr[tr.matched_entity_ids != ""].assign(sx=lambda x: x.matched_entity_ids.str.split(",")).explode("sx").set_index("sx").source1_entity_id
for c in ("US", "India"):
    core = pd.read_parquet(W / f"rf_train_s1_{c}.parquet", columns=["id", "core"]).set_index("id").core
    tw = core.map(core.value_counts())
    parts = []
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"train_source{i}.parquet").iter_batches(500_000, columns=["entity_id", "business_name", "business_address", "country"]):
            d = b.to_pandas()
            d = d[d.country == c]
            a = d.business_address.fillna("").str.strip().str.upper()
            parts.append(pd.DataFrame({"id": d.entity_id.to_numpy(), "src": i, "h": pd.util.hash_pandas_object(d.business_name.fillna(""), index=False).to_numpy(),
                                       "na": (a.isin(["", "NULL", "<NULL>"])).to_numpy()}))
    x = pd.concat(parts, ignore_index=True)
    x["owner"] = own.reindex(x.id).to_numpy()
    x["twins"] = tw.reindex(x.owner).fillna(0).to_numpy()
    grp = x.groupby("h")
    x["g_n"] = grp.id.transform("size")
    x["g_addr_owned"] = x.h.map(x[~x.na & x.owner.notna()].groupby("h").owner.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "MULTI"))
    q = x[x.na & x.owner.notna() & (x.twins >= 2) & x.g_addr_owned.notna() & (x.g_addr_owned != "MULTI")]
    same = (q.owner == q.g_addr_owned)
    base = (1 / q.twins).mean()
    print(f"{c}: address-less owned SX of twin S1: {int((x.na & x.owner.notna() & (x.twins >= 2)).sum()):,}; "
          f"with an exact-raw-name addressed copy of ONE owner: {len(q):,}; same owner {same.mean():.3f} (baseline 1/twins {base:.3f}); "
          f"same source {((q.owner == q.g_addr_owned) & True).sum():,}", flush=True)
    for s in (2, 3):
        qs = q[q.src == s]
        print(f"   source S{s}: n {len(qs):,} same owner {(qs.owner == qs.g_addr_owned).mean():.3f}", flush=True)
    del x, parts
