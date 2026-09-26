"""P54 idea 3: content-identical SX duplicates (same raw name, address, country; S2 and S3 pooled) where one copy is
matched to an S1 and another copy is unassigned. Fix: give the unassigned copies to the same S1 (organiser ruling:
list every duplicate ID), past the 5/6 cap. Skips groups whose assigned copies sit at > 1 S1.

    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/dup54.py scan  SUB          # test counts -> data/work/p54/dup_<SUB>.parquet
    probes/P54-fresh-batch/py.sh probes/P54-fresh-batch/dup54.py train              # does a copy share its twin's owner?
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p54"
COLS = ["entity_id", "business_name", "business_address", "country"]


def eid(s):
    """'S2-123' -> 2e10 + 123 (int64), so 10M ids fit in 80 MB."""
    return s.str.slice(1, 2).astype(np.int64) * 10**10 + s.str.slice(3).astype(np.int64)


def keys(split, norm=False):
    """(id, key, country) for every S2/S3 row of the split whose key occurs >= 2 times; batched, ids as int64."""
    import pyarrow.parquet as pq
    ids, ks, cs = [], [], []
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"{split}_source{i}.parquet").iter_batches(500_000, columns=COLS):
            d = b.to_pandas()
            x = d[["business_name", "business_address", "country"]]
            if norm:
                x = x.apply(lambda s: s.str.lower().str.strip())
            ks.append(pd.util.hash_pandas_object(x, index=False).to_numpy())
            ids.append(eid(d.entity_id).to_numpy())
            cs.append(d.country.astype("category"))
            del d, x
    k = pd.DataFrame({"id": np.concatenate(ids), "key": np.concatenate(ks)})
    k["country"] = pd.api.types.union_categoricals(cs)
    del ids, ks, cs
    return k[k.key.duplicated(keep=False)].reset_index(drop=True)


def pairs(path, col, s1=None):
    m = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    m = m[(m[col] != "") & (True if s1 is None else m.source1_entity_id.isin(s1))]
    m = m.assign(sx=m[col].str.split(",")).explode("sx")[["source1_entity_id", "sx"]].rename(columns={"source1_entity_id": "s1"})
    m["sx"] = eid(m.sx)
    return m


NORM = "norm" in sys.argv[2:]        # key on lower/strip text instead of raw


def scan(sub):
    t0 = time.time()
    k = keys("test", NORM)
    dup = k.id.to_numpy()
    m = pairs(ROOT / f"output/{sub}/matching_results.tsv", "matched_entity_ids")
    m = m[m.sx.isin(dup)]
    k = k.merge(m, left_on="id", right_on="sx", how="left").drop(columns="sx")
    g = k.groupby("key", observed=True).agg(n=("id", "size"), n_acc=("s1", "count"), n_s1=("s1", "nunique"), country=("country", "first"))
    mixed = g[(g.n_acc > 0) & (g.n_acc < g.n)]
    one = mixed[mixed.n_s1 == 1]
    fix = k[k.key.isin(one.index) & k.s1.isna()].drop(columns="s1").merge(
        k[k.key.isin(one.index) & k.s1.notna()][["key", "s1"]].drop_duplicates(), on="key")
    c = pairs(ROOT / f"output/{sub}/candidate_pairs.tsv", "candidate_entity_ids", set(fix.s1))
    fix["in_cand"] = pd.MultiIndex.from_frame(fix[["s1", "id"]]).isin(pd.MultiIndex.from_frame(c[["s1", "sx"]]))
    tag = f"{sub}{'_norm' if NORM else ''}"
    fix.to_parquet(OUT / f"dup_{tag}.parquet")
    info = {"dup_groups": int(len(g)), "dup_rows": int(g.n.sum()),
            "groups_by_country": g.country.value_counts().to_dict(),
            "mixed_groups": int(len(mixed)), "mixed_by_country": mixed.country.value_counts().to_dict(),
            "mixed_multi_s1": int((mixed.n_s1 > 1).sum()), "all_assigned_multi_s1": int(((g.n_acc == g.n) & (g.n_s1 > 1)).sum()),
            "none_assigned_groups": int((g.n_acc == 0).sum()), "none_assigned_by_country": g[g.n_acc == 0].country.value_counts().to_dict(),
            "fix_pairs": int(len(fix)), "fix_by_country": fix.country.value_counts().to_dict(),
            "fix_in_cand": int(fix.in_cand.sum()), "fix_s1": int(fix.s1.nunique()), "runtime_s": round(time.time() - t0)}
    print(json.dumps(info, indent=1))
    (OUT / f"dup_{tag}.json").write_text(json.dumps(info, indent=1))


def train():
    """Train truth: for each content-identical group, do its matched copies share one owner, and are copies ever split
    between an owner and 'no match'?"""
    t0 = time.time()
    k = keys("train", NORM)
    gt = pd.read_parquet(W / "train_ground_truth.parquet")
    gt = gt[gt.matched_entity_ids != ""]
    gt = gt.assign(sx=gt.matched_entity_ids.str.split(",")).explode("sx")
    gt["sx"] = eid(gt.sx)
    gt = gt[gt.sx.isin(k.id.to_numpy())][["source1_entity_id", "sx"]]
    k = k.merge(gt, left_on="id", right_on="sx", how="left")
    g = k.groupby("key", observed=True).agg(n=("id", "size"), n_m=("source1_entity_id", "count"),
                             n_s1=("source1_entity_id", "nunique"), country=("country", "first"))
    info = {"groups": int(len(g)), "all_same_owner": int(((g.n_m == g.n) & (g.n_s1 == 1)).sum()),
            "all_unmatched": int((g.n_m == 0).sum()), "split_owner_vs_none": int(((g.n_m > 0) & (g.n_m < g.n)).sum()),
            "multi_owner": int((g.n_s1 > 1).sum()), "by_country": g.country.value_counts().to_dict(),
            "runtime_s": round(time.time() - t0)}
    # P(copy shares the owner | one copy has an owner): per matched copy, the share of its other copies with the same owner
    mg = g[g.n_m > 0]
    info["p_copy_same_owner_given_owned"] = round(float(((mg.n_m - 1) / (mg.n - 1)).mean()), 4) if len(mg) else None
    print(json.dumps(info, indent=1))
    (OUT / f"dup_train{'_norm' if NORM else ''}.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"scan": lambda: scan(sys.argv[2]), "train": train}[sys.argv[1]]()
