"""P58: twin candidate expansion on the #8c holdout world. For every address-less train SX (P35 placeholder set) whose
expansion key is shared by >= 2 train S1 of its country, add EVERY S1 with that key as a candidate.
Expansion key = the key of the SX's #8c argmax S1 when the SX is in the P23 dump, else the SX's own key (P40 floor.key).

    probes/P58-twin-cands/py.sh probes/P58-twin-cands/frame58.py [c ...]   # -> data/work/p58/ex_{c}.parquet

ex_{c}: one row per (address-less SX, S1 of its key group): sx, s1, src, kfull (group size), in_set (#8c candidate), p (#8c,
NaN if new), am, holdout, y (full train truth), sx_name, s1_name, own_in (SX has a true owner inside the group), by_am.
Groups larger than GMAX are dropped (a twin there needs a unique raw signal; counted in the log).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P49-twin-prior"))
from frame import key, NA  # noqa: E402

W = Path(__file__).resolve().parents[2] / "data/work"
OUT = W / "p58"
GMAX = 30


def noaddr_sx(c: str) -> pd.DataFrame:
    """entity_id, business_name of the country's address-less S2/S3 records (streamed)."""
    na = pa.array(sorted(NA), pa.string())
    parts = []
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"train_source{i}.parquet").iter_batches(
                columns=["entity_id", "business_name", "business_address", "country"], batch_size=1_000_000):
            a = pc.utf8_upper(pc.utf8_trim_whitespace(b.column(2)))
            m = pc.and_(pc.equal(b.column(3), c), pc.is_in(a, value_set=na))
            parts.append(pa.Table.from_batches([b]).filter(m).select(["entity_id", "business_name"]).to_pandas())
    return pd.concat(parts, ignore_index=True)


def build(c: str):
    t0 = time.time()
    s1 = pq.read_table(W / "train_source1.parquet", columns=["entity_id", "business_name", "country"],
                       filters=[("country", "=", c)]).to_pandas().drop(columns="country")
    s1["k"] = [key(x) for x in s1.business_name]
    s1["kfull"] = s1.groupby("k").entity_id.transform("size")
    multi = s1[s1.kfull >= 2]
    ho = set(pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index)
    sx = noaddr_sx(c)
    sx["k_own"] = [key(x) for x in sx.business_name]
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "p", "am"])
    k1 = s1.set_index("entity_id").k
    amk = d[d.am].assign(k=lambda t: k1.reindex(t.s1).to_numpy()).set_index("sx").k
    sx["k_am"] = amk.reindex(sx.entity_id).to_numpy()
    sx["by_am"] = sx.k_am.notna()
    sx["k"] = sx.k_am.where(sx.by_am, sx.k_own)
    gsz = multi.groupby("k").size()
    sx["kfull"] = gsz.reindex(sx.k).fillna(0).to_numpy(int)
    big = int((sx.kfull > GMAX).sum())
    sx = sx[(sx.kfull >= 2) & (sx.kfull <= GMAX)]
    ex = sx[["entity_id", "business_name", "k", "kfull", "by_am"]].rename(columns={"entity_id": "sx", "business_name": "sx_name"}) \
        .merge(multi[["entity_id", "business_name", "k"]].rename(columns={"entity_id": "s1", "business_name": "s1_name"}), on="k")
    ex["src"] = ex.sx.str[:2]
    ex["holdout"] = ex.s1.isin(ho)
    ex = ex.merge(d[["s1", "sx", "p", "am"]], on=["s1", "sx"], how="left")
    ex["in_set"] = ex.p.notna()
    ex["am"] = ex.am.fillna(False).astype(bool)
    gt = pq.read_table(W / "gt_pairs.parquet", filters=[("sx", "in", list(set(ex.sx)))]).to_pandas()
    ya = set(zip(gt.s1, gt.sx))
    ex["y"] = [(a, b) in ya for a, b in zip(ex.s1, ex.sx)]
    g = ex.groupby("sx")
    ex["own_in"] = g.y.transform("any")
    ex["old_own_in"] = (ex.y & ex.in_set).groupby(ex.sx).transform("any")
    ex["nset_old"] = g.in_set.transform("sum")
    ex = ex.drop(columns="k")
    ex.to_parquet(OUT / f"ex_{c}.parquet", index=False)
    s = ex.drop_duplicates("sx")
    owned = s[s.sx.isin(set(gt.sx))]
    amb = s[s.nset_old >= 2]                                   # P49's frame: >= 2 same-key twins already in the set
    amb_o = amb[amb.own_in]
    hw = ex[ex.holdout].sx.unique()
    print(f"{c}: {len(s):,} address-less SX in groups 2..{GMAX} ({big:,} in bigger groups), {len(ex):,} rows "
          f"({ex.in_set.mean():.3f} already candidates, +{(~ex.in_set).sum():,} new pairs); SX with any true owner {len(owned):,}, "
          f"owner in group {s.own_in.sum():,} ({s.own_in.sum() / max(len(owned), 1):.3f}); "
          f"P49-like (nset_old>=2) {len(amb):,}: owner in group {len(amb_o):,}, of which in old set {amb_o.old_own_in.mean():.3f} -> 1.000; "
          f"SX touching a holdout S1 {len(hw):,}; by_am {s.by_am.mean():.3f}; kfull {s.kfull.describe()[['mean', '50%', 'max']].round(1).to_dict()}; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for c in sys.argv[1:] or ("India", "US"):
        build(c)
