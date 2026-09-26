"""P35: the #8c holdout yardstick frame with the SX attributes twin reasoning needs.

    .venv/bin/python probes/P35-twin-propagate/frame.py [India US]     # ~2 min, < 1.5 GB per country

data/work/p35/ho_<c>.parquet = data/work/p23/pairs_<c> (every scored pair of the S2/S3 a holdout S1 competes for,
#8c's final p, am = argmax S1 of the SX, pred = am & p >= t) + per SX: src (2/3), noaddr (raw address empty or a NULL
placeholder), core / city / hn (P3 record features, data/work/rf_train_sx_<c>).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p35"
NULLS = {"", "null", "<null>", "none", "n/a", "na", "-"}


def read_ids(path, ids, cols, key):
    v = pa.array(sorted(ids), pa.large_string())
    f, out = pq.ParquetFile(path), []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        out.append(t.filter(pc.is_in(t[key].cast(pa.large_string()), value_set=v)).to_pandas())
        del t
    return pd.concat(out, ignore_index=True)


def noaddr(a: pd.Series) -> np.ndarray:
    return a.str.strip().str.lower().isin(NULLS).to_numpy()


def build(c: str):
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet")
    ids = set(pd.unique(d.sx))
    raw = pd.concat([read_ids(W / f"train_source{i}.parquet", ids, ["entity_id", "business_address"], "entity_id")
                     for i in (2, 3)], ignore_index=True)
    na = pd.Series(noaddr(raw.business_address), index=raw.entity_id.to_numpy())
    del raw
    rf = read_ids(W / f"rf_train_sx_{c}.parquet", ids, ["id", "core", "city", "hn"], "id").set_index("id")
    d["src"] = np.where(d.sx.str.startswith("S2-"), 2, 3).astype(np.int8)
    d["noaddr"] = na.reindex(d.sx).fillna(False).to_numpy(bool)
    for k in ("core", "city", "hn"):
        d[k] = rf[k].reindex(d.sx).fillna("").to_numpy()
    d.to_parquet(OUT / f"ho_{c}.parquet", index=False)
    sx = d.drop_duplicates("sx")
    print(f"{c}: {len(d):,} pairs, {len(sx):,} SX, address-less {sx.noaddr.mean():.3f}, core missing "
          f"{(sx.core == '').mean():.4f}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for c in sys.argv[1:] or ["India", "US"]:
        build(c)
