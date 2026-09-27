"""Tiny local rehearsal channels for solution_p26.py: 8% of the SX (whole candidate lists) of X file 000 per country.
    .venv/bin/python probes/P26-france-da/rh_build.py   -> data/work/p26/rh/{train,test}
"""
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

W = Path(__file__).resolve().parents[2] / "data/work"
R = W / "p26/rh"


def tiny(src, dst, rate=0.08):
    dst.parent.mkdir(parents=True, exist_ok=True)
    w = None
    for b in pq.ParquetFile(src).iter_batches(batch_size=200_000):
        t = pa.Table.from_batches([b])
        qx = t.column("qx").to_numpy().astype(np.uint64)
        t = t.filter(pa.array((qx * np.uint64(40503) % np.uint64(1000)) < rate * 1000))
        w = w or pq.ParquetWriter(dst, t.schema)
        w.write_table(t)
    w.close()


for c in ("US", "India"):
    tiny(W / f"p4/X/train_{c}_sx_000.parquet", R / f"train/p4/X/train_{c}_sx_000.parquet")
tiny(W / "p14/w_idfx/p4/X/test_France_sx_000.parquet", R / "test/fr/X/test_France_sx_000.parquet")
x = pq.read_table(R / "test/fr/X/test_France_sx_000.parquet", columns=["q1", "qx"]).to_pandas()
k = pd.read_parquet(W / "p26/sm/fr/keys_France.parquet")
k = k.merge(x, on=["q1", "qx"])
k.to_parquet(R / "test/fr/keys_France.parquet", index=False)
for n in ("meta", "m2_big.txt"):
    (R / "test" / n).unlink(missing_ok=True)
    (R / "test" / n).symlink_to(W / "p26/sm" / n)
print("keys", len(k))
