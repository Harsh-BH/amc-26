"""P44-meta-stack: house-number lookup tables for the hn-focused 2nd GBDT's SageMaker job.

hn/{split}_{c}_{s1|sx}.parquet: one string column `hn` ('' = none) in q1 / qx code order (features.ids), so the job
turns any X-file row into (hn1[q1], hnx[qx]) with two takes.
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/hn_tables.py
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
import features as F  # noqa: E402
from worlds import WORK  # noqa: E402

OUT = WORK / "p44ms/hn"


def table(split: str, side: str, c: str, ids: pa.Array) -> pa.Table:
    t = pq.read_table(WORK / f"rf_{split}_{side}_{c}.parquet", columns=["id", "hn"])
    pos = pc.index_in(ids, value_set=t.column("id"))
    assert pos.null_count == 0, f"{split} {side} {c}: ids without rule features"
    return pa.table({"hn": pc.fill_null(t.column("hn"), "").combine_chunks().take(pos)})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for c in ("India", "US"):
            id1, idx = F.ids(split, c)
            for side, ids in (("s1", id1), ("sx", idx)):
                t = table(split, side, c, ids)
                pq.write_table(t, OUT / f"{split}_{c}_{side}.parquet")
                h = t.column("hn")
                print(split, c, side, len(t), f"has hn {pc.mean(pc.not_equal(h, '').cast(pa.int8())).as_py():.3f}",
                      flush=True)
            del id1, idx
