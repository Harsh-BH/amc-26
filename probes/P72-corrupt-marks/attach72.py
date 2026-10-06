"""P72: MK columns for the rows of a (s1, sx) frame from the RAW source files (streamed by row group, ids filtered).
    .venv/bin/python probes/P72-corrupt-marks/attach72.py <world> <c> [split]
      A_<world>_<c> (P57; world 'test' = P50's T_<c> ids) -> data/work/p72/mk_<world>_<c>.parquet (row order of A)
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path[:0] = ["probes/P72-corrupt-marks", "code/business_entity_resolution/src"]
import features as F  # noqa: E402
import marks as M  # noqa: E402

W = Path("data/work")
OUT = W / "p72"


def raw(split: str, src: str, want: pa.Array, c: str = None, vocab=None) -> pd.DataFrame:
    """entity_id -> (name, addr) for the ids in want; with c + vocab, also counts the country's name tokens."""
    f, parts = pq.ParquetFile(W / f"{split}_{src}.parquet"), []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["entity_id", "business_name", "business_address", "country"])
        if vocab is not None:
            for s in t.filter(pc.equal(t["country"], c)).column("business_name").to_pylist():
                vocab.update(set(M.tok(s)))
        t = t.filter(pc.is_in(t["entity_id"], value_set=want))
        parts.append(t.select([0, 1, 2]).to_pandas())
        del t
    return pd.concat(parts, ignore_index=True).set_index("entity_id")


def vocab_path(split, c):
    return OUT / f"vocab_{split}_{c}.pkl"


def main(w: str, c: str, split: str = "train"):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    src = W / "p57" / f"A_{w}_{c}.parquet" if w != "test" else W / "p72" / f"ids_test_{c}.parquet"
    A = pd.read_parquet(src, columns=["q1", "qx"])
    id1, idx = F.ids(split, c)                       # q1 / qx are codes into P3's text order
    A = pd.DataFrame({"s1": id1.take(pa.array(A.q1.to_numpy())).to_numpy(zero_copy_only=False),
                      "sx": idx.take(pa.array(A.qx.to_numpy())).to_numpy(zero_copy_only=False)})
    del id1, idx
    vp = vocab_path(split, c)
    vocab = pickle.loads(vp.read_bytes()) if vp.exists() else M.Counter()
    r1 = raw(split, "source1", pa.array(A.s1.unique(), pa.large_string()), c, None if vp.exists() else vocab)
    if not vp.exists():
        vp.write_bytes(pickle.dumps(vocab))
    rx = pd.concat([raw(split, s, pa.array(A.sx.unique(), pa.large_string())) for s in ("source2", "source3")])
    print(c, w, "raw", len(r1), len(rx), "vocab", len(vocab), f"{time.time() - t0:.0f}s", flush=True)
    a1, ax = r1.reindex(A.s1), rx.reindex(A.sx)
    assert a1.business_name.notna().all() and ax.business_name.notna().all()
    d = M.marks(a1.business_name.to_numpy(), a1.business_address.to_numpy(), ax.business_name.to_numpy(),
                ax.business_address.to_numpy(), vocab)
    d.to_parquet(OUT / f"mk_{w}_{c}.parquet")
    print(c, w, len(d), f"{time.time() - t0:.0f}s", d.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
