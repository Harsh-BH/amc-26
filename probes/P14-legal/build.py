"""P14-legal: build the train-side inputs of the matcher A/B with the pipeline's own code (src/ unmodified):
parquet copies, gt_pairs + folds, P3 blocking for every train S1, the rules.feats caches and the 65-feature X files.
Separate from legal.py because rules.feats' forkserver workers re-import __main__: never edit this file mid-run.

    .venv/bin/python probes/P14-legal/build.py          # ~1 h 40 min, peak ~4 GB; resumable (every step is cached)
    .venv/bin/python probes/P14-legal/build.py verify   # ctx_lean == features.ctx_step on India, file for file

features.ctx_step holds a country's whole candidate table (US train: 18.6M pairs, > 4.5 GB with no swap); ctx_lean
writes the same files in bounded memory.
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path[:0] = ["code/business_entity_resolution/src", "."]



def ctx_lean(split: str, c: str, out: Path):
    """features.ctx_step's files in bounded memory. block.finalize writes the candidates sorted by (q1, qx), so the
    S1 context is computed over whole runs of S1 at a time; rows are bucketed by SX file and each bucket is stably
    sorted by qx: the same rows, values and order as ctx_step's one global stable sort."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    import block as B
    import features as F
    out.mkdir(parents=True, exist_ok=True)
    id1, idx = F.ids(split, c)
    files, offs = F.sx_files(split, c)
    buckets, carry = [[] for _ in files], None

    def emit(d):
        for k, v in F._group_ctx(d.q1.to_numpy(), np.nan_to_num(d.block_score.to_numpy(np.float32)), "s1").items():
            d[k] = v
        fk = np.searchsorted(offs, d.qx.to_numpy(), side="right") - 1
        for k in np.unique(fk):
            buckets[k].append(d[fk == k])

    for rb in pq.ParquetFile(B.P3 / f"cands_{split}_{c}.parquet").iter_batches(2_000_000, columns=["s1", "sx"] + F.BLOCK):
        t = pa.Table.from_batches([rb])
        q1, qx = pc.index_in(t.column("s1"), value_set=id1), pc.index_in(t.column("sx"), value_set=idx)
        assert q1.null_count == 0 and qx.null_count == 0, "every candidate id is a record of the split"
        d = t.select(F.BLOCK).to_pandas().assign(q1=q1.to_numpy().astype(np.int32), qx=qx.to_numpy().astype(np.int32))
        d = pd.concat([carry, d], ignore_index=True) if carry is not None else d
        tail = d.q1.to_numpy() == d.q1.iloc[-1]                 # the last S1 may continue in the next batch
        carry, d = d[tail].reset_index(drop=True), d[~tail].reset_index(drop=True)
        assert (np.diff(d.q1.to_numpy()) >= 0).all(), "candidates sorted by S1"
        if len(d):
            emit(d)
    if carry is not None and len(carry):
        emit(carry)
    for k, f in enumerate(files):
        d = pd.concat(buckets[k], ignore_index=True)
        buckets[k] = None
        d = d.iloc[np.argsort(d.qx.to_numpy(), kind="stable")].reset_index(drop=True)
        d["via"] = d.via.astype("category")
        d.to_parquet(out / f"{split}_{f.stem}.parquet", index=False)


def verify(c: str = "India"):
    import pandas as pd

    import features as F
    ref, new = Path(tempfile.mkdtemp(dir="data/work")), Path(tempfile.mkdtemp(dir="data/work"))
    F.CTX = ref
    F.ctx_step("train", c)
    ctx_lean("train", c, new)
    for p in sorted(ref.glob("*.parquet")):
        a, b = pd.read_parquet(p), pd.read_parquet(new / p.name)
        a["via"], b["via"] = a.via.astype(str), b.via.astype(str)
        assert list(a.columns) == list(b.columns) and a.dtypes.equals(b.dtypes) and a.equals(b), p.name
        print(f"{p.name}: {len(a):,} rows identical", flush=True)
    shutil.rmtree(ref), shutil.rmtree(new)


if __name__ == "__main__" and sys.argv[1:] == ["verify"]:
    verify()
elif __name__ == "__main__":
    import numpy as np
    import pyarrow.parquet as pq

    import block as B
    import features as F
    import io_tsv
    import run
    import train_matcher
    import worlds
    from worlds import WORK

    t0 = time.time()
    if not all((WORK / f"{s}_source{i}.parquet").exists() for s in ("train", "test") for i in (1, 2, 3)):
        io_tsv.convert()
    if not (WORK / "s1_folds.parquet").exists():
        worlds.build_base()
    # p3/txt came from Harsh's Drive: its row order must be this machine's parquet order (features.raw_sx asserts it
    # 30 min in, after blocking). Check it now; on a mismatch rebuild the texts (~6 min) instead.
    for c in train_matcher.countries():
        ids = lambda src: [i for d in B._batches(src, c) for i in d.entity_id]
        txt = sorted(B.TXT["train"].glob(f"{c}_sx_*.parquet"))
        got1 = pq.read_table(B.TXT["train"] / f"{c}_s1.parquet", columns=["entity_id"]).column(0).to_pylist()
        gotx = [i for p in txt for i in pq.read_table(p, columns=["entity_id"]).column(0).to_pylist()]
        if got1 != ids("train_source1") or gotx != ids("train_source2") + ids("train_source3"):
            print(f"{c}: cached texts are in a different row order -- rebuilding them", flush=True)
            shutil.rmtree(B.TXT["train"])
            break
        print(f"{c}: text order checked ({len(got1):,} S1, {len(gotx):,} SX)", flush=True)
    for c in train_matcher.countries():
        run.prepare("train", c)
        if not all(p.exists() for p in F.xfiles("train", c)):
            B.split_cands("train", c) if not (B.P3 / f"cands_train_{c}.parquet").exists() else None
            ctx_lean("train", c, F.CTX)                  # in place of F.build's ctx_step
            F.feat_step("train", c)
            for p in F.CTX.glob(f"train_{c}_sx_*.parquet"):
                p.unlink()
        print(f"{c}: X files done at {(time.time() - t0) / 60:.0f} min, peak {B.peak_gb():.2f} GB", flush=True)
    n = sum(pq.ParquetFile(p).metadata.num_rows for c in train_matcher.countries() for p in F.xfiles("train", c))
    print(f"build done: {n:,} train pairs in {(time.time() - t0) / 60:.0f} min", flush=True)
