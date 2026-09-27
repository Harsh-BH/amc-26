"""P65 prep: SageMaker inputs for the ByT5 corrector (data/work/p65/sm/, then S3).

  train C N        N true pairs of country C whose S1 is NOT fold 0 (the holdout): src = SX "name | addr" (raw),
                   tgt = its S1 "name | addr" -> train_C.parquet
  frame SRC TGT    every row of P42's LOCO frame loco_SRC_TGT.npz (TGT fold 0, SX-argmax pairs, same order: P42 prep's
                   S_holdout filter), with ids + raw texts -> frame_SRC_TGT.parquet (row, s1, sx, g, sx_txt, s1_txt)

    .venv/bin/python probes/P65-byt5-corrector/prep65.py train US 400000 | frame US India
"""
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
W, OUT = ROOT / "data/work", ROOT / "data/work/p65/sm"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def txt(files, ids) -> pd.Series:
    """entity_id -> 'name | addr' for the given ids, streamed row groups (never a whole source in RAM)."""
    vs, parts = pa.array(np.unique(ids), pa.large_string()), []
    for f in files:
        pf = pq.ParquetFile(W / f)
        for b in pf.iter_batches(500_000, columns=["entity_id", "business_name", "business_address"]):
            b = b.filter(pc.is_in(b.column(0), value_set=vs))
            if b.num_rows:
                d = b.to_pandas()
                parts.append(pd.Series((d.business_name + " | " + d.business_address).to_numpy(), index=d.entity_id.to_numpy()))
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def train(c, n):
    t0 = time.time()
    fo = pd.read_parquet(W / "s1_folds.parquet")
    keep = fo.entity_id[(fo.country == c) & (fo.fold != 0)].to_numpy()
    del fo
    gp = pq.read_table(W / "gt_pairs.parquet")
    gp = gp.filter(pc.is_in(gp.column("s1"), value_set=pa.array(keep, pa.large_string()))).to_pandas()
    gp = gp.sample(n=min(int(n), len(gp)), random_state=65).reset_index(drop=True)
    s1 = txt(["train_source1.parquet"], gp.s1.to_numpy())
    sx = txt(["train_source2.parquet", "train_source3.parquet"], gp.sx.to_numpy())
    d = pd.DataFrame({"src": sx.reindex(gp.sx).to_numpy(), "tgt": s1.reindex(gp.s1).to_numpy()})
    assert d.notna().all().all()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"train_{c}.parquet", index=False)
    b = d.src.str.len()
    print(f"train {c}: {len(d):,} pairs, src chars p50/p95/p99 {b.quantile([.5, .95, .99]).tolist()}; "
          f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


def frame(src, tgt):
    import features as F
    t0 = time.time()
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "am_all"])
    S = S[(S.country.to_numpy() == ci) & S.am_all.to_numpy()].reset_index(drop=True)
    z = np.load(W / f"p42/loco_{src}_{tgt}.npz")
    assert len(S) == len(z["g"]) and np.allclose(S.p.to_numpy(np.float64), z["g"]), "P42 frame order"
    id1, idx = F.ids("train", tgt)
    q1 = S.s1.to_numpy(np.int64) - ci * OFF1
    qx = S.sx.to_numpy(np.int64) - ci * OFF2
    e1 = id1.take(pa.array(q1)).to_numpy(zero_copy_only=False)
    ex = idx.take(pa.array(qx)).to_numpy(zero_copy_only=False)
    del id1, idx
    t1 = txt(["train_source1.parquet"], e1)
    tx = txt(["train_source2.parquet", "train_source3.parquet"], ex)
    d = pd.DataFrame({"row": np.arange(len(S), dtype=np.int32), "s1": e1, "sx": ex, "g": z["g"].astype(np.float32),
                      "sx_txt": tx.reindex(ex).to_numpy(), "s1_txt": t1.reindex(e1).to_numpy()})
    assert d.notna().all().all() and d.sx.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"frame_{src}_{tgt}.parquet", index=False)
    print(f"frame {src}>{tgt}: {len(d):,} rows; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)




def france():
    """P55's France frame (argmax pairs of france.npz at the current recipe), row = its X row order -> frame_France."""
    t0 = time.time()
    sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT / "probes/P48-fr-ce-all")]
    import p42
    import p55
    f = p55.france_frame()
    z, rows = f["z"], f["rows"]
    i1, ix = p42.fr_ids()
    e1, ex = np.asarray(i1)[z["s1"][rows]], np.asarray(ix)[z["sx"][rows]]
    g = f["g"].astype(np.float32)
    del f, z
    t1 = txt(["test_source1.parquet"], e1)
    tx = txt(["test_source2.parquet", "test_source3.parquet"], ex)
    d = pd.DataFrame({"row": np.arange(len(rows), dtype=np.int32), "zrow": rows.astype(np.int32), "s1": e1, "sx": ex,
                      "g": g, "sx_txt": tx.reindex(ex).to_numpy(), "s1_txt": t1.reindex(e1).to_numpy()})
    assert d.notna().all().all() and d.sx.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "frame_France.parquet", index=False)
    print(f"frame France: {len(d):,} rows, band {((g > .02) & (g < .98)).sum():,}; {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)


if __name__ == "__main__":
    {"train": train, "frame": frame, "france": france}[sys.argv[1]](*sys.argv[2:])
