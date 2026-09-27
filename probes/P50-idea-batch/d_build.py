"""P50 D: CE text-order TTA channel. 10k band argmax pairs per country (A frames, r notna = #8c band), P45's
full-coverage texts. holdout_India = all 20k in the trained order (ta = S1 text, tb = SX text), holdout_US = the same
20k swapped (ta = SX, tb = S1); the P13 bge_ckpt job scores both. Stubs = P45's.
    .venv/bin/python probes/P50-idea-batch/d_build.py"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

DL, OUT = Path("data/work/p50/dl"), Path("data/work/p50/ce")
OUT.mkdir(parents=True, exist_ok=True)
parts = []
for c in ("India", "US"):
    A = pd.read_parquet(f"data/work/p44ms/A_{c}.parquet", columns=["s1", "sx", "r", "y", "p_r_all"])
    A = A[A.r.notna()].sample(10_000, random_state=50)
    t = pq.read_table(DL / f"holdout_{c}.parquet")
    t = t.filter(pc.is_in(t.column("s1"), value_set=pa.array(A.s1.unique(), pa.large_string())))
    d = t.to_pandas().merge(A[["s1", "sx", "r", "p_r_all"]], on=["s1", "sx"], how="inner").drop_duplicates(["s1", "sx"])
    assert len(d) == 10_000, len(d)
    d["country"] = c
    parts.append(d)
    del t
d = pd.concat(parts, ignore_index=True)
d.to_parquet(Path("data/work/p50") / "d_sample.parquet", index=False)
cols = ["s1", "sx", "ta", "tb", "y", "p"]
d[cols].to_parquet(OUT / "holdout_India.parquet", index=False)
d.rename(columns={"ta": "tb", "tb": "ta"})[cols].to_parquet(OUT / "holdout_US.parquet", index=False)
for f in DL.glob("*.parquet"):
    if not f.name.startswith("holdout_"):
        shutil.copy(f, OUT / f.name)
print(len(d), d.groupby("country").y.mean().to_dict(), "r vs p_r_all max diff", float(np.abs(d.r - d.p_r_all).max()))
print(d[["ta", "tb"]].head(3).map(len))
