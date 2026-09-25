"""Smoke test of src/rerank.py's training entry points into a scratch dir (never touches data/work/p4b):
band India (LightGBM fold 2 -> folds 3-9 band) compared with P4b's trainband_India, then train on n pairs.

    flock /tmp/amc-gpu.lock .venv/bin/python probes/P9-integrate/ce_smoke.py
"""
import shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path[:0] = ["code/business_entity_resolution/src", "."]
import rerank as RR
from worlds import WORK

t0 = time.time()
ref = pd.read_parquet(RR.OUT / "trainband_India.parquet")
RR.OUT = WORK / "p9" / "ce_smoke"
RR.OUT.mkdir(parents=True, exist_ok=True)
RR.band("India")
b = pd.read_parquet(RR.OUT / "trainband_India.parquet")
same_rows = len(b) == len(ref) and (b[["q1", "qx"]].to_numpy() == ref[["q1", "qx"]].to_numpy()).all()
dp = float(np.abs(b.p.to_numpy() - ref.p.to_numpy()).max()) if same_rows else None
print(f"band India: {len(b):,} rows vs P4b {len(ref):,}, same rows {same_rows}, max |dp| {dp}", flush=True)
RR.train(countries=("India",), tag="_smoke", n=4000)
ok = (RR.OUT / "ce_smoke.pt").exists()
from ledger import log
log("exp", probe="P9-integrate", subproblem="integration", name="ce-train-entrypoint-smoke", world="train",
    metrics={"band_rows": len(b), "band_rows_p4b": len(ref), "band_same_rows": int(same_rows), "band_max_dp": dp,
             "train_wrote_ce": int(ok)},
    config={"code": "src/rerank.py band India + train(countries=India, n=4000)", "out": str(RR.OUT)},
    hypothesis="the promoted CE training entry points run from src/ and rebuild P4b's train band",
    verdict="keep" if ok and same_rows else "drop", runtime_s=round(time.time() - t0), peak_gb=round(RR.peak(), 2))
shutil.rmtree(RR.OUT)
