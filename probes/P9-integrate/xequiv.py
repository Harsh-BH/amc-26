"""Rebuild cached P4 X files with src/features.py into a temp dir and compare them to data/work/p4/X.

    .venv/bin/python probes/P9-integrate/xequiv.py test France France_sx_000
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path[:0] = ["code/business_entity_resolution/src", "."]
import features as F
import block as B

split, c, stems = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
out = Path("/tmp/p9/X")
t0 = time.time()
F.ctx_step(split, c)
F.feat_step(split, c, only=stems, out=out)
res = {}
for s in sorted(stems):
    a = pd.read_parquet(F.XDIR / f"{split}_{s}.parquet")
    b = pd.read_parquet(out / f"{split}_{s}.parquet")
    same_cols = list(a.columns) == list(b.columns)
    diff = {col: int((~((a[col].to_numpy() == b[col].to_numpy()) | (pd.isna(a[col]).to_numpy() & pd.isna(b[col]).to_numpy()))).sum())
            for col in a.columns}
    res[s] = {"rows": (len(a), len(b)), "same_cols": same_cols, "cells_differing": sum(diff.values()),
              "cols_differing": {k: v for k, v in diff.items() if v}}
    print(s, res[s], flush=True)
for p in F.CTX.glob(f"{split}_{c}_sx_*.parquet"):
    p.unlink()
print(f"{time.time() - t0:.0f}s peak {B.peak_gb():.2f} GB")
from ledger import log
log("exp", probe="P9-integrate", subproblem="integration", name=f"x-rebuild-equiv-{split}-{c}", world=split,
    metrics={f"{s}_cells_differing": v["cells_differing"] for s, v in res.items()} | {f"{s}_rows": v["rows"][1] for s, v in res.items()},
    config={"code": "src/features.py ctx_step + feat_step", "vs": "data/work/p4/X (P4 featrun.py)"},
    hypothesis="the promoted feature builder rebuilds P4's cached X files exactly", 
    verdict="keep" if all(v["cells_differing"] == 0 and v["same_cols"] and v["rows"][0] == v["rows"][1] for v in res.values()) else "drop",
    runtime_s=round(time.time() - t0), peak_gb=round(B.peak_gb(), 2))
