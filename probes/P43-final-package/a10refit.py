"""P43: does run.ensure_a10 (the zip's refit path for P34's a10 booster) rebuild data/work/p34/a10_h0.txt?
Fits into data/work/p43/a10_refit.txt from the holdout (CE_MISSING fallback, as P23's dump), then compares the model
text and the predictions on P34's holdout feature files (ho_{c}.parquet).
    cd ../amc-26-wt/p43-final-package && .venv/bin/python /home/harsh1/github-repos/amc-26/probes/P43-final-package/a10refit.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
import a10 as A  # noqa: E402
import lightgbm as lgb  # noqa: E402
import run  # noqa: E402

t0 = time.time()
run.use("sub37")
out = Path("data/work/p43/a10_refit.txt")
out.unlink(missing_ok=True)
run.A10 = str(out.relative_to("data/work"))
new = run.ensure_a10()
ref = lgb.Booster(model_file="data/work/p34/a10_h0.txt")
strip = lambda s: [l for l in s.splitlines() if not l.startswith(("[", "parameters", "end of parameters"))]
same_text = strip(out.read_text()) == strip(Path("data/work/p34/a10_h0.txt").read_text())
d = {}
for c in ("India", "US"):
    X = pd.read_parquet(f"data/work/p34/ho_{c}.parquet", columns=A.COLS)
    d[c] = float(np.abs(new.predict(X, raw_score=True) - ref.predict(X, raw_score=True)).max())
print({"trees_same_text": same_text, "max_abs_d_raw": d, "bytes_equal": out.read_bytes() ==
       Path("data/work/p34/a10_h0.txt").read_bytes(), "runtime_s": round(time.time() - t0)}, flush=True)
