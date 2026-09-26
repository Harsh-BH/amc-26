"""P56: precision of sub44r's OOF p'' by band x greedy group state (rk_any / rk_same) x source, India/US holdout.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/diag.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm1 as A1

BANDS = [0.3, 0.5, 0.6, 0.66, 0.71, 0.76, 0.82, 0.9, 0.97, 1.01]
for c in A1.LB_W:
    d = A1.load(c)
    b = np.digitize(d["pa"], BANDS) - 1
    m = b >= 0
    f = pd.DataFrame({"band": np.array([f"{BANDS[i]:.2f}" for i in b[m]]), "src": np.where(d["s3"][m], "S3", "S2"),
                      "rk_any": d["rk_any"][m], "rk_same": d["rk_same"][m], "y": d["y"][m]})
    for by in ("rk_any", "src"):
        t = f.groupby(["band", by]).y.agg(["mean", "size"]).unstack(by)
        print(f"== {c} precision by p'' band x {by}\n", t.round(3).to_string(), flush=True)
