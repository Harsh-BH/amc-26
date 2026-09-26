"""P49: sub44r's combiner (te44 VAR=r, same params / seeds / rounds) refit on holdout half 1, so half-0 S1 get an
out-of-fold p'' too. -> data/work/p49/ho_p44r_h1fit_{c}.npy (all A rows). Self-check: the half-0 fit path reproduces
P44's saved ho_p44r_{c}.npy.
    P44_VARIANT=r probes/P49-twin-prior/py.sh probes/P49-twin-prior/swapfit.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

assert os.environ.get("P44_VARIANT") == "r"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P44-meta-stack"))
import lightgbm as lgb  # noqa: E402
import te44 as T  # noqa: E402

D = T.MT.prep(T.MODELS)
for fit in (0, 1):
    tr = {c: D[c][0].half.to_numpy() == fit for c in T.E.C}
    X = pd.concat([D[c][0].loc[tr[c], T.FINAL] for c in T.E.C])
    y = np.concatenate([D[c][0].y.to_numpy()[tr[c]] for c in T.E.C])
    base = np.concatenate([D[c][0].lp.to_numpy()[tr[c]] for c in T.E.C])
    bs = [lgb.train(T.PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True), 200)
          for s in T.SEEDS]
    del X
    for c in T.E.C:
        p = T.combine_p(bs, D[c][0]).astype(np.float32)
        if fit == 0:
            ref = np.load(T.E.OUT / f"ho_p44r_{c}.npy")
            print(c, "h0 fit vs P44's saved p'': max |diff|", float(np.abs(p - ref).max()), flush=True)
            assert np.allclose(p, ref, atol=1e-5)
        else:
            np.save(T.E.W / f"p49/ho_p44r_h1fit_{c}.npy", p)
            r = T.E.delta(D[c][0], D[c][1], p >= T.E.T, 0)
            print(c, f"h1 fit -> h0 vs #34: {r[0]:+.6f} ± {r[1]:.6f}", flush=True)
