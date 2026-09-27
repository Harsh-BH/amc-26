"""P73 veto arm: fc_<frame> = the CONFIDENT rows (g >= .98) of P67's CE frames, same layout as fb_<frame>
(s1 = sx = frame row, ta = S1 text, tb = SX text, p = g), built with P67's frames() logic -> data/work/p73/ch/fc_*.parquet."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P4b-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import rerank as rr  # noqa: E402
W, OUT = ROOT / "data/work", ROOT / "data/work/p73/ch"
def write(name, ta, tb, g):
    n = len(g); m = g >= 0.98
    d = pd.DataFrame({"s1": np.arange(n, dtype=np.int64), "sx": np.arange(n, dtype=np.int64), "ta": ta, "tb": tb, "p": g.astype(np.float32)})[m]
    d.to_parquet(OUT / f"fc_{name}.parquet", index=False)
    print(f"fc_{name}: {int(m.sum()):,} of {n:,} rows", flush=True)
for dr, tgt in (("US_India", "India"), ("India_US", "US")):
    ids = pd.read_parquet(W / f"p48/loco_{dr}_ids.parquet", columns=["q1", "qx"])
    g = np.load(W / f"p42/loco_{dr}.npz")["g"]; assert len(ids) == len(g)
    keep = np.flatnonzero(g >= 0.98)
    ta = np.empty(len(g), object); tb = np.empty(len(g), object)
    a, b = rr.texts("train", tgt, ids.q1.to_numpy()[keep], ids.qx.to_numpy()[keep]); ta[keep], tb[keep] = a, b
    write(dr, ta, tb, g); del ids, ta, tb
f = pd.read_parquet(W / "p65/sm/frame_France.parquet", columns=["row", "s1", "sx", "g"])
assert (f.row.to_numpy() == np.arange(len(f))).all()
g = f.g.to_numpy(); keep = np.flatnonzero(g >= 0.98)
q1, qx = rr.rows("test", "France", f.s1.iloc[keep], f.sx.iloc[keep])
ta = np.empty(len(g), object); tb = np.empty(len(g), object)
a, b = rr.texts("test", "France", q1, qx); ta[keep], tb[keep] = a, b
write("France", ta, tb, g)
