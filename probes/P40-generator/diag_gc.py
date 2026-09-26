"""P40: are the missed address-less SX 'group completion' material? (#34 holdout, oof). Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "P36-iu-residual")]
from base import state
from cat import OUT
from loss import bits

for c in ("India", "US"):
    s = state(c)
    pr, n = s["pairs"], len(s["ids"])
    is2 = pr.sx.str.startswith("S2-").to_numpy()
    acc = pr.acc.to_numpy()
    h2 = np.bincount(pr.g[acc & is2], minlength=n) > 0
    h3 = np.bincount(pr.g[acc & ~is2], minlength=n) > 0
    e = pd.read_parquet(OUT / f"err_{c}.parquet")
    e = e[~e.type.isin(["fp_ns", "fp_s"])]
    B = bits(e.ops.to_numpy())
    emp = B.A_EMPTY.to_numpy()
    ex2 = e.sx.str.startswith("S2-").to_numpy()
    g = e.g.to_numpy()
    lacks = np.where(ex2, ~h2[g], ~h3[g])
    grp_empty = ~h2[g] & ~h3[g]
    df = pd.DataFrame(dict(type=e.type.to_numpy(), emp=emp, lacks_src=lacks & ~grp_empty, grp_empty=grp_empty))
    print(c, "FN pairs by type x (address-less) -> share whose S1 group lacks the SX's source (non-empty group) / empty group")
    print(df.groupby(["type", "emp"]).agg(n=("lacks_src", "size"), lacks=("lacks_src", "mean"), empty=("grp_empty", "mean")).round(3).to_string())
    # base rate: among TRUE accepted pairs, share whose S1 would lack the source without them? -> shape
    nt = s["n_true"]
    print("predicted shape (non-empty S1 with truth): both", round((h2 & h3)[nt > 0].mean(), 4), "S2-only", round((h2 & ~h3)[nt > 0].mean(), 4),
          "S3-only", round((~h2 & h3)[nt > 0].mean(), 4), "empty", round((~h2 & ~h3)[nt > 0].mean(), 4))
