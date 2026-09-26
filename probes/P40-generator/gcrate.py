"""P40: true rate of rejected argmax pairs by p' band x (S1 group lacks the SX's source). Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT
from loss import bits

for c in ("India", "US"):
    d = pd.read_parquet(OUT / f"am_{c}.parquet")
    n = d.g.max() + 1
    is2 = d.sx.str.startswith("S2-").to_numpy()
    acc = d.acc.to_numpy()
    h2 = np.bincount(d.g[acc & is2], minlength=n) > 0
    h3 = np.bincount(d.g[acc & ~is2], minlength=n) > 0
    g = d.g.to_numpy()
    grp = np.where(~h2[g] & ~h3[g], "empty", np.where(np.where(is2, ~h2[g], ~h3[g]), "lacks_src", "has_src"))
    B = bits(d.ops.to_numpy())
    r = d[~acc].assign(grp=grp[~acc], band=pd.cut(d.pa[~acc], [0, .1, .3, .5, .6, .71]), noaddr=B.A_EMPTY.to_numpy()[~acc])
    print(c)
    print(r.pivot_table(index="band", columns="grp", values="y", aggfunc=["size", "mean"], observed=True).round(3).to_string())
