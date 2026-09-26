"""P40: rejected argmax address-less pairs whose S1 name is unique (core twin count k = 1) by p' band. Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT
from loss import bits

for c in ("India", "US"):
    d = pd.read_parquet(OUT / f"am_{c}.parquet")
    B = bits(d.ops.to_numpy())
    k = pd.read_parquet(OUT / f"s1twin_{c}.parquet").set_index("id").k.reindex(d.s1).fillna(1).to_numpy()
    emp = B.A_EMPTY.to_numpy()
    r = d[emp & ~d.acc.to_numpy()].assign(k1=k[emp & ~d.acc.to_numpy()] == 1,
                                          band=pd.cut(d.pa[emp & ~d.acc.to_numpy()], [0, .1, .3, .5, .6, .71]))
    print(c, "rejected address-less argmax pairs: n and true rate by p' band x unique core name")
    print(r.pivot_table(index="band", columns="k1", values="y", aggfunc=["size", "mean"], observed=True).round(3).to_string())
