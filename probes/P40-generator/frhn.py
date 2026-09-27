"""P40: France argmax pairs by hn relation x p band (per S1), vs US/India holdout argmax pairs. Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT
from loss import bits


def hncat(B):
    return np.select([B.A_EMPTY, B.A_HN_SAME, B.A_HN_SIB, B.A_HN_SMALL, B.A_HN_OTHER, B.A_HN_LEADDROP, B.A_HN_PREFIXNUM,
                      B.A_HN_DROP, B.A_HN_ADD], ["noaddr", "same", "sib3", "d<=2", "other", "leaddrop", "prefixnum", "drop", "add"], "none")


fr = pd.read_parquet(OUT / "fr_pairs.parquet")
B = bits(fr.ops.to_numpy())
fr["hn"] = hncat(B)
fr["band"] = np.where(fr.acc, "acc", pd.cut(fr.p, [0, .05, .3, .5, .7, .9, 1.01], right=False).astype(str))
t = fr.pivot_table(index="hn", columns="band", values="p", aggfunc="size").fillna(0) / 259452
print("France argmax pairs per S1 by hn relation x (accepted | rejected p band)")
print(t.round(4).to_string())
for c in ("US", "India"):
    am = pd.read_parquet(OUT / f"am_{c}.parquet")
    Ba = bits(am.ops.to_numpy())
    am["hn"] = hncat(Ba)
    n = am.g.max() + 1
    x = am.assign(k=np.where(am.acc, "acc", "rej")).pivot_table(index="hn", columns="k", values="y", aggfunc=["size", "mean"]).fillna(0)
    x[("size", "acc")] /= n
    x[("size", "rej")] /= n
    print(c, "holdout argmax pairs per S1 (size) and true rate (mean) by hn relation")
    print(x.round(4).to_string())
