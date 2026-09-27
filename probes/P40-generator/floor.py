"""P40: Bayes floor. Loss on error pairs whose SX has no address and whose true S1's normalised FULL name (legal forms
kept) is shared by >= 2 train S1 of the country: the SX carries no evidence that separates the twins. Aggregates only."""
import re
import sys
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT, W
from loss import bits


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))


t = pq.read_table(W / "train_source1.parquet", columns=["entity_id", "business_name", "country"]).to_pandas()
t["k"] = [key(x) for x in t.business_name]
cnt = t.groupby(["country", "k"]).entity_id.transform("size")
kfull = pd.Series(cnt.to_numpy(), index=t.entity_id.to_numpy())
del t
E = pd.read_parquet(OUT / "err_w.parquet")
B = bits(E.ops.to_numpy())
E["noaddr"] = B.A_EMPTY.to_numpy()
E["kfull"] = kfull.reindex(E.s1).to_numpy()
E["tie"] = E.noaddr & (E.kfull >= 2)
tot = E.lb.sum()
print(f"total LB {tot:.6f}")
for nm, m in (("addr-less FN, exact full-name twin (k>=2)", E.fn & E.tie), ("addr-less FN, unique full name", E.fn & E.noaddr & (E.kfull == 1)),
              ("addr-less FP (SX assigned to a wrong S1)", ~E.fn & E.noaddr)):
    print(f"{nm}: pairs {int(m.sum())}, LB {E.lb[m].sum():.6f} ({E.lb[m].sum() / tot:.3f})")
print("addr-less FN by kfull:", E[E.fn & E.noaddr].groupby(E.kfull.clip(upper=5)).lb.sum().round(6).to_dict())
