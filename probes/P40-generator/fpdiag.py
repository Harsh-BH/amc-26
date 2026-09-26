"""P40: FP pairs (distractor kinds) and generic-word FNs by house-number relation / name ops. Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT
from loss import bits, fp_kind

E = pd.read_parquet(OUT / "err_w.parquet")
B = bits(E.ops.to_numpy())
hn = np.select([B.A_EMPTY, B.A_HN_SAME, B.A_HN_SIB, B.A_HN_SMALL, B.A_HN_OTHER, B.A_HN_LEADDROP, B.A_HN_PREFIXNUM, B.A_HN_DROP, B.A_HN_ADD],
               ["noaddr", "same", "sib3", "d<=2", "other", "leaddrop", "prefixnum", "drop", "add"], "none")
nm = np.select([B.N_EXACT | B.N_CASE, B.N_GIB_NAME, B.N_UNRELATED, B.N_GENERIC_ADD, B.N_WEB | B.N_HANDLE, B.N_SCRIPT],
               ["exact/case", "gib", "unrelated", "generic_add", "web", "script"], "other")
E["hn"], E["nm"] = hn, nm
fp = E[~E.fn].assign(kind=fp_kind(E[~E.fn]))
print("FP pairs: kind x hn relation (LB-eq x1e5)")
print((fp.pivot_table(index="hn", columns="kind", values="lb", aggfunc="sum").fillna(0) * 1e5).round(1).to_string())
print("FP pairs: name relation x hn (count)")
print(fp.pivot_table(index="nm", columns="hn", values="lb", aggfunc="size").fillna(0).astype(int).to_string())
fn = E[E.fn & B.N_GENERIC_ADD.to_numpy()]
print("generic-add FN: hn x type (LB x1e5)")
print((fn.pivot_table(index="hn", columns="type", values="lb", aggfunc="sum").fillna(0) * 1e5).round(1).to_string())
