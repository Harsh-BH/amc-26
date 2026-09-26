"""P40: address-less true pairs: FN rate / loss by the true S1's core-name twin count (full train). Aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT
from loss import bits

tp = pd.read_parquet(OUT / "tp.parquet")
B = bits(tp.ops.to_numpy())
tp["noaddr"] = B.A_EMPTY.to_numpy()
E = pd.read_parquet(OUT / "err_w.parquet")
fn = E[E.fn]
key = dict(zip(zip(fn.s1, fn.sx), zip(fn.type, fn.lb)))
tp["fn"] = [(a, b) in key for a, b in zip(tp.s1, tp.sx)]
tp["lb"] = [key.get((a, b), ("", 0.0))[1] for a, b in zip(tp.s1, tp.sx)]
tw = pd.concat([pd.read_parquet(OUT / f"s1twin_{c}.parquet") for c in ("India", "US")]).set_index("id")
tp["k"] = tw.k.reindex(tp.s1).to_numpy()
tp["exact_name"] = B.N_EXACT.to_numpy() | B.N_CASE.to_numpy()
for nm, d in (("address-less", tp[tp.noaddr]), ("with address", tp[~tp.noaddr])):
    print(nm, "by true-S1 twin count k: n, FN rate, LB lost")
    print(d.groupby([d.c, d.k.clip(upper=4)]).agg(n=("fn", "size"), fn=("fn", "mean"), lb=("lb", "sum")).round(5).to_string())
