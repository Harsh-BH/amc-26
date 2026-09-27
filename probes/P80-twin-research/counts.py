"""2-twin groups, owner among the twins: P(the twin with MORE addressed copies owns x), by (fewer, more). Needs frame.py output."""
import sys, numpy as np, pandas as pd
S = sys.argv[1]
x = pd.read_parquet(S + "/amb.parquet"); tw = pd.read_parquet(S + "/s1tw.parquet").set_index("s1")
print(f"ambiguous address-less SX {len(x):,}: owned by a same-key twin {x.own_in.mean():.3f}, "
      f"by another S1 {(x.owner.notna() & ~x.own_in).mean():.3f}, by none {x.owner.isna().mean():.3f}")
x = x[x.own_in & (x.tw.str.len() == 2)].copy()
a, b = np.array([t[0] for t in x.tw]), np.array([t[1] for t in x.tw])
na, nb = tw.naddr.reindex(a).to_numpy(), tw.naddr.reindex(b).to_numpy()
big = np.where(na > nb, a, b)
x["lo"], x["hi"], x["big_owns"] = np.minimum(na, nb).clip(max=4), np.maximum(na, nb).clip(max=6), x.owner.to_numpy() == big
t = x[x.lo != x.hi].groupby(["country", "lo", "hi"]).big_owns.agg(["mean", "size"])
print(t[t["size"] >= 30].round(3).to_string())
