"""P40: how the generator makes address-less SX (holdout world truth). Prints aggregates only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat import OUT, world
from loss import bits

s1, sx, t = world()
tp = pd.read_parquet(OUT / "tp.parquet")
B = bits(tp.ops.to_numpy())
tp["empty"] = B.A_EMPTY.to_numpy()
tp["src"] = tp.sx.str[:2]
e = pd.concat([pd.read_parquet(OUT / f"err_{c}.parquet") for c in ("India", "US")])
fnk = set(zip(e.s1[e.type.isin(["fp_ns", "fp_s"]) == False], e.sx[e.type.isin(["fp_ns", "fp_s"]) == False]))
tp["fn"] = [(a, b) in fnk for a, b in zip(tp.s1, tp.sx)]
print("empty share of true pairs by c/src:\n", tp.groupby(["c", "src"]).empty.mean().round(4).to_string())
# per (s1, src) consistency: of groups with >=2 members, share where empties are all-or-none
g = tp.groupby(["s1", "src"]).empty.agg(["sum", "size"])
g2 = g[g["size"] >= 2]
mixed = ((g2["sum"] > 0) & (g2["sum"] < g2["size"])).mean()
allm = (g2["sum"] == g2["size"]).mean()
print(f"(s1,src) groups >=2: {len(g2)}; all-empty {allm:.4f}; mixed {mixed:.4f}; none {1-mixed-allm:.4f}")
print("P(empty) given k members:", g.assign(r=g["sum"]/g["size"]).groupby("size").r.mean().round(4).head(8).to_dict())
# per-S1: how many empty SX per S1
ps = tp.groupby("s1").empty.sum()
print("empty SX per S1 dist:", ps.value_counts().sort_index().head(6).to_dict())
# name ops of empty SX
Bm = B[tp.empty.to_numpy()]
print("name-op rates, empty SX vs addr SX:")
r = pd.DataFrame({"empty": Bm.mean(), "addr": B[~tp.empty.to_numpy()].mean(),
                  "fn_empty": [tp.fn.to_numpy()[tp.empty.to_numpy() & B[o].to_numpy()].mean() if (tp.empty.to_numpy() & B[o].to_numpy()).any() else np.nan for o in B.columns]})
print(r[r.index.str.startswith("N_")].round(4).to_string())
print("FN rate empty by c:", tp[tp.empty].groupby("c").fn.mean().round(4).to_dict())
