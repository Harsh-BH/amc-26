"""Dense accepted pairs with addr score exactly 4: house numbers both present and different, vs one side missing."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
import rules as R
from metric import read_sets
from worlds import WORK
t = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
tk = set(zip(t.s1, t.sx))
pred = read_sets(ROOT / "output/holdout/matching_results.tsv")
pr = pd.DataFrame([(s, x) for s, v in pred.items() for x in v], columns=["s1", "sx"])
f1 = pd.concat([pd.read_parquet(WORK / f"rf_train_s1_{c}.parquet") for c in ("India", "US")]).set_index("id")
fx = pd.concat([pd.read_parquet(WORK / f"rf_train_sx_{c}.parquet") for c in ("India", "US")])
fx = fx[fx.id.isin(set(pr.sx))].set_index("id")
a, b = f1.loc[pr.s1].reset_index(drop=True), fx.loc[pr.sx].reset_index(drop=True)
pr["sc"] = R.addr_score(a, b)
pr["true"] = [k in tk for k in zip(pr.s1, pr.sx)]
h1, hx = a.hn.to_numpy() != "", b.hn.to_numpy() != ""
pr["hn"] = pd.Series(pd.Categorical.from_codes((h1 & hx) * 2 + (h1 ^ hx) * 1, ["neither", "one side", "both, differ"]))
g = pr[pr.sc == 4]
print(pd.crosstab(g.hn, g.true, margins=True))
print("precision of sc=4 by hn state:", g.groupby("hn", observed=True).true.mean().round(3).to_dict())
d = pr[~((pr.sc == 4) & (pr.hn == "both, differ"))]
print(f"drop sc=4 & both hn differ: pairs {len(d):,} precision {d.true.mean():.4f} (was {pr.true.mean():.4f})")
