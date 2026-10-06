import sys, csv
sys.path.insert(0, "code/business_entity_resolution/src")
import pandas as pd
from metric import score
import run
t = pd.read_parquet("data/work/worlds/holdout/truth.parquet")
s1 = pd.read_parquet("data/work/worlds/holdout/s1.parquet", columns=["entity_id"]).entity_id
truth = {s: set() for s in s1}
for a, b in zip(t.s1, t.sx): truth[a].add(b)
m = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
pred = {a: set(filter(None, b.split(","))) for a, b in zip(m.source1_entity_id, m.matched_entity_ids)}
h = pd.Series(run.half(list(truth)), index=list(truth))
print({f"half{k}": round(score(pred, {s: v for s, v in truth.items() if h[s] == k})["f05"], 5) for k in (0, 1)},
      "all", round(score(pred, truth)["f05"], 5))
