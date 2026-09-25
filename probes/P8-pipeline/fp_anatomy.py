"""Where the dense-holdout (full train corpus) rule false positives come from: parentless S2/S3 (distractors) vs
records of another S1; singleton vs non-singleton target; same name core as the true parent (chain) or not."""
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src")]
from metric import read_sets
from worlds import WORK
pred = read_sets(ROOT / "output/holdout/matching_results.tsv")
t = pd.read_parquet(WORK / "worlds/holdout/truth.parquet")
truth = {s: set() for s in pred}
for a, b in zip(t.s1, t.sx):
    truth.setdefault(a, set()).add(b)
fp = pd.DataFrame([(s, x) for s, v in pred.items() for x in v - truth[s]], columns=["s1", "sx"])
gt = pq.read_table(WORK / "gt_pairs.parquet")
gt = gt.filter(pc.is_in(gt["sx"], value_set=pa.array(fp.sx.unique()))).to_pandas().rename(columns={"s1": "parent"})
fp = fp.merge(gt, on="sx", how="left")
fp["singleton_target"] = fp.s1.map(lambda s: not truth[s])
core = pd.concat([pd.read_parquet(WORK / f"rf_train_s1_{c}.parquet", columns=["id", "core"]) for c in ("India", "US")]).set_index("id").core
fp["same_core_parent"] = fp.parent.notna() & (fp.s1.map(core) == fp.parent.map(core))
n_pred = sum(map(len, pred.values()))
print(f"predicted pairs {n_pred:,}, FP {len(fp):,} ({len(fp) / n_pred:.3f})")
print("FP sx has a true parent:", fp.parent.notna().mean().round(3), " parent is another holdout S1:",
      fp.parent.isin(set(truth)).mean().round(3))
print("FP to a singleton S1:", fp.singleton_target.mean().round(3), " parent has same core (chain):", fp.same_core_parent.mean().round(3))
print(pd.crosstab(fp.parent.notna().rename("has_parent"), fp.singleton_target, normalize=True).round(3))
single = [s for s in truth if not truth[s]]
print(f"singletons {len(single):,}, with >=1 FP {sum(bool(pred[s]) for s in single) / len(single):.3f}")
