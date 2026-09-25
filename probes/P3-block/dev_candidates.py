"""End-to-end check of block.candidates() with the recommended CFG on a world (vecs fitted on the 300k train sample)."""
import sys, time, resource
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), "code/business_entity_resolution/src"]
import pandas as pd, block as B, evalb as E
from worlds import load_world

world = sys.argv[1] if len(sys.argv) > 1 else "dev"
s1, sx, truth = load_world(world)
vecs = B.fit_views(pd.read_parquet("data/work/p3/fit300k.parquet"), B.CFG, sample=300_000)
t0 = time.time()
c = B.candidates(s1, sx, B.CFG, vecs)
secs = time.time() - t0
tp = E.truth_df(truth); tp["country"] = tp.s1.map(s1.set_index("entity_id").country); tp["cls"] = E.classify(tp, s1, sx)
m = E.metrics(c[["s1", "sx"]], tp, s1)
m["s1_no_cands"] = 1 - c.s1.nunique() / len(s1)
print({k: round(v, 4) for k, v in m.items()}, f"{secs:.0f}s", "peak GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
print(c.via.value_counts(normalize=True).round(3).to_dict())
