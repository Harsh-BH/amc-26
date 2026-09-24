"""Build the P1 analysis sample from the train world (kept small: the probe is memory-capped).

30k S1 with their full clusters (~100k true pairs) + 100k random same-country S2/S3 -> /tmp/p1/samp_*.parquet
"""
import sys, resource
sys.path[:0] = ["code/business_entity_resolution/src"]
import numpy as np, pandas as pd
from worlds import WORK

W = WORK / "worlds" / "train"
s1 = pd.read_parquet(W / "s1.parquet")
links = pd.read_parquet(W / "truth.parquet")
a = s1.sample(30_000, random_state=0)
kids = links[links.s1.isin(a.entity_id)]
sx = pd.read_parquet(W / "sx.parquet")
rng = np.random.default_rng(0)
ra = a.sample(100_000, replace=True, random_state=1)[["entity_id", "country"]].reset_index(drop=True)
rb = np.empty(len(ra), dtype=object)
for c in ra.country.unique():
    m = (ra.country == c).to_numpy()
    rb[m] = rng.choice(sx.entity_id[sx.country == c].to_numpy(), m.sum())
ra["sx"] = rb
ra = ra.rename(columns={"entity_id": "s1"})[["s1", "sx"]]
sx = sx[sx.entity_id.isin(set(kids.sx) | set(ra.sx))]
a.to_parquet("/tmp/p1/samp_s1.parquet", index=False)
sx.to_parquet("/tmp/p1/samp_sx.parquet", index=False)
kids.to_parquet("/tmp/p1/samp_true.parquet", index=False)
ra.to_parquet("/tmp/p1/samp_rand.parquet", index=False)
print(len(a), "S1", len(kids), "true pairs", len(ra), "random pairs", len(sx), "sx rows",
      "peak GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
