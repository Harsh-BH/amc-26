"""France test rules-features with the france.py layer, chunked single-process (2 GB cap):
data/work/p6/rf_test_{s1,sx}_France_fr.parquet, same columns as data/work/rf_test_*_France.parquet."""
import resource, sys, time
from pathlib import Path
import pandas as pd
sys.path[:0] = [str(Path(__file__).parent)]
import france as F

W = Path("data/work")
for side, files in (("s1", [1]), ("sx", [2, 3])):
    out = W / f"p6/rf_test_{side}_France_fr.parquet"
    if out.exists():
        continue
    parts = []
    for k in files:
        d = pd.read_parquet(W / f"test_source{k}.parquet", filters=[("country", "==", "France")])
        for i in range(0, len(d), 200_000):
            t0 = time.time()
            parts.append(F.feats(d.iloc[i:i + 200_000]))
            print(side, k, i, f"{time.time() - t0:.0f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f}GB", flush=True)
        del d
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
