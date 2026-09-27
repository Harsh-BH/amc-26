"""P50 integrator: P55's cached France combiner score c for other pieces' France adds (conflict check).
    cc_check.py NEW OLD   France pairs NEW adds vs OLD -> c distribution (c < .6 = P55 would veto, > .95 = would add)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT / "probes/P55-fr-combiner"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p42, p55  # noqa: E402

z = p42.fr_frame()
f = p55.france_frame()
cf = np.load(ROOT / "data/work/p55/cf_plain.npy")
i1, ix = p42.fr_ids()
key = pd.Series(cf, index=pd.MultiIndex.from_arrays([i1[z["s1"][f["rows"]]], ix[z["sx"][f["rows"]]]]))
key = key[~key.index.duplicated()]
new, old = p42.read_fr(ROOT / "output" / sys.argv[1]), p42.read_fr(ROOT / "output" / sys.argv[2])
ex = lambda s: set((a, x) for a, v in s.items() if v for x in v.split(","))
add, rm = ex(new) - ex(old), ex(old) - ex(new)
for name, s in (("adds", add), ("removes", rm)):
    if not s:
        print(name, 0)
        continue
    c = key.reindex(pd.MultiIndex.from_tuples(sorted(s))).to_numpy()
    print(f"{sys.argv[1]} vs {sys.argv[2]} {name}: {len(s):,}; in P55 frame {np.isfinite(c).sum():,}; c<.6 {(c < .6).sum():,}, "
          f".6-.95 {((c >= .6) & (c <= .95)).sum():,}, >.95 {(c > .95).sum():,}; median c {np.nanmedian(c):.3f}", flush=True)
