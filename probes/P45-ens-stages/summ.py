"""Cross-fit summary of zoo eval JSONs: per arm, h0->h1, h1->h0 and their mean (each S1 reported once, by the model
fit on the other half) ± SE; gate = cross-fit >= +0.00015 at >= 3 SE, both countries >= -0.00005.
    summ.py data/work/p45/zoo_eval_<arms>.json ..."""
import json
import sys

import numpy as np

for f in sys.argv[1:]:
    r = json.load(open(f))
    for k in [k for k in r if k.startswith("h0/") and not k.endswith("_vs_8c")]:
        a, b = r[k], r.get("h1/" + k[3:])
        if not b or "India" not in a:
            continue
        cf = {c: ((a[c]["d"] + b[c]["d"]) / 2, np.hypot(a[c]["se"], b[c]["se"]) / 2) for c in ("India", "US")}
        d, se = (a["lb"]["d"] + b["lb"]["d"]) / 2, np.hypot(a["lb"]["se"], b["lb"]["se"]) / 2
        ok = d >= 0.00015 and d >= 3 * se and min(v[0] for v in cf.values()) >= -0.00005
        print(f"{k[3:]:14s} h0>h1 {a['lb']['d']:+.6f} | h1>h0 {b['lb']['d']:+.6f} | cross-fit {d:+.6f} ± {se:.6f} ({d / se:+.1f} SE) "
              f"India {cf['India'][0]:+.6f} US {cf['US'][0]:+.6f} | {'PASS' if ok else 'fail'}")
