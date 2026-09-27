"""P76: R3's label null. The fixed rule (t +3 steps on n_am == '2') on S1 labels permuted across S1, 8 perms: is it the
segment, or would stricter t on any same-size random S1 set do as well?  -> data/work/p76/r3null.json"""
import json
import sys

import numpy as np

sys.path.insert(0, "probes/P76-segment-hill")
import fr_hill as H  # noqa: E402

out = {}
for perm in [None] + list(range(1, 9)):
    D = H.load(perm)
    V = H.vals(D)
    H.attach(D, V, H.FAMS)
    sh = H.zero(V, H.FAMS)
    sh[("t", "n_am")][V["n_am"].index("2")] = 3
    r = {}
    for k, d in D.items():
        x = H.per_s1(d, sh) - d["f0"]
        r["|".join(k)] = [float(x[d["hv"] == h].mean()) for h in (0, 1)]
    out[str(perm)] = r
    print(perm, {k: [round(v, 6) for v in vv] for k, vv in r.items()}, flush=True)
json.dump(out, open("data/work/p76/r3null.json", "w"), indent=1)
