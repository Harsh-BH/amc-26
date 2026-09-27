"""P76: simple per-segment France t rules (read off joint_lo's knobs that both fits agree on) on the LOCO chain analogue,
no fitting: every half x direction x analogue, vs the sub66 recipe. Rules only raise t for a segment (remove pairs).
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/rules.py      -> data/work/p76/rules.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "probes/P76-segment-hill")
import fr_hill as H  # noqa: E402

RULES = {"R1 twin5+ t+2": [("t", "twin", "5+", 2)],
         "R2 twin>=2 t+2": [("t", "twin", "5+", 2), ("t", "twin", "2-4", 2)],
         "R3 n_am2 t+3": [("t", "n_am", "2", 3)],
         "R4 R2+R3": [("t", "twin", "5+", 2), ("t", "twin", "2-4", 2), ("t", "n_am", "2", 3)],
         "R5 all twins t+2": [("t", "twin", v, 2) for v in ("1", "2-4", "5+")],
         "R6 twin5+ t+1": [("t", "twin", "5+", 1)]}
D = H.load()
V = H.vals(D)
H.attach(D, V, H.FAMS)
out = {}
for name, rule in RULES.items():
    sh = H.zero(V, H.FAMS)
    for st, f, v, k in rule:
        sh[(st, f)][V[f].index(v)] = k
    r = {}
    for key, d in D.items():
        pred = H.decide(d, sh)
        x = H.per_s1(d, sh) - d["f0"]
        base = H.decide(d, H.zero(V, H.FAMS))
        rm = base & ~pred
        r["|".join(key)] = {"h0": float(x[d["hv"] == 0].mean()), "h1": float(x[d["hv"] == 1].mean()),
                            "removed_per_s1": float(rm.sum() / d["n"]), "removed_prec": float(d["y"][rm].mean()) if rm.any() else None}
    out[name] = r
    print(name, {k: (round(v["h0"], 6), round(v["h1"], 6), round(v["removed_per_s1"], 4), v["removed_prec"] and round(v["removed_prec"], 3))
                 for k, v in r.items()}, flush=True)
json.dump(out, open("data/work/p76/rules.json", "w"), indent=1)
