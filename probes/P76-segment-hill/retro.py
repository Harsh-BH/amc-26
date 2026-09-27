"""P76: does the LOCO chain analogue retrodict the two LB-measured France bar steps? rescue90 (LB +0.000313) and gc70
(LB +0.000344) -> France-F ~ LB / 0.1498. LOCO dF of each step on the sub66 chain analogue, both directions, both analogues.
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/retro.py      -> data/work/p76/retro.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "probes/P76-segment-hill")
import fr_hill as H  # noqa: E402

D = H.load()
out = {}
for k, d in D.items():
    tm = d["tm"]
    inf = np.full(len(d["p"]), np.inf)
    t = np.full(len(d["p"]), tm["0.9868"])

    def f(r, gc, hn=inf):
        pred = H.fast_decide(d, t, r, gc, hn)
        return H.S.fbeta(np.bincount(d["s1"][pred & d["y"]], minlength=d["n"]), np.bincount(d["s1"][pred], minlength=d["n"]), d["ntrue"])
    r9, g7 = np.full(len(d["p"]), tm["0.9"]), np.full(len(d["p"]), tm["0.7"])
    f0, f1, f2 = f(inf, inf), f(r9, inf), f(r9, g7)
    out["|".join(k)] = {"rescue90": [float((f1 - f0)[d["hv"] == h].mean()) for h in (0, 1)],
                        "gc70": [float((f2 - f1)[d["hv"] == h].mean()) for h in (0, 1)]}
out["lb_implied_franceF"] = {"rescue90": 0.000313 / 0.1498, "gc70": 0.000344 / 0.1498}
json.dump(out, open("data/work/p76/retro.json", "w"), indent=1)
print(json.dumps(out, indent=1))
