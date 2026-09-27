"""P58: where is the oracle? Oracle re-decision of the expanded frame split by SX class: twins' raw names differ (rawdiff) vs
raw-identical twins (only counts can separate them), by group size, and owner new vs already a candidate.
    probes/P58-twin-cands/py.sh probes/P58-twin-cands/split58.py"""
import numpy as np
import d58
from d58 import fb, LB_W

for c in LB_W:
    d, ex = d58.load(c)
    f0 = fb(d["tp"], d["npred"], d["n_true"])
    sxc = ex.groupby("sx").agg(rd=("rawdiff", "max"), k=("kfull", "max"),
                               newown=("in_set", lambda v: False), ).reset_index()
    own_new = ex[ex.y].groupby("sx").in_set.max() == 0
    parts = {"rawdiff": ex.rawdiff > 0, "raw-identical": ex.rawdiff == 0, "K<=3": ex.kfull <= 3, "K4-10": (ex.kfull > 3) & (ex.kfull <= 10),
             "K11-30": ex.kfull > 10, "owner new pair": ex.sx.isin(own_new[own_new].index), "owner old pair": ex.sx.isin(own_new[~own_new].index)}
    msg = []
    for name, m in parts.items():
        sub = ex[m.to_numpy()]
        dl = fb(*[x for x in (lambda t: t)(0) or ()], ) if False else None
        s = sub.y.to_numpy().astype(float)
        dl = d58.apply(d, sub, s, 0.5, 0.5) - f0
        msg.append(f"{name} {dl.mean():+.6f} ({sub.sx.nunique():,} SX)")
    print(c, " | ".join(msg), flush=True)
