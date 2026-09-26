"""P40 step 4c: every op as a France add-class under LOCO (both directions) + France volume, at FR t 0.9 / 0.7.
Scaled LB = LOCO dF per added pair x France pairs per S1 x 0.1497.
    P37_BASE=output/sub37_gc70 probes/P40-generator/py.sh probes/P40-generator/loco_allops.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loco40 as L  # noqa: E402
import ops as O  # noqa: E402
from cand import raw  # noqa: E402
from cat import OUT  # noqa: E402
from loss import bits  # noqa: E402

P = L.P
TS_ = (0.7, 0.9)
dens = L.fr_dens()
res = {}
for src, tgt in (("US", "India"), ("India", "US")):
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["y"], a["sib"], a["n_s1"]
    ntrue, tb = a["t2"] + a["t3"], P.TS[(src, tgt)]
    tmap = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tmap[P.T_RESCUE])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tmap[L.T_GC], "oneall")
    per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
    f0 = per(pred)
    pool = am & ~pred & (p >= tmap[min(TS_)])
    ids1, idsx = a["s1_ids"][pool], a["sx_ids"][pool]
    r1, rx = raw(ids1, (1,)), raw(idsx, (2, 3))
    B = bits(O.tag_frame(r1.business_name.reindex(ids1).to_numpy(), r1.business_address.reindex(ids1).to_numpy(),
                         rx.business_name.reindex(idsx).to_numpy(), rx.business_address.reindex(idsx).to_numpy(),
                         np.full(pool.sum(), tgt)))
    for o in O.OPS:
        cls = np.zeros(len(p), bool)
        cls[pool] = B[o].to_numpy()
        for t in TS_:
            add = L.add_rule(s1, sx, is2, p, am, pred, sib, cls, tmap[t])
            if add.sum() < 20:
                continue
            dl = per(pred | add) - f0
            res.setdefault(f"{o}@{t}", {})[f"{src}->{tgt}"] = dict(d=dl.mean(), added=int(add.sum()), prec=float(y[add].mean()),
                                                                   pp=dl.sum() / add.sum())
    print(src, tgt, "done", flush=True)
# France volumes
d, a = P.fr_load()
s1, sx, is2, p, am, pred, sib = (a[k] for k in ("s1", "sx", "is2", "p", "am", "pred", "sib"))
fr = pd.read_parquet(OUT / "fr_pairs.parquet", columns=["s1", "sx", "ops"])
om = pd.Series(fr.ops.to_numpy(), index=pd.MultiIndex.from_arrays([fr.s1, fr.sx]))
pool = am & ~pred & (p >= min(TS_))
Bf = bits(om.reindex(pd.MultiIndex.from_arrays([d.s1.to_numpy()[pool], d.sx.to_numpy()[pool]])).to_numpy().astype(np.uint64))
rows = []
for o in O.OPS:
    cls = np.zeros(len(p), bool)
    cls[pool] = Bf[o].to_numpy()
    for t in TS_:
        k = f"{o}@{t}"
        nfr = int(L.add_rule(s1, sx, is2, p, am, pred, sib, cls, t).sum())
        r = res.get(k, {})
        row = dict(rule=k, france_added=nfr)
        for dirn in ("US->India", "India->US"):
            x = r.get(dirn)
            row[f"{dirn}_added"], row[f"{dirn}_prec"] = (x["added"], round(x["prec"], 3)) if x else (0, np.nan)
            row[f"{dirn}_lb_scaled"] = round(x["pp"] * nfr / 259452 * 0.1497, 6) if x else np.nan
        rows.append(row)
R = pd.DataFrame(rows).set_index("rule")
R["min_lb"] = R[["US->India_lb_scaled", "India->US_lb_scaled"]].min(axis=1)
R = R.sort_values("min_lb", ascending=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 150)
print(R.head(40).to_string())
R.to_csv(OUT / "loco_allops.csv")
