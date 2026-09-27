"""P35 a3: address-less SX whose argmax holdout S1 X has a confident addressed twin, but y=False: where do they belong?
(train truth for all folds: distractor / another S1 among its candidates / another S1 outside them)"""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load
from twins import twin_frame

gt = pd.read_parquet("data/work/gt_pairs.parquet")
own = gt.drop_duplicates("sx").set_index("sx").s1
for c in ("India", "US"):
    S = load(c); d = S["d"]
    tw = twin_frame(d, d.am.to_numpy(), 0.71, 0.85)
    k = d.merge(tw, on=["s1", "sx"], how="left"); k["has"] = k.tw_p.notna()
    cand = set(zip(k.s1, k.sx))
    for name, m in (("A all am&~pred&sup", k.holdout & k.noaddr & k.am & ~k.pred & k.has),
                    ("A p<0.1", k.holdout & k.noaddr & k.am & ~k.pred & k.has & (k.p < 0.1)),
                    ("A p>=0.3", k.holdout & k.noaddr & k.am & ~k.pred & k.has & (k.p >= 0.3))):
        a = k[m]
        tru = a.sx.map(own)
        kind = np.select([a.y, tru.isna(), [(t, x) in cand for t, x in zip(tru, a.sx)]],
                         ["y", "distractor", "other_S1_cand"], "other_S1_notcand")
        print(c, name, len(a), pd.Series(kind).value_counts(normalize=True).round(3).to_dict())
    # base rate: all address-less SX in the frame
    sx = k[k.noaddr].drop_duplicates("sx")
    print(c, "address-less SX in frame: distractor share", round(sx.sx.map(own).isna().mean(), 3),
          "| addressed SX:", round(k[~k.noaddr].drop_duplicates("sx").sx.map(own).isna().mean(), 3))
