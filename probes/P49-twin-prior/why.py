"""P49: what sub44r already does on the ambiguous SX, overall and on test 3's slice (raw names differ, unique raw-closest
twin). Aggregates only.   probes/P49-twin-prior/py.sh probes/P49-twin-prior/why.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dec as DC  # noqa: E402

for c in DC.LB_W:
    d = DC.load(c)
    tw = DC.feats(c, d["tw"])
    tw = tw[tw.groupby("sx").holdout.transform("any")]
    tw["own"] = tw.groupby("sx").y_full.transform("any")
    am = tw[tw.am]
    ho_am = am[am.holdout]
    print(f"{c}: ambiguous SX with a holdout twin {tw.sx.nunique():,}; owner in set {tw.groupby('sx').own.first().mean():.3f}; "
          f"argmax at a holdout twin {len(ho_am):,}, sub44r accepts {ho_am.acc.mean():.3f}, precision {ho_am[ho_am.acc].y_full.mean():.3f}; "
          f"argmax p'' median {ho_am.pa.median():.3f}", flush=True)
    sl = tw[(tw.rawdiff > 0) & (tw.d_min_uniq > 0)]
    xs = set(sl.sx)
    a = am[am.sx.isin(xs)]
    same = a.s1.to_numpy() == sl.set_index("sx").s1.reindex(a.sx).to_numpy()
    print(f"   test-3 slice: {len(xs):,} SX; the argmax IS the raw-closest twin {same.mean():.3f}; raw-closest is the owner "
          f"{sl.y_full.mean():.3f}; sub44r accepts the argmax {a.acc.mean():.3f} (precision {a[a.acc].y_full.mean():.3f}); "
          f"accepted with argmax = closest {(a.acc & same).mean():.3f}", flush=True)
