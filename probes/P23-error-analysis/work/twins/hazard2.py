"""hazard.py re-scored against the FULL train truth (dump y marks holdout-S1 truth only)."""
import numpy as np, pandas as pd
W = "data/work/"
q = pd.read_parquet(W + "p23/twins_hazard.parquet")
gt = pd.read_parquet(W + "gt_pairs.parquet")
q = q.merge(gt.assign(yt=True), on=["s1", "sx"], how="left")
q["yt"] = q.yt.fillna(False).astype(bool)
own = q.groupby("sx").yt.transform("any")
print(f"contested address-less SX: {q.sx.nunique():,}; owner among its candidates: {q[own].sx.nunique():,}; "
      f"distractor/owner not a candidate: {q[~own].sx.nunique():,}")
lo = np.log(np.clip(q.p, 1e-4, 1 - 1e-4) / (1 - np.clip(q.p, 1e-4, 1 - 1e-4)))
for a in (0.0, 0.5, 1.0, 2.0):
    q["w"] = np.exp(a * lo) * q.h
    q["post"] = q.w / q.groupby("sx").w.transform("sum")
    top = q.loc[q.groupby("sx").post.idxmax()]
    cur = q.loc[q.groupby("sx").p.idxmax()]
    print(f"\nalpha={a}: top-posterior is the owner {top.yt.mean():.3f} (argmax by p today: {cur.yt.mean():.3f}); "
          f"among SX whose owner IS a candidate: {top[top.sx.isin(q[own].sx)].yt.mean():.3f} vs {cur[cur.sx.isin(q[own].sx)].yt.mean():.3f}")
    for tau in (0.6, 0.7, 0.75, 0.8, 0.9):
        t = top[top.post >= tau]
        new = t[~t.pred]
        print(f"   post>={tau}: {len(t):6,} SX, precision {t.yt.mean():.3f} | not accepted today: {len(new):5,}, precision "
              f"{new.yt.mean() if len(new) else float('nan'):.3f} (India {int((new.country=='India').sum())}, US {int((new.country=='US').sum())})")
