"""P35 a7: (1) rule-A precision where the matcher is blind: SX core far from X's core but near a confident twin's;
(2) artefact-aware budget of the address-less loss: oracle-fix one category of address-less SX at a time."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load, gain

for c in ("India", "US"):
    S = load(c, ["s1", "sx", "holdout", "y", "p", "am", "pred", "noaddr", "p_m2", "r"]); d = S["d"]
    x = pd.read_parquet(f"data/work/p35/gx_{c}.parquet", columns=["tw_p", "tw_sim", "sim1"])
    k = d.loc[x.index].join(x)
    A = k[k.holdout & k.am & ~k.pred & (k.tw_p > 0)]
    print(f"\n== {c} rule-A by sim1 (SX core vs S1 core):",
          A.groupby(pd.cut(A.sim1, [-.01, .5, .7, .85, .95, 1]), observed=True).y.agg(["size", "mean"]).round(3).to_dict())
    na, y, am, pred = d.noaddr.to_numpy(), d.y.to_numpy(), d.am.to_numpy(), d.pred.to_numpy()
    e5 = ((d.p_m2 > 0.02) & (d.p_m2 < 0.98) & d.r.isna()).to_numpy()
    q = pd.factorize(d.sx)[0]
    anyf = lambda m: pd.Series(m).groupby(q).transform("any").to_numpy()
    sx_y = anyf(y)                                   # SX has a true holdout S1 among its candidates
    sx_am_y = anyf(am & y)
    thief_e5 = anyf(am & e5)
    cats = {"lowp_FN (argmax true, p<t)": na & sx_am_y & anyf(am & y & ~pred),
            "stolen, thief e5": na & sx_y & ~sx_am_y & thief_e5,
            "stolen, thief clean": na & sx_y & ~sx_am_y & ~thief_e5,
            "FP (pred wrong, no true cand)": na & anyf(pred & ~y) & ~sx_y,
            "FP (pred wrong, true cand)": na & anyf(pred & ~y) & sx_y}
    for n, m in cats.items():
        g = gain(S, np.where(m, y, pred))
        print(f"  {n}: {int(len(np.unique(q[m & (S['g'] >= 0)])))} SX, oracle fix {g['all']:+.5f}")
    miss = pd.read_parquet(f"data/work/p23/missed_{c}.parquet")
    print(f"  blocking misses (true pair not a candidate): {len(miss):,} pairs")
