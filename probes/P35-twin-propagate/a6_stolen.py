"""P35 a6: address-less SX whose true holdout S1 is a candidate but not the argmax ('stolen'). Who holds the argmax
(holdout S1 / competing train S1, e5 row or not), how close are the two p, and do load/twin features separate them?"""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd
from ev import load

gt = pd.read_parquet("data/work/gt_pairs.parquet")
own = gt.drop_duplicates("sx").set_index("sx").s1
for c in ("India", "US"):
    S = load(c, ["s1", "sx", "holdout", "y", "p", "am", "pred", "noaddr", "p_m2", "r"]); d = S["d"]
    x = pd.read_parquet(f"data/work/p35/gx_{c}.parquet")
    d = d.loc[x.index].join(x[["n_conf", "n_conf_src", "tw_p", "sim1", "n_same1"]])
    d["e5"] = (d.p_m2 > 0.02) & (d.p_m2 < 0.98) & d.r.isna()
    d["true"] = d.s1.to_numpy() == d.sx.map(own).to_numpy()          # full train truth, holdout or not
    ysx = set(d.sx[d.y & ~d.am])
    st = d[d.sx.isin(ysx)]
    a = st[st.am].set_index("sx"); t = st[st.y].set_index("sx")
    print(f"\n== {c}: stolen address-less SX {len(ysx):,}; argmax holder holdout {a.holdout.mean():.3f}, "
          f"argmax row e5 {a.e5.mean():.3f}, argmax is TRUE (dup truth) {a.true.mean():.3f}")
    j = t.join(a, lsuffix="_t", rsuffix="_a")
    print("  p_true median", j.p_t.median().round(3), "p_argmax median", j.p_a.median().round(3),
          "| p_a >= .71:", (j.p_a >= .71).mean().round(3))
    for f in ("n_conf", "n_conf_src", "sim1", "n_same1"):
        print(f"  {f}: true>argmax {(j[f+'_t'] > j[f+'_a']).mean():.3f}  equal {(j[f+'_t'] == j[f+'_a']).mean():.3f}  true<argmax {(j[f+'_t'] < j[f+'_a']).mean():.3f}")
    print("  twin at true only", ((j.tw_p_t > 0) & (j.tw_p_a == 0)).mean().round(3), "at argmax only",
          ((j.tw_p_t == 0) & (j.tw_p_a > 0)).mean().round(3), "both", ((j.tw_p_t > 0) & (j.tw_p_a > 0)).mean().round(3))
    # contrast: correctly-assigned address-less multi-candidate SX (argmax true & pred), vs its runner-up
    ok = d[d.am & d.y & d.pred & (d.groupby("sx").s1.transform("size") > 1)].set_index("sx")
    ru = d[~d.am & d.sx.isin(ok.index)].sort_values("p", ascending=False).drop_duplicates("sx").set_index("sx")
    k = ok.join(ru, lsuffix="_t", rsuffix="_a", how="inner")
    for f in ("n_conf", "n_conf_src"):
        print(f"  [correct TP vs runner-up] {f}: true>ru {(k[f+'_t'] > k[f+'_a']).mean():.3f} equal {(k[f+'_t'] == k[f+'_a']).mean():.3f} true<ru {(k[f+'_t'] < k[f+'_a']).mean():.3f}")
