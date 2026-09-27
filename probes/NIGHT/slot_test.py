# Label-free "missing slot" test: an SX that truly belongs to S1 fills one of S1's ~3.46 truth slots, so S1's OTHER
# accepted members should be fewer than for an S1 whose candidate is a decoy. Calibrate on US holdout, apply to France.
import numpy as np, pandas as pd
W = "data/work"
d = pd.read_parquet(f"{W}/p23/pairs_US.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred", "p"]); d = d[d.holdout].reset_index(drop=True)
nacc = d[d.pred].groupby("s1").size()
c = d[d.am & ~d.pred & d.s1.isin(nacc.index)].copy()          # rejected argmax candidates of S1 with >= 1 accepted
c["others"] = nacc.reindex(c.s1).to_numpy()
print("US holdout, rejected argmax candidates: mean other-accepted | true", round(c[c.y].others.mean(), 3), "false", round(c[~c.y].others.mean(), 3),
      "| all accepted-S1 mean", round(nacc.mean(), 3))
print("US by p band:"); print(c.assign(pb=pd.cut(c.p, [0, .1, .3, .5, .71])).groupby(["pb", "y"], observed=True).others.mean().unstack().round(3).to_string())
fr = pd.read_parquet(f"{W}/p31/pairs_France.parquet", columns=["s1", "sx", "p", "acc", "argmax"])
facc = fr[fr.acc].groupby("s1").size()
print("France accepted-S1 mean", round(facc.mean(), 3))
n55 = pd.read_parquet(f"{W}/n55/cells.parquet"); n54 = pd.read_parquet(f"{W}/p54/cells.parquet")
taken = set(fr.sx[fr.acc])
def show(name, x):
    x = x[~x.acc & ~x.sx.isin(taken)] if "acc" in x else x
    x = x[x.s1.isin(facc.index)]
    o = facc.reindex(x.s1).to_numpy()
    print(f"  {name:38s} n {len(x):6d}  mean other-accepted {o.mean():.3f}  share with >=4: {(o >= 4).mean():.3f}")
for w in ("fils", "cie", "associes", "groupe", "developpement"):
    show(f"N55/P46 word '{w}'", n55[n55.w == w])
n54f = n54.merge(fr[["s1", "sx", "acc", "argmax"]], on=["s1", "sx"]); n54f = n54f[n54f.argmax.astype(bool)]
show("N54 decoy words (unaccepted, argmax)", n54f)
