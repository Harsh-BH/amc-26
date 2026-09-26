"""Brief 4 resize, org words only (R1 fillers excluded): flips at t .985 on #9ab by s(w) bin and p band, with the
formula precision P(w) = .875(1-d)/(.875(1-d)+eps d), d = min(1, s/0.68), and the k=3 EV per flip. Aggregates only."""
import numpy as np, pandas as pd
FILL = {"fils", "cie", "associes", "freres", "frs"}
a = pd.read_parquet("data/work/p20/pairs_France.parquet", columns=["s1", "sx", "kind", "word", "hrel"])
s = a[a.kind == "sub1"]; del a
g = s.groupby("word").hrel.agg(n="size", sib3=lambda x: (x == "sib3").mean())
p = pd.read_parquet("data/work/p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"])
p = p.sort_values("p", ascending=False).drop_duplicates("sx")
e = s[(s.hrel == "eq") & ~s.word.isin(FILL)].merge(p, on=["s1", "sx"], how="inner")
e["s"] = e.word.map(g.sib3); e = e[e.word.map(g.n) >= 20]
f = e[e.p < .985]
print(f"org-word sub1/eq argmax (support>=20): {len(e):,}, accepted {(e.p >= .985).mean():.3f}; flips {len(f):,} on {f.s1.nunique():,} S1")
for eps in (0.045, 0.075):
    for norm_ in (0.68, 0.62):
        d = np.minimum(1, f.s / norm_); P = .875 * (1 - d) / (.875 * (1 - d) + eps * d)
        for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, 1.01)):
            k = (f.s > lo) & (f.s <= hi) if lo else (f.s <= hi)
            ev = (P[k] * 0.091 - (1 - P[k]) * 0.242).sum()
            print(f"eps {eps} norm {norm_} s in ({lo},{hi}]: flips {int(k.sum()):,} meanP {P[k].mean():.3f} EV {ev:,.0f} S1-units = {ev*6e-7:+.5f} LB")
for lo, hi in ((0, .5), (.5, .9), (.9, .985)):
    k = (f.p >= lo) & (f.p < hi)
    print(f"flips with p in [{lo},{hi}): {int(k.sum()):,}; s<=0.4 {int((k & (f.s <= .4)).sum()):,}")
