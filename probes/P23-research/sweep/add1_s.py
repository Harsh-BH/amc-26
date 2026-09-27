"""add1/eq cell on #9ab with the same per-word s(w) gate (s over France add1 pairs). Aggregates only."""
import numpy as np, pandas as pd
FILL = {"fils", "cie", "associes", "freres", "frs"}
a = pd.read_parquet("data/work/p20/pairs_France.parquet", columns=["s1", "sx", "kind", "word", "hrel"])
s = a[a.kind == "add1"]; del a
g = s.groupby("word").hrel.agg(n="size", sib3=lambda x: (x == "sib3").mean())
p = pd.read_parquet("data/work/p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"])
p = p.sort_values("p", ascending=False).drop_duplicates("sx")
e = s[s.hrel == "eq"].merge(p, on=["s1", "sx"], how="inner"); e["s"] = e.word.map(g.sib3); e["nw"] = e.word.map(g.n)
print(f"add1/eq argmax on #9ab {len(e):,} acc {(e.p >= .985).mean():.3f}; fillers {int(e.word.isin(FILL).sum()):,}; support<20 {int((e.nw < 20).sum()):,}")
e = e[~e.word.isin(FILL) & (e.nw >= 20)]; f = e[e.p < .985]
for cut in (0.1, 0.2, 0.3, 0.4, 0.6, 1.01):
    k = f.s <= cut
    print(f"non-filler s<={cut}: flips {int(k.sum()):,} on {f.s1[k].nunique():,} S1, p_med {f.p[k].median():.3f}")
