"""France suffix-construct rule sizing (label-free): argmax pairs whose cores differ ONLY by French suffix words,
at an equal house number, with no other candidate S1 sharing the suffix-stripped core. Counts by p band."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

W = Path("/home/harsh1/github-repos/amc-26/data/work")
prob = sys.argv[1] if len(sys.argv) > 1 else str(W / "p14/prob_test_France_idf144.parquet")
t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.975
FR = {"fils", "cie", "compagnie", "associes", "freres", "frs"}
a = pd.read_parquet(prob, columns=["s1", "sx", "p"])
r1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
rx = pd.read_parquet(W / "rf_test_sx_France.parquet", columns=["id", "core", "hn", "city"]).set_index("id")
strip = lambda c: " ".join(w for w in (c or "").split() if w not in FR)
k1 = r1.core.fillna("").map(strip)
a["k1"] = k1.reindex(a.s1).to_numpy()
a["kx"] = rx.core.fillna("").map(strip).reindex(a.sx).to_numpy()
a["c1"], a["cx"] = r1.core.reindex(a.s1).fillna("").to_numpy(), rx.core.reindex(a.sx).fillna("").to_numpy()
a["h1"], a["hx"] = r1.hn.reindex(a.s1).fillna("").to_numpy(), rx.hn.reindex(a.sx).fillna("").to_numpy()
a["y1"], a["yx"] = r1.city.reindex(a.s1).fillna("").to_numpy(), rx.city.reindex(a.sx).fillna("").to_numpy()
a["kmatch"] = (a.k1 == a.kx) & (a.k1 != "")
nk = a[a.kmatch].groupby("sx").size()                       # candidates of this SX sharing the stripped core
am = a.sort_values("p", ascending=False).drop_duplicates("sx")
sfx = am.kmatch & (am.c1 != am.cx)
hn = (am.h1 != "") & (am.h1 == am.hx)
city = (am.y1 != "") & (am.y1 == am.yx)
uniq = am.sx.map(nk).fillna(0).to_numpy() == 1
print(f"argmax pairs {len(am):,}; cores differ only by FR suffix words: {int(sfx.sum()):,}")
for lab, m in (("+hn eq", sfx & hn), ("+hn eq +unique", sfx & hn & uniq), ("+hn eq +city eq +unique", sfx & hn & city & uniq),
               ("hn missing +city eq +unique", sfx & (am.hx == "") & city & uniq)):
    g = am[m]
    print(f" {lab:28s} n {len(g):6,} accepted@{t} {(g.p >= t).mean():.3f}  p bands <.3 {(g.p < .3).mean():.3f} "
          f".3-.7 {((g.p >= .3) & (g.p < .7)).mean():.3f} .7-{t} {((g.p >= .7) & (g.p < t)).mean():.3f}; would flip {int((g.p < t).sum()):,} "
          f"(S1 touched {g[g.p < t].s1.nunique():,})")
# contrast: same test with no suffix difference (plain core equal) = the model's usual acceptance of such pairs
ce = (am.c1 == am.cx) & (am.c1 != "") & hn & city & uniq
print(f" contrast core-equal +hn +city +unique: n {int(ce.sum()):,} accepted@{t} {(am.p[ce] >= t).mean():.3f}")
