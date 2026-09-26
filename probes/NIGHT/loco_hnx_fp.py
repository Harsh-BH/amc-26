# US>India LOCO: what are simulated hn-extra's FALSE adds? name relation x gained-word lens band.
import sys; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor"), str(R / "probes/P20-france-patterns")]
import p46, r05 as RR, features as F
from patterns import name_rel
from loco import CI, OFF1, OFF2, TS
W = R / "data/work"; SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
src, tgt = sys.argv[1], sys.argv[2]; ci, t = CI[tgt], TS[(src, tgt)][0]
S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"]); S = S[S.country.to_numpy() == ci]
id1, idx = F.ids("train", tgt)
d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False), "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                  "p": S.p.to_numpy(), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()}); del S
d = d[d.am].reset_index(drop=True)
base = d.p >= t
r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"]); rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core", "is_web"])
h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
both = (h1 != "") & (hx != ""); dl = np.full(len(d), np.nan)
dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3))
ne = set(d.s1[base])
hnx = ~base & (h1 != "") & (h1 == hx) & ~sib & (d.p >= 0.8) & d.s1.isin(ne)
x = d[hnx].copy()
c1, cx = r1.core.reindex(x.s1).fillna("").to_numpy(), rx.core.reindex(x.sx).fillna("").to_numpy(); wx = rx.is_web.reindex(x.sx).fillna(False).to_numpy()
nr = [name_rel(a, b, w) for a, b, w in zip(c1, cx, wx)]
x["rel"] = [k for k, _ in nr]
x["gain"] = [([w for w in b.split() if w not in set(a.split())] or [""])[0] for a, b in zip(c1, cx)]
lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
x["s"] = lt.s.reindex(x.gain).to_numpy(); x["n"] = lt.n.reindex(x.gain).to_numpy()
x["band"] = np.select([x.gain == "", x.n.fillna(0) < 20, x.s <= 0.05, x.s <= 0.25], ["none", "rare(n<20)", "s<=.05", ".05-.25"], ">.25")
print(f"{src}>{tgt} hn-extra adds {len(x)} prec {x.y.mean():.3f}")
print(x.groupby(["rel", "band"]).agg(n=("y", "size"), true=("y", "mean")).query("n >= 30").sort_values("n", ascending=False).round(3).to_string())
