# LOCO: simulate France-like add rules (hn-extra 0.8, gc 0.7 one-sided missing source, rescue 0.9 empty S1) and
# decompose each rule's adds by name relation x gained-word lens band (after removing N54 cells), with truth.
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
d = d[d.am].reset_index(drop=True); d["src"] = d.sx.str[:2]
r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"]); rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core", "is_web"])
h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
both = (h1 != "") & (hx != ""); dl = np.full(len(d), np.nan)
dl[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
sib = np.isin(dl, list(SIB3)) | np.isin(-dl, list(SIB3)); hn_eq = (h1 != "") & (h1 == hx)
c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]; dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1, cx)]
d["gain"] = [x[0] if len(x) == 1 else "" for x in g]
lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
D = set(lt[(lt.n >= 20) & (lt.s > 0.25)].index)
n54 = hn_eq & np.array([len(x) == 1 and len(z) <= 1 for x, z in zip(g, dr)]) & d.gain.isin(D).to_numpy()
p = d.p.to_numpy(); base = (p >= t) & ~n54
grp = d[base].groupby("s1").src.agg(set)
hnx = ~base & ~n54 & hn_eq & ~sib & (p >= 0.8) & d.s1.isin(grp.index).to_numpy()
acc = base | hnx; grp = d[acc].groupby("s1").src.agg(set)
miss = np.array([(s not in grp.get(a, set())) and len(grp.get(a, set())) == 1 for a, s in zip(d.s1, d.src)])
gc = ~acc & ~n54 & ~sib & (p >= 0.7) & miss
acc |= gc; filled = set(d.s1[acc])
resc_c = ~acc & ~n54 & ~sib & (p >= 0.9) & ~d.s1.isin(filled).to_numpy()
best = d[resc_c].sort_values("p", ascending=False).drop_duplicates("s1").index
resc = np.zeros(len(d), bool); resc[best] = True
d["rel"] = [name_rel(a, b, w)[0] for a, b, w in zip(c1, cx, wx)]
d["s"] = lt.s.reindex(d.gain).to_numpy(); d["n"] = lt.n.reindex(d.gain).to_numpy()
d["band"] = np.select([d.gain == "", d.n.fillna(0) < 20, d.s <= 0.05, d.s <= 0.25], ["none", "rare", "s<=.05", ".05-.25"], ">.25")
d["hrel"] = np.where(hn_eq, "hn_eq", np.where(both, "hn_diff", "hn_missing"))
for name, m in (("hn-extra", hnx), ("gc", gc), ("rescue", resc)):
    x = d[m]
    print(f"== {src}>{tgt} {name}: {len(x)} adds, prec {x.y.mean():.3f}")
    print(x.groupby(["rel", "hrel", "band"]).agg(n=("y", "size"), true=("y", "mean")).query("n >= 40").sort_values("n", ascending=False).head(10).round(3).to_string())
