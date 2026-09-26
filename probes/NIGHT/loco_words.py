# Which words drive N54 in each LOCO target, and their target-lens s / e — compare with France's decoy-word profile.
import sys; from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa
R = Path(".").resolve()
sys.path[:0] = [str(R / "probes/P46-noise-word-tier"), str(R / "probes/P28-addr-anchor")]
import p46, r05 as RR, features as F
from loco import CI, OFF1, OFF2, TS
W = R / "data/work"
for src, tgt in (("US", "India"), ("India", "US")):
    ci, t = CI[tgt], TS[(src, tgt)][0]
    S = pd.read_parquet(W / f"p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"]); S = S[S.country.to_numpy() == ci]
    id1, idx = F.ids("train", tgt)
    d = pd.DataFrame({"s1": id1.take(pa.array(S.s1.to_numpy() - ci * OFF1)).to_numpy(zero_copy_only=False), "sx": idx.take(pa.array(S.sx.to_numpy() - ci * OFF2)).to_numpy(zero_copy_only=False),
                      "p": S.p.to_numpy(), "y": S.is_true.to_numpy(), "am": S.am_all.to_numpy()})
    d = d[d.am].reset_index(drop=True)
    lt = p46.lens_table(d[["s1", "sx"]], "train", tgt)
    D = lt[(lt.n >= 20) & (lt.s > 0.2)]
    r1 = RR._rf(W / f"rf_train_s1_{tgt}.parquet", pd.unique(d.s1), ["hn", "core"]); rx = RR._rf(W / f"rf_train_sx_{tgt}.parquet", pd.unique(d.sx), ["hn", "core"])
    h1, hx = r1.hn.reindex(d.s1).fillna("").to_numpy(), rx.hn.reindex(d.sx).fillna("").to_numpy()
    c1, cx = r1.core.reindex(d.s1).fillna("").to_numpy(), rx.core.reindex(d.sx).fillna("").to_numpy()
    g = [[w for w in b.split() if w not in set(a.split())] for a, b in zip(c1, cx)]; dr = [[w for w in a.split() if w not in set(b.split())] for a, b in zip(c1, cx)]
    d["w"] = [x[0] if len(x) == 1 and len(z) == 1 else "" for x, z in zip(g, dr)]
    c = d[(h1 != "") & (h1 == hx) & d.w.isin(D.index)]
    agg = c.groupby("w").agg(n_cell=("y", "size"), true=("y", "mean"), acc=("p", lambda s: (s >= t).mean())).join(D[["n", "s", "e"]])
    agg["n_acc"] = (agg.n_cell * agg.acc).round()
    print(f"== {src}>{tgt}: D words {len(D)}, cell {len(c)} pairs, true {c.y.mean():.3f}; weighted s {np.average(agg.s, weights=agg.n_cell):.3f} e {np.average(agg.e, weights=agg.n_cell):.3f}")
    print(agg.sort_values("n_cell", ascending=False).head(12).round(3).to_string())
