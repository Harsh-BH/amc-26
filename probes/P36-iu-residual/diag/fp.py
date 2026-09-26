import sys; sys.path.insert(0, "probes/P36-iu-residual")
import numpy as np, pandas as pd
from base import A, LB_W, OUT, W, state
rng = np.random.default_rng(1); lines = []
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
GT = pq.read_table(W / "gt_pairs.parquet")
for c in LB_W:
    s = state(c); e = pd.read_parquet(OUT / f"err_{c}.parquet"); e = e[e.type.str.startswith("fp")].reset_index(drop=True)
    gt = GT.filter(pc.is_in(GT["sx"], value_set=pa.array(e.sx.unique(), pa.large_string()))).to_pandas()
    own = gt.set_index("sx").s1
    e["own"] = own.reindex(e.sx).to_numpy()
    ids1 = pd.unique(pd.concat([e.s1, e.own.dropna()]))
    s1t, sxt = A.texts("train", c, ids1, pd.unique(e.sx))
    i1 = {k: i for i, k in enumerate(ids1)}; ix = {k: i for i, k in enumerate(pd.unique(e.sx))}
    S = s1t.iloc[e.s1.map(i1)].reset_index(drop=True); X = sxt.iloc[e.sx.map(ix)].reset_index(drop=True)
    e["kat"] = np.asarray(A.CATS, object)[A.cat_frame(S.name, X.name, X.addr, X.is_web, S.hn, X.hn)]
    ho_ids = set(s["ids"])
    e["own_kind"] = np.where(e.own.isna(), "distractor", np.where(e.own.isin(ho_ids), "holdout_s1", "train_s1"))
    print(c, len(e), "own:", e.own_kind.value_counts().to_dict(), "| cat:", e.kat.value_counts().to_dict())
    print("  p' bands", pd.cut(e.pa, [.71, .8, .9, .97, .99, 1]).value_counts(sort=False).to_dict())
    # does the own S1 exist among the SX candidates, and with what p?
    full = pq.read_table(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "p"], filters=[("sx", "in", list(e.sx.unique()))]).to_pandas()
    k = full.merge(e[["sx", "own"]].dropna(), left_on=["sx", "s1"], right_on=["sx", "own"])
    print("  own S1 is a candidate of the SX:", len(k), "of", int(e.own.notna().sum()), "own p median", round(k.p.median(), 3))
    for i in rng.choice(len(e), 15, replace=False):
        r = e.iloc[i]
        o = f"\n     OWN({r.own_kind}): {s1t.iloc[i1[r.own]]['name']} | {s1t.iloc[i1[r.own]].addr}" if isinstance(r.own, str) else "\n     OWN: none (distractor)"
        lines.append(f"[{c} {r.type} {r.kat}] pa={r.pa:.3f} p8c={r.p:.3f} m2={r.p_m2:.3f}\n     S1: {S.name[i]} | {S.addr[i]}\n     SX: {X.name[i]} | {X.addr[i]}{o}")
open(OUT / "examples_fp.txt", "w").write("\n".join(lines) + "\n")
