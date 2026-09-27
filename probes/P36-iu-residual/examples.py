"""P36 step 1b: pattern read of the biggest buckets (lowp, stolen, block_addr) on the #34 holdout.
Category mix (a10.cat_frame), p bands, thief anatomy, and 30 raw examples per bucket (15 India + 15 US) written to
data/work/p36/examples.txt (gitignored: raw rows never go into RESULT.md).
    PYTHONPATH=probes/P36-iu-residual probes/P36-iu-residual/py.sh probes/P36-iu-residual/examples.py
"""
import numpy as np
import pandas as pd

from base import A, LB_W, OUT, W, state

rng = np.random.default_rng(0)
lines = []
pd.set_option("display.width", 220)
for c in LB_W:
    s = state(c)
    e = pd.read_parquet(OUT / f"err_{c}.parquet")
    full = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "p", "am", "y"])
    thief = full[full.am].set_index("sx")[["s1", "p", "holdout"]]
    ncand = full.groupby("sx").size()
    e = e[e.type.isin(["lowp", "stolen", "block_addr"])].reset_index(drop=True)
    e["t_s1"] = thief.s1.reindex(e.sx).to_numpy()
    e["t_p"] = thief.p.reindex(e.sx).to_numpy()
    e["t_ho"] = thief.holdout.reindex(e.sx).to_numpy()
    e["n_cand_sx"] = ncand.reindex(e.sx).fillna(0).to_numpy()
    s1t, sxt = A.texts("train", c, pd.unique(pd.concat([e.s1, e.t_s1.dropna()])), pd.unique(e.sx))
    s1i = {k: i for i, k in enumerate(pd.unique(pd.concat([e.s1, e.t_s1.dropna()])))}
    sxi = {k: i for i, k in enumerate(pd.unique(e.sx))}
    S = s1t.iloc[e.s1.map(s1i)].reset_index(drop=True)
    X = sxt.iloc[e.sx.map(sxi)].reset_index(drop=True)
    e["kat"] = np.asarray(A.CATS, object)[A.cat_frame(S.name, X.name, X.addr, X.is_web, S.hn, X.hn)]
    for t in ("lowp", "stolen", "block_addr"):
        m = (e.type == t).to_numpy()
        print(f"\n== {c} {t}: {m.sum()} pairs; category mix", e.kat[m].value_counts(normalize=True).round(3).to_dict())
        if t == "lowp":
            print("   p' bands", pd.cut(e.pa[m], [0, .1, .3, .5, .6, .71]).value_counts(sort=False).to_dict())
            print("   p8c bands", pd.cut(e.p[m], [0, .1, .3, .5, .6, .71, 1]).value_counts(sort=False).to_dict())
        if t == "stolen":
            tp = e.t_p[m]
            print(f"   thief is a holdout S1 {e.t_ho[m].mean():.3f}; thief p>=t {(tp >= A.T).mean():.3f}; "
                  f"thief p bands {pd.cut(tp, [0, .1, .3, .5, .71, .9, 1]).value_counts(sort=False).to_dict()}")
            print("   true-pair p bands", pd.cut(e.p[m], [0, .01, .1, .3, .5, .71, 1]).value_counts(sort=False).to_dict())
        for i in rng.choice(np.flatnonzero(m), 15, replace=False):
            r = e.iloc[i]
            th = ""
            if isinstance(r.t_s1, str) and t != "lowp":
                ts = s1t.iloc[s1i[r.t_s1]]
                th = f"\n     THIEF p={r.t_p:.3f} ho={r.t_ho}: {ts['name']} | {ts.addr}"
            lines.append(f"[{c} {t} {r.kat}] p8c={r.p:.3f} pa={r.pa:.3f} m2={r.p_m2:.3f} r={r.r:.3f} ncand={r.n_cand_sx:.0f}\n"
                         f"     S1: {S.name[i]} | {S.addr[i]}\n     SX: {X.name[i]} | {X.addr[i]}{th}")
(OUT / "examples.txt").write_text("\n".join(lines) + "\n")
print("examples ->", OUT / "examples.txt")
