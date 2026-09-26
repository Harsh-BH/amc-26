"""France (label-free): name-relation mix (namerel.rel) of accepted pairs by p band, of gc70's adds, and of two-sided
hn-equal extras. Train prior: 2.5-3.6% of true pairs are rel 0 (unrelated name).
    .venv/bin/python probes/P37-fr-complete/fr_namerel.py
"""
import json, sys
import numpy as np
import pandas as pd
sys.path[:0] = ["probes/P37-fr-complete"]
import p37 as P
from namerel import rel_arr

d, a = P.fr_load()
c1 = pd.read_parquet(P.W / "rf_test_s1_France.parquet", columns=["id", "core", "city"]).set_index("id")
cx = pd.read_parquet(P.W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core", "city"]).set_index("id")
gc = P.gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["pred"], a["sib"], 0.7, "oneall")
b = a["pred"] | gc
ex = P.gc_add(a["s1"], a["sx"], a["is2"], a["p"], a["am"], b, a["sib"], 0.9, "allhn", a["hc"] == 0)
m = b | ex
o = np.flatnonzero(m)
r = np.full(len(d), -1, np.int8)
r[o] = rel_arr(c1.core.reindex(d.s1.to_numpy()[o]).fillna("").to_numpy(), cx.core.reindex(d.sx.to_numpy()[o]).fillna("").to_numpy())
out = {}
acc0 = a["pred"]
for lo, hi in ((0.9868, 0.995), (0.995, 0.999), (0.999, 1.01)):
    k = acc0 & (a["p"] >= lo) & (a["p"] < hi)
    out[f"accepted[{lo},{hi})"] = {"n": int(k.sum()), **{f"rel{j}": round((r[k] == j).mean(), 4) for j in range(5)}}
for nm, k in (("accepted<0.9868 (rescue/rules)", acc0 & (a["p"] < 0.9868)), ("gc70_adds", gc), ("gc70_adds_hn_eq", gc & (a["hc"] == 0)),
              ("gc70_adds_hn_not_eq", gc & (a["hc"] != 0)), ("extras_hn_eq_0.9", ex)):
    out[nm] = {"n": int(k.sum()), **{f"rel{j}": round((r[k] == j).mean(), 4) for j in range(5)}}
for k, v in out.items():
    print(f"{k:34s} {v}", flush=True)
json.dump(out, open("probes/P37-fr-complete/fr_namerel.json", "w"), indent=1)
