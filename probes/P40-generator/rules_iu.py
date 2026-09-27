"""P40 step 3: India/US op rules on the #34 holdout (out of fold), paired per-S1 delta by half.
Rules add rejected argmax pairs (free: argmax => not accepted elsewhere) of an op class with p' >= tau:
  gc_lacks : S1 group non-empty and lacks the SX's source      noaddr_uniq : address-less SX, S1 core name unique
  gib      : synthetic-brand SX                                  union of the three
    probes/P40-generator/py.sh probes/P40-generator/rules_iu.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "P36-iu-residual")]
from base import A, state
from cat import OUT
from loss import bits

LB_W = {"India": 0.468, "US": 0.383}
t0 = time.time()
res = {}
for c in LB_W:
    s = state(c)
    n, nt, h = len(s["ids"]), s["n_true"], s["h"]
    d = pd.read_parquet(OUT / f"am_{c}.parquet")
    assert d.g.max() < n
    B = bits(d.ops.to_numpy())
    acc, y, g, pa = d.acc.to_numpy(), d.y.to_numpy(), d.g.to_numpy(), d.pa.to_numpy()
    is2 = d.sx.str.startswith("S2-").to_numpy()
    h2 = np.bincount(g[acc & is2], minlength=n) > 0
    h3 = np.bincount(g[acc & ~is2], minlength=n) > 0
    lacks = np.where(is2, ~h2[g], ~h3[g]) & (h2[g] | h3[g])
    k = pd.read_parquet(OUT / f"s1twin_{c}.parquet").set_index("id").k.reindex(d.s1).fillna(1).to_numpy()
    cls = {"gc_lacks": lacks, "noaddr_uniq": B.A_EMPTY.to_numpy() & (k == 1), "gib": B.N_GIB_NAME.to_numpy()}
    cls["union"] = cls["gc_lacks"] | cls["noaddr_uniq"] | cls["gib"]
    # base per-S1 counts from the full pair table (acc includes non-argmax? no: acc = am & p' >= T)
    pr = s["pairs"]
    tp0 = np.bincount(pr.g[pr.acc & pr.y], minlength=n)
    np0 = np.bincount(pr.g[pr.acc], minlength=n)
    f0 = A.fb(tp0, np0, nt)
    assert np.allclose(f0, s["f34"])
    for name, m in cls.items():
        for tau in (0.5, 0.6, 0.65):
            add = m & ~acc & (pa >= tau)
            f1 = A.fb(tp0 + np.bincount(g[add & y], minlength=n), np0 + np.bincount(g[add], minlength=n), nt)
            dl = f1 - f0
            r = res.setdefault(f"{name}@{tau}", {})
            for hh in (0, 1):
                x = dl[h == hh]
                r[f"{c}_h{hh}"] = (x.mean() * LB_W[c], x.std(ddof=1) / np.sqrt(len(x)) * LB_W[c])
            r[f"{c}_n"], r[f"{c}_prec"] = int(add.sum()), float(y[add].mean()) if add.any() else np.nan
for k_, r in res.items():
    tot = [r["India_h%d" % hh][0] + r["US_h%d" % hh][0] for hh in (0, 1)]
    se = [np.hypot(r["India_h%d" % hh][1], r["US_h%d" % hh][1]) for hh in (0, 1)]
    print(f"{k_:18s} LB-eq h0 {tot[0]:+.6f} ± {se[0]:.6f}  h1 {tot[1]:+.6f} ± {se[1]:.6f} | India n {r['India_n']} prec {r['India_prec']:.3f}"
          f" US n {r['US_n']} prec {r['US_prec']:.3f}")
res["runtime_s"] = time.time() - t0
import json
(Path(__file__).parent / "rules_iu.json").write_text(json.dumps(res, indent=1, default=float))
