"""P71 A build: sub66_final + France pool pairs admitted by the learned word list (label-free, from work/lens_France.csv).

  swap words  sub:w with SX/S1 ratio >= 3, n >= 20, s <= 0.10 (tier a: all but groupe; tier b: + groupe)
  insert words ins:w with ratio >= 3, n >= 20, s <= 0.10 (fillers), and, for tier i only, branch-profile inserts whose
              baseline-excess estimate >= 0.8 (développement .805, groupe .874)

    .venv/bin/python probes/P71-fr-noiseword/build71.py a|b|bi OUT
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "probes/P71-fr-noiseword")
import p71 as P  # noqa: E402

t0 = time.time()
tier, out = sys.argv[1], Path(sys.argv[2])
L = pd.read_csv(P.WORK / "lens_France.csv", index_col=0)
L = L[(L.n >= P.SUPPORT) & (L.ratio >= P.RATIO_MIN)]
ins = L[L.index.str.startswith("ins:") & (L.n >= 100) & (L.s >= 0.3)]
b = float(np.median(ins.n_eq / ins.n_sib.clip(lower=1)))
L["est"] = 1 - b * L.n_sib / L.n_eq.clip(lower=1)
words = set(L.index[L.s <= 0.10])
if tier == "a":
    words -= {"sub:groupe"}
if tier == "bi":
    words |= set(L.index[L.index.str.startswith("ins:") & (L.s >= 0.3) & (L.est >= 0.8)])
pool = pd.read_parquet(P.WORK / "pool_France.parquet")
ok = P.admit(pool, words)
add = pool[ok].copy()
add["k"] = add.kind + ":" + add.ins
rank = {k: i for i, k in enumerate(["sub:developpement", "sub:groupe", "ins:developpement", "ins:groupe"])}
add = add.assign(o=add.k.map(rank).fillna(-1)).sort_values(["o", "s1", "sx"]).reset_index(drop=True)
s1 = pd.read_parquet(P.W / "test_source1.parquet", columns=["entity_id", "country"])
fr = set(s1.entity_id[s1.country == "France"])
del s1
pred = P.base_lines(P.BASE, fr)
new, rows = P.merge(pred, add)
chg = set(rows.s1)
P.write(P.BASE, out, new, chg)
kk = rows.merge(add[["s1", "sx", "k"]], on=["s1", "sx"]).k
shape = lambda Q: {"empty": round(sum(1 for v in Q.values() if not v) / len(fr), 5), "mps": round(sum(len(v) for v in Q.values()) / len(fr), 4)}
info = {"tier": tier, "baseline_b": round(b, 4), "words": sorted(words), "pool_admitted": int(ok.sum()), "added": len(rows),
        "s1_changed": len(chg), "empties_filled": sum(1 for s in chg if not pred.get(s)), "lf4_vetoed": int((P.admit(pool, words) != P.admit(pool.assign(lf=0), words)).sum()),
        "by_word": {str(a): int(v) for a, v in kk.value_counts().items()}, "shape_before": shape(pred), "shape_after": shape(new),
        "runtime_s": round(time.time() - t0), "peak_gb": P.peak()}
(out / "p71_info.json").write_text(json.dumps(info, indent=1))
rows.to_parquet(P.WORK / f"adds_{tier}.parquet", index=False)
print(json.dumps(info), flush=True)
