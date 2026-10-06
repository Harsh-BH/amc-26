"""P71 A on test France: token ratios, lens per kind:word, add pool on sub66_final -> work/{ratio,lens,pool}_France.*

    .venv/bin/python probes/P71-fr-noiseword/fr71.py [France]
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "probes/P71-fr-noiseword")
import p71 as P  # noqa: E402

t0 = time.time()
c = sys.argv[1] if len(sys.argv) > 1 else "France"
s1, sx = P.load(c)
print(f"{c}: {len(s1):,} S1, {len(sx):,} SX loaded {time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)
r = P.ratios(s1, sx)
r.sort_values("dfx", ascending=False).to_csv(P.WORK / f"ratio_{c}.csv")
pred = P.base_lines(P.BASE, set(s1.index))
owned = {x for v in pred.values() for x in v}
acc = pd.DataFrame([(a, b) for a, v in pred.items() for b in v], columns=["s1", "sx"])
d = P.pair_frame(P.cand_pairs(P.BASE, set(s1.index)), s1, sx)
print(f"{c}: {len(d):,} candidate pairs, {len(owned):,} owned SX {time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)
key = pd.MultiIndex.from_frame(d[["s1", "sx"]])
d["acc"] = key.isin(pd.MultiIndex.from_frame(acc))
a = d[d.acc & d.D.notna()]
r0 = float((a.D == 0).sum() / max((a.D < 0).sum(), 1))
print(f"{c}: accepted lines with hn both sides {len(a):,}: eq {float((a.D == 0).mean()):.4f} neg {float((a.D < 0).mean()):.4f} "
      f"sib3 {float(a.D.isin(P.SIB3).mean()):.4f} -> r0 {r0:.2f}; accepted pairs in cand {int(d.acc.sum()):,} of {len(acc):,}", flush=True)
L = P.lens(d, r0)
L["word"] = L.index.str.split(":").str[1]
L = L.join(r[["df1", "dfx", "ratio"]], on="word")
acc_share = d[d.kind.isin(["ins", "sub"]) & ~d.ins.str.contains(" ") & d.place & (d.D == 0)]
acc_share = acc_share.assign(k=acc_share.kind + ":" + acc_share.ins).groupby("k").acc.mean()
L["acc_eq"] = acc_share.reindex(L.index)
L.sort_values("n", ascending=False).to_csv(P.WORK / f"lens_{c}.csv")
pool = P.pool(d, s1, sx, owned, c, "test")
pool.to_parquet(P.WORK / f"pool_{c}.parquet", index=False)
kinds = d[d.D == 0].kind.value_counts().to_dict()
print(f"{c}: hn-equal candidate kinds {kinds}", flush=True)
print(f"{c}: pool {len(pool):,} (street_eq {float(pool.street_eq.mean()):.3f}, lf4 {int((pool.lf == 4).sum())}) "
      f"{time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)
top = pool.assign(k=pool.kind + ":" + pool.ins).k.value_counts().head(40)
print(top.to_string(), flush=True)
json.dump({"r0": r0, "n_pairs": len(d), "pool": len(pool), "hn_eq_kinds": kinds}, open(P.WORK / f"fr71_{c}.json", "w"))
