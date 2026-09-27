"""P50: re-filter P62's France acronym change set for another base: drop adds whose SX is already in a France line of
BASE or whose S1 would pass the 5 S2 / 6 S3 caps (tiers taken in order T2, T1, T3).  -> OUT.parquet (s1, sx, tier)
    acro_fit.py BASE OUT.parquet"""
import sys
from pathlib import Path
import pandas as pd
sys.path[:0] = ["probes/P42-fr-joint-tune", "."]
import p42
base, out = sys.argv[1:3]
lines = p42.read_fr(Path("output") / base).to_dict()
used = {x for v in lines.values() if v for x in v.split(",")}
cnt = {k: [sum(x.startswith("S2-") for x in v.split(",")) if v else 0, sum(x.startswith("S3-") for x in v.split(",")) if v else 0]
       for k, v in lines.items()}
d = pd.read_csv("data/work/p62/acro_cs_france.tsv", sep="\t", dtype={"s1": str, "sx": str, "tier": str})
d["o"] = d.tier.map({"T2": 0, "T1": 1, "T3": 2})
keep, why = [], {"sx_used": 0, "cap": 0, "not_france_s1": 0}
for r in d.sort_values(["o", "s1", "sx"]).itertuples():
    if r.s1 not in cnt:
        why["not_france_s1"] += 1
        continue
    if r.sx in used:
        why["sx_used"] += 1
        continue
    j = 0 if r.sx.startswith("S2-") else 1
    if cnt[r.s1][j] >= (5, 6)[j]:
        why["cap"] += 1
        continue
    cnt[r.s1][j] += 1
    used.add(r.sx)
    keep.append((r.s1, r.sx, r.tier))
k = pd.DataFrame(keep, columns=["s1", "sx", "tier"])
k.to_parquet(out, index=False)
print(f"{base}: {len(d):,} P62 rows -> {len(k):,} kept {k.tier.value_counts().to_dict()}; dropped {why}; "
      f"in_cand False among kept {int((~d.set_index(['s1', 'sx']).in_cand.reindex(pd.MultiIndex.from_frame(k[['s1', 'sx']]))).sum())}", flush=True)
