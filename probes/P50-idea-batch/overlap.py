"""P50 integrator: pair-level diff of two submission folders + conflict checks against earlier pieces.
    overlap.py NEW OLD [VETO_FROM VETO_TO]   pairs NEW adds / removes vs OLD; of the adds, how many a previous piece
    removed (VETO_FROM − VETO_TO, e.g. sub44r − sub48 = P48's veto) or the India/US combiner scores below t."""
import sys
import numpy as np, pandas as pd


def pairs(folder):
    s = set()
    with open(f"output/{folder}/matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, v = line.rstrip("\n").split("\t", 1)
            if v:
                s.update((a, x) for x in v.split(","))
    return s


new, old = pairs(sys.argv[1]), pairs(sys.argv[2])
add, rm = new - old, old - new
print(f"{sys.argv[1]} vs {sys.argv[2]}: +{len(add):,} / -{len(rm):,} pairs", flush=True)
del new
if len(sys.argv) > 4:
    vf, vt = pairs(sys.argv[3]), pairs(sys.argv[4])
    veto = vf - vt
    print(f"  adds that {sys.argv[3]}->{sys.argv[4]} removed: {len(add & veto)} of {len(veto):,}", flush=True)
    del vf, vt
s1 = pd.read_parquet("data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
a = pd.DataFrame(sorted(add), columns=["s1", "sx"])
a["country"] = s1.reindex(a.s1).to_numpy()
print("  adds by country", a.country.value_counts().to_dict(), flush=True)
for c in ("India", "US"):
    import glob
    f = sorted(glob.glob(f"data/work/p50/prob_test_{c}_fullzoo.parquet"))
    if not f:
        continue
    d = pd.read_parquet(f[0], columns=["s1", "sx", "p", "am"])
    m = a[a.country == c].merge(d, on=["s1", "sx"], how="left")
    print(f"  {c} adds: {len(m)}; in the combiner's pair file {int(m.p.notna().sum())}; of those am & p < .71 (combiner said no): "
          f"{int((m.am.fillna(False) & (m.p < 0.71)).sum())}; not argmax {int((~m.am.fillna(True)).sum())}", flush=True)
