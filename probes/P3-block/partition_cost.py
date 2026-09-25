"""Share of S1 x SX slots the state partition still searches: sum_g p1(g) * px(g) + px(no state).
Train: from full.py prep texts. Test: P1 state on a 60k-row sample per source and country (unsupervised)."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np, pandas as pd, pyarrow.parquet as pq, block as B
from full import TXT, WORK


def frac(st1: pd.Series, stx: pd.Series) -> float:
    p1, px = st1.value_counts(normalize=True), stx.value_counts(normalize=True)
    no = px.get("", 0.0)
    return float(sum(p1.get(g, 0) * px[g] for g in px.index if g) + no), float(no)


def sample(split, src, country, n=60_000):
    t = pq.read_table(WORK / f"{split}_{src}.parquet", filters=[("country", "==", country)])
    d = t.slice(0, n).to_pandas()
    return B.texts(d, {**B.CFG, "translit": "none"}).st


rows = []
for c in ("India", "US"):
    fs = sorted(TXT.glob(f"{c}_sx_*.parquet"))
    if fs:
        st1 = pd.read_parquet(TXT / f"{c}_s1.parquet", columns=["st"]).st
        stx = pd.concat([pd.read_parquet(p, columns=["st"]).st for p in fs])
        rows.append(("train-full", c, *frac(st1, stx)))
for c in ("France", "India", "US"):
    st1 = sample("test", "source1", c)
    stx = pd.concat([sample("test", s, c) for s in ("source2", "source3")])
    rows.append(("test-sample", c, *frac(st1, stx)))
print(pd.DataFrame(rows, columns=["split", "country", "slot_frac", "sx_no_state"]).round(4).to_string())
