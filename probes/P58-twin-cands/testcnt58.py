"""P58: label-free test counts -- address-less SX (own key) in same-key S1 groups of size 2..30 and the candidate pairs the
expansion would add (upper bound: every group S1, before removing pairs already in candidate_pairs).
    probes/P58-twin-cands/py.sh probes/P58-twin-cands/testcnt58.py"""
import pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from frame58 import W, key, NA, GMAX

na = pa.array(sorted(NA), pa.string())
for c in ("India", "US", "France"):
    s1 = pq.read_table(W / "test_source1.parquet", columns=["business_name", "country"], filters=[("country", "=", c)]).to_pandas()
    g = pd.Series([key(x) for x in s1.business_name]).value_counts()
    n = 0; rows = 0; sx_all = 0
    for i in (2, 3):
        for b in pq.ParquetFile(W / f"test_source{i}.parquet").iter_batches(columns=["business_name", "business_address", "country"], batch_size=1_000_000):
            a = pc.utf8_upper(pc.utf8_trim_whitespace(b.column(1)))
            m = pc.and_(pc.equal(b.column(2), c), pc.is_in(a, value_set=na))
            names = b.column(0).filter(m).to_pylist()
            sx_all += len(names)
            k = g.reindex([key(x) for x in names]).fillna(0)
            k = k[(k >= 2) & (k <= GMAX)]
            n += len(k); rows += int(k.sum())
    print(f"{c}: S1 {len(s1):,}; address-less SX {sx_all:,}; in same-key groups 2..{GMAX}: {n:,} SX -> {rows:,} candidate rows "
          f"(+{rows / len(s1):.3f} per S1 upper bound)", flush=True)
