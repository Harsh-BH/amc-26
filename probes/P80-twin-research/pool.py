"""Pool test for R1 on P57's holdout frames (M2 p, labels): R1 ~ address-less SX, name_ratio 100 to exactly one candidate S1,
SX carries a legal form equal to that S1's, 2-3 candidate S1 with the same core. Precision over all R1 rows and over the pool a
strict bar leaves (p < .9868, France's), floor p >= .3 like xfc. In-country M2, so optimistic vs LOCO."""
import pandas as pd, pyarrow.parquet as pq
for c in ("India", "US"):
    d = pq.read_table(f"data/work/p57/holdout_m2_{c}.parquet", columns=["qx", "p", "y", "addr_empty_x", "name_ratio", "core_eq",
                                                                        "legal_nx", "legal_eq", "pos"]).to_pandas()
    d["ex"] = d.name_ratio >= 0.99999; g = d.groupby("qx")
    d["n_ex"] = g.ex.transform("sum"); d["n_core"] = g.core_eq.transform("sum")
    r = d[(d.pos >= 0) & (d.addr_empty_x == 1) & d.ex & (d.n_ex == 1) & (d.legal_nx >= 1) & (d.legal_eq == 1)]
    for nm, m in (("core 1 (xfc-like)", r.n_core == 1), ("core 2-3 (R1)", r.n_core.between(2, 3)), ("core >=4", r.n_core >= 4)):
        s = r[m]; pool = s[(s.p < .9868) & (s.p >= .3)]
        print(f"{c} {nm}: all n {len(s):,} prec {s.y.mean():.3f} | pool .3<=p<.9868 n {len(pool):,} prec {pool.y.mean():.3f} | "
              f"p<.3 n {int((s.p < .3).sum())} prec {s.y[s.p < .3].mean():.3f}")
