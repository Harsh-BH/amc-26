"""N69c: address-less SX whose core is shared by >=2 S1: pick the S1 whose full normalised name (legal form kept) equals the
SX's, when exactly one does. Train truth (India/US)."""
import sys, duckdb, numpy as np, pandas as pd
sys.path[:0] = ["probes/P64-legal-form-feat", "code/business_entity_resolution/src"]
import normalize as N
con = duckdb.connect(); con.execute("SET memory_limit='3500MB'")
for c in ("India", "US"):
    ex = pd.read_parquet(f"data/work/n69/ex_train_{c}.parquet")
    ex = ex[ex.n_core >= 2]
    xs = con.execute(f"""select entity_id sx, business_name nm from 'data/work/train_source2.parquet' where country='{c}' union all
                        select entity_id, business_name from 'data/work/train_source3.parquet' where country='{c}'""").df().set_index("sx").nm
    ex["full_x"] = N.name_norm(pd.Series(xs.reindex(ex.sx).fillna("").to_numpy())).to_numpy()
    s1 = con.execute(f"select r.id s1, r.core, t.business_name nm from 'data/work/rf_train_s1_{c}.parquet' r join 'data/work/train_source1.parquet' t on t.entity_id=r.id").df()
    s1 = s1[s1.core.isin(set(ex.core))]
    s1["full"] = N.name_norm(s1.nm).to_numpy()
    g = s1.groupby(["core", "full"]).s1.agg(["first", "size"])
    k = g.reindex(pd.MultiIndex.from_arrays([ex.core, ex.full_x]))
    ex["n_full"] = k["size"].fillna(0).astype(int).to_numpy(); ex["full_s1"] = np.where(ex.n_full == 1, k["first"].fillna("").to_numpy(), "")
    u = ex[ex.n_full == 1]
    print(f"{c}: shared-core addr-less SX {len(ex):,} | unique full-name S1 {len(u):,} ({len(u)/len(ex):.3f}) precision {(u.full_s1 == u.true_s1).mean():.3f}"
          f" | by n_core { {n: round(float((z.full_s1 == z.true_s1).mean()), 3) for n, z in u.groupby(u.n_core.clip(2, 5))} }")
