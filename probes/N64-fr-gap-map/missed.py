import pandas as pd, duckdb
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
con=duckdb.connect(); con.execute("SET memory_limit='3000MB'")
for c in ("US","India"):
    m=pd.read_parquet(f"data/work/p23/missed_{c}.parquet"); print(c, len(m), "sx_in_pool", m.sx_in_pool.mean().round(3))
    con.register("m",m)
    x=con.execute("""select a.nm s1n, a.ad s1a, coalesce(b.business_name,c.business_name) sxn, coalesce(b.business_address,c.business_address) sxa, m.sx_in_pool
      from m join (select entity_id, business_name nm, business_address ad from 'data/work/train_source1.parquet') a on a.entity_id=m.s1
      left join 'data/work/train_source2.parquet' b on b.entity_id=m.sx left join 'data/work/train_source3.parquet' c on c.entity_id=m.sx using sample 25 rows (reservoir, 7)""").df()
    print(x.to_string())
