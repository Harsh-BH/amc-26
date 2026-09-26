# Train truth: an address-less SX whose core equals the core of exactly k S1 (same country) -- how often is one of them its true S1?
import duckdb, sys
con = duckdb.connect(); con.execute("SET memory_limit='2500MB'; SET threads=4")
for c in ["US", "India"]:
    r = con.execute(f"""
    WITH al AS (SELECT entity_id id FROM (SELECT * FROM 'data/work/train_source2.parquet' UNION ALL SELECT * FROM 'data/work/train_source3.parquet')
                WHERE country='{c}' AND trim(business_address) IN ('','NULL','<NULL>')),
    x AS (SELECT r.id, r.core FROM 'data/work/rf_train_sx_{c}.parquet' r JOIN al USING(id)),
    s AS (SELECT id, core FROM 'data/work/rf_train_s1_{c}.parquet'),
    k AS (SELECT core, count(*) k FROM s GROUP BY 1),
    t AS (SELECT sx, s1 FROM 'data/work/gt_pairs.parquet'),
    j AS (SELECT x.id, x.core, coalesce(k.k,0) k, t.s1, ts.core tcore FROM x LEFT JOIN k USING(core) LEFT JOIN t ON t.sx=x.id LEFT JOIN s ts ON ts.id=t.s1)
    SELECT CASE WHEN k=0 THEN '0' WHEN k=1 THEN '1' WHEN k=2 THEN '2' WHEN k<=5 THEN '3-5' ELSE '6+' END kb, count(*) n,
           avg((s1 IS NOT NULL)::int) has_true, avg((tcore=core)::int) true_core_eq
    FROM j GROUP BY 1 ORDER BY 1""").df()
    print(c); print(r)
