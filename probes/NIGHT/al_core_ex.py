import duckdb, pandas as pd
con = duckdb.connect(); con.execute("SET memory_limit='2500MB'; SET threads=4")
c = "US"
df = con.execute(f"""
WITH sxr AS (SELECT * FROM 'data/work/train_source2.parquet' WHERE country='{c}' UNION ALL SELECT * FROM 'data/work/train_source3.parquet' WHERE country='{c}'),
al AS (SELECT entity_id id FROM sxr WHERE trim(business_address) IN ('','NULL','<NULL>')),
x AS (SELECT r.id, r.core FROM 'data/work/rf_train_sx_{c}.parquet' r JOIN al USING(id)),
s AS (SELECT id, core FROM 'data/work/rf_train_s1_{c}.parquet'),
k AS (SELECT core FROM s GROUP BY 1 HAVING count(*)=1),
u AS (SELECT x.id sx, s.id s1 FROM x JOIN k USING(core) JOIN s USING(core)),
p AS (SELECT * FROM 'data/work/p23/pairs_{c}.parquet' WHERE holdout),
r AS (SELECT p.s1, p.sx, p.p FROM u JOIN p USING(s1, sx) WHERE NOT p.pred AND NOT p.y AND p.p > 0.1),
g AS (SELECT sx, s1 ts1 FROM 'data/work/gt_pairs.parquet')
SELECT r.p, a.business_name n1, a.business_address a1, xr.business_name nx, b.business_name nt, b.business_address atr
FROM r JOIN 'data/work/train_source1.parquet' a ON a.entity_id=r.s1 JOIN sxr xr ON xr.entity_id=r.sx
LEFT JOIN g ON g.sx=r.sx LEFT JOIN 'data/work/train_source1.parquet' b ON b.entity_id=g.ts1 LIMIT 25""").df()
for t in df.itertuples():
    print(f"p={t.p:.2f} SX[{t.nx}]\n   S1 [{t.n1} | {t.a1}]\n   TRUE[{t.nt} | {t.atr}]")
