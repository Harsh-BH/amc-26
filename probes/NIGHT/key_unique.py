# Train truth: SX whose (core, city, hn) key matches exactly ONE S1 of its country — how often is that S1 its true S1?
import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='2000MB'; SET threads=4")
for c in ["US", "India"]:
    print(c, con.execute(f"""
    WITH s AS (SELECT id, core, city, hn FROM 'data/work/rf_train_s1_{c}.parquet' WHERE hn<>'' AND city<>''),
    k AS (SELECT core, city, hn, min(id) s1 FROM s GROUP BY 1,2,3 HAVING count(*)=1),
    x AS (SELECT id, core, city, hn FROM 'data/work/rf_train_sx_{c}.parquet' WHERE hn<>'' AND city<>''),
    j AS (SELECT x.id sx, k.s1 FROM x JOIN k USING(core, city, hn)),
    g AS (SELECT sx, s1 ts1 FROM 'data/work/gt_pairs.parquet')
    SELECT count(*) n, avg((g.ts1 = j.s1)::int) prec, avg((g.ts1 IS NULL)::int) decoy FROM j LEFT JOIN g USING(sx)""").df().T)
