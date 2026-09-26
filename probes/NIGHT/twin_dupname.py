# Verify P52 hypothesis A on train truth: for address-less SX x of a twin S1 (core shared by >=2 S1), does the true S1's
# cluster contain an ADDRESSED SX with x's exact raw name less often than a false twin's cluster?
import duckdb, sys
con = duckdb.connect(); con.execute("SET memory_limit='2000MB'; SET threads=4; SET preserve_insertion_order=false")
c = sys.argv[1]
con.execute(f"""
CREATE TABLE sx AS SELECT entity_id id, business_name n, trim(business_address) IN ('','NULL','<NULL>') al,
   substr(entity_id,1,2) src FROM (SELECT * FROM 'data/work/train_source2.parquet' UNION ALL SELECT * FROM 'data/work/train_source3.parquet') WHERE country='{c}';
CREATE TABLE s1 AS SELECT id, core FROM 'data/work/rf_train_s1_{c}.parquet';
CREATE TABLE tw AS SELECT core FROM s1 GROUP BY 1 HAVING count(*) BETWEEN 2 AND 5;
CREATE TABLE g AS SELECT s1, sx FROM 'data/work/gt_pairs.parquet' WHERE s1 IN (SELECT id FROM s1);
-- address-less true SX of twin S1s
CREATE TABLE x AS SELECT g.sx, g.s1 ts1, s1.core, sx.n, sx.src FROM g JOIN sx ON sx.id=g.sx AND sx.al JOIN s1 ON s1.id=g.s1 WHERE s1.core IN (SELECT core FROM tw);
-- each (x, twin S)
CREATE TABLE xt AS SELECT x.*, s.id s, s.id = x.ts1 is_true FROM x JOIN s1 s ON s.core = x.core;
-- addressed members of S with the same raw name as x (excluding x)
CREATE TABLE h AS SELECT xt.sx, xt.s, max((m.n = xt.n)::int) h_raw, max((lower(m.n)=lower(xt.n))::int) h_low,
    max((m.n = xt.n AND m.src = xt.src)::int) h_raw_src, count(m.id) n_mem
  FROM xt JOIN g ON g.s1 = xt.s JOIN sx m ON m.id = g.sx AND NOT m.al AND m.id <> xt.sx GROUP BY 1,2;
""")
print(c, con.execute("""SELECT is_true, count(*) n, avg(coalesce(h_raw,0)) h_raw, avg(coalesce(h_low,0)) h_low,
   avg(coalesce(h_raw_src,0)) h_raw_samesrc, avg(coalesce(n_mem,0)) mean_members
   FROM xt LEFT JOIN h USING(sx, s) GROUP BY 1""").df())
# decisive: among x whose twins split on h_raw (some twins h=1, some h=0): P(true twin has h=0)
print(con.execute("""WITH a AS (SELECT xt.sx, xt.is_true, coalesce(h.h_raw,0) hr FROM xt LEFT JOIN h USING(sx,s)),
  b AS (SELECT sx, sum(hr) nh, count(*) k, max(CASE WHEN is_true THEN hr END) true_h FROM a GROUP BY 1)
  SELECT k, count(*) n_x, avg((nh>0 AND nh<k)::int) split_share,
         avg(CASE WHEN nh>0 AND nh<k THEN (true_h=0)::int END) true_in_h0_group,
         avg(CASE WHEN nh>0 AND nh<k THEN 1.0*(k-nh)/k END) chance_h0
  FROM b GROUP BY 1 ORDER BY 1""").df())
