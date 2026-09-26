# Is P40's address-less "Bayes floor" real? Check raw-name surface forms of twin S1s (train, per country).
import duckdb, sys
con = duckdb.connect(); con.execute("SET memory_limit='2GB'; SET threads=4")
c = sys.argv[1]
con.execute(f"""
CREATE TABLE s1 AS SELECT entity_id id, business_name n, business_address a FROM 'data/work/train_source1.parquet' WHERE country='{c}';
CREATE TABLE sx AS SELECT entity_id id, business_name n, business_address a FROM 'data/work/train_source2.parquet' WHERE country='{c}'
   UNION ALL SELECT entity_id, business_name, business_address FROM 'data/work/train_source3.parquet' WHERE country='{c}';
CREATE TABLE g AS SELECT s1, sx FROM 'data/work/gt_pairs.parquet' WHERE s1 IN (SELECT id FROM s1);
CREATE TABLE s1k AS SELECT id, n, a, lower(n) kl, regexp_replace(lower(n),'[^a-z0-9]','','g') kn FROM s1;
CREATE TABLE cnt AS SELECT kn, count(*) c_kn, count(DISTINCT n) d_raw, count(DISTINCT kl) d_low FROM s1k GROUP BY kn;
""")
# address-less true SX
q = """
WITH al AS (SELECT g.s1, g.sx, x.n xn FROM g JOIN sx x ON x.id=g.sx WHERE trim(x.a) IN ('','NULL','<NULL>'))
SELECT count(*) n_pairs,
  avg((c.c_kn>=2)::int) twin_share,
  avg(CASE WHEN c.c_kn>=2 THEN (c.d_raw>1)::int END) twins_raw_differ,
  avg(CASE WHEN c.c_kn>=2 THEN (c.d_low>1)::int END) twins_lower_differ,
  avg((al.xn = k.n)::int) sx_eq_raw_s1,
  avg(CASE WHEN c.c_kn>=2 AND c.d_raw>1 THEN (al.xn=k.n)::int END) sx_eq_raw_when_twins_differ
FROM al JOIN s1k k ON k.id=al.s1 JOIN cnt c ON c.kn=k.kn
"""
print(c, con.execute(q).df().T)
# among twins-with-differing-raw: does SX raw name equal the TRUE S1's raw more than a twin's raw?
q2 = """
WITH al AS (SELECT g.s1, g.sx, x.n xn FROM g JOIN sx x ON x.id=g.sx WHERE trim(x.a) IN ('','NULL','<NULL>')),
t AS (SELECT al.*, k.kn, k.n tn FROM al JOIN s1k k ON k.id=al.s1 JOIN cnt c ON c.kn=k.kn WHERE c.c_kn>=2 AND c.d_raw>1)
SELECT count(*) n,
  avg((xn=tn)::int) eq_true,
  avg((xn IN (SELECT n FROM s1k o WHERE o.kn=t.kn AND o.id<>t.s1))::int) eq_other_twin,
  avg((lower(xn)=lower(tn))::int) leq_true
FROM t
"""
print(con.execute(q2).df().T)
