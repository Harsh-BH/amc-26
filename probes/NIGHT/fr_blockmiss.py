# France: SX never in any candidate list — do they have an S1 with the same core (and same city)? Compare with US/India
# test SX never in candidates (p34 te_pairs are India/US test pairs? use p3 cands instead) and train truth rates.
import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='2000MB'; SET threads=4")
con.execute("""
CREATE TABLE cs AS SELECT DISTINCT sx FROM 'data/work/p31/pairs_France.parquet';
CREATE TABLE x AS SELECT id, core, city, hn FROM 'data/work/rf_test_sx_France.parquet' WHERE id NOT IN (SELECT sx FROM cs);
CREATE TABLE s AS SELECT id, core, city, hn FROM 'data/work/rf_test_s1_France.parquet';
CREATE TABLE sk AS SELECT core, count(*) k FROM s GROUP BY 1;
""")
print(con.execute("""SELECT count(*) n_nocand,
  avg((x.core IN (SELECT core FROM s))::int) core_match_any,
  avg((EXISTS (SELECT 1 FROM s WHERE s.core=x.core AND s.city=x.city AND x.city<>''))::int) core_city_match,
  avg((EXISTS (SELECT 1 FROM s WHERE s.core=x.core AND s.city=x.city AND s.hn=x.hn AND x.hn<>''))::int) core_city_hn_match,
  avg((x.core='')::int) empty_core
  FROM x""").df().T)
print(con.execute("""SELECT coalesce(sk.k,0) k, count(*) n FROM x LEFT JOIN sk USING(core) GROUP BY 1 ORDER BY 1 LIMIT 8""").df().T)
