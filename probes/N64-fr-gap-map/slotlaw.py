import duckdb
con=duckdb.connect(); con.execute("SET memory_limit='4000MB'")
con.execute("""create temp table c as select s1, sum(case when sx like 'S2-%' then 1 else 0 end) n2, sum(case when sx like 'S3-%' then 1 else 0 end) n3
from 'data/work/gt_pairs.parquet' group by 1""")
print("truth (train, non-singleton):")

print(con.execute("select avg((n2>0)::int) has2, avg((n3>0)::int) has3, avg((n2>0 and n3>0)::int) bothsrc from c").df())
# sub76 per country
con.execute("""create temp table m as select source1_entity_id s1, matched_entity_ids ms from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true)""")
con.execute("""create temp table mm as select m.s1, t.country, coalesce(len(list_filter(string_split(ms,','), x->x like 'S2-%')),0) n2,
 coalesce(len(list_filter(string_split(ms,','), x->x like 'S3-%')),0) n3 from m join 'data/work/test_source1.parquet' t on m.s1=t.entity_id""")
print(con.execute("select country, avg((n2>0 and n3>0)::int) bothsrc, avg((n2>0 and n3=0)::int) only2, avg((n3>0 and n2=0)::int) only3, avg(n2) m2, avg(n3) m3 from mm where n2+n3>0 group by 1").df())
