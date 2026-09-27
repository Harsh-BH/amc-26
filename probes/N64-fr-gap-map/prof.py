import duckdb
con=duckdb.connect(); con.execute("SET memory_limit='1500MB'")
con.execute("""create view m as select * from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true)""")
con.execute("create view s1 as select entity_id, country from 'data/work/test_source1.parquet'")
print(con.execute("""select country, count(*) n, avg(case when coalesce(matched_entity_ids,'')='' then 1 else 0 end) empty,
 avg(case when coalesce(matched_entity_ids,'')='' then 0 else len(string_split(matched_entity_ids,',')) end) mean_k,
 avg(case when len(string_split(coalesce(matched_entity_ids,''),','))=1 and coalesce(matched_entity_ids,'')<>'' then 1 else 0 end) k1,
 avg(case when len(string_split(coalesce(matched_entity_ids,''),','))=2 then 1 else 0 end) k2,
 avg(case when len(string_split(coalesce(matched_entity_ids,''),','))=3 then 1 else 0 end) k3,
 avg(case when len(string_split(coalesce(matched_entity_ids,''),','))=4 then 1 else 0 end) k4,
 avg(case when len(string_split(coalesce(matched_entity_ids,''),','))>=5 then 1 else 0 end) k5
 from m join s1 on m.source1_entity_id=s1.entity_id group by 1""").df().to_string())
