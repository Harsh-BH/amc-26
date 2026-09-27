import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='3000MB'")
con.execute("""create temp table own as select unnest(string_split(matched_entity_ids,',')) sx from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
for c, n1 in (("US",663106),("India",809986),("France",259452)):
    r = con.execute(f"""select case when trim(t.business_address) in ('','NULL','<NULL>') then 'empty' else 'addr' end k, count(*)/{n1} per_s1, avg((o.sx is not null)::int) assigned
       from (select entity_id, business_address from 'data/work/test_source2.parquet' where country='{c}' union all select entity_id, business_address from 'data/work/test_source3.parquet' where country='{c}') t
       left join own o on o.sx=t.entity_id group by 1 order by 1""").df()
    print(c); print(r.round(4).to_string())
# train truth: address-less SX that are true matches, per S1
for c in ("US","India"):
    n1 = con.execute(f"select count(*) from 'data/work/train_source1.parquet' where country='{c}'").fetchone()[0]
    r = con.execute(f"""select case when trim(t.business_address) in ('','NULL','<NULL>') then 'empty' else 'addr' end k, count(*)/{n1} per_s1, avg((g.sx is not null)::int) true_rate
       from (select entity_id, business_address from 'data/work/train_source2.parquet' where country='{c}' union all select entity_id, business_address from 'data/work/train_source3.parquet' where country='{c}') t
       left join 'data/work/gt_pairs.parquet' g on g.sx=t.entity_id group by 1 order by 1""").df()
    print("train", c); print(r.round(4).to_string())
