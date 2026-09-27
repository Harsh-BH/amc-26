import duckdb, sys
sys.path[:0]=["code/business_entity_resolution/src"]
con = duckdb.connect(); con.execute("SET memory_limit='3000MB'")
con.execute("""create temp table own as select unnest(string_split(matched_entity_ids,',')) sx from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
for c in ("US","India","France"):
    n1 = con.execute(f"select count(*) from 'data/work/rf_test_s1_{c}.parquet'").fetchone()[0]
    r = con.execute(f"""select (coalesce(r.hn,'')='') nohn, count(*)/{n1} per_s1, avg((o.sx is not null)::int) assigned
      from 'data/work/rf_test_sx_{c}.parquet' r left join own o on o.sx=r.id group by 1 order by 1""").df()
    print(c, n1); print(r.round(4).to_string())
# France: fixed-hn view — raw address empty / NULL vs has text without hn
r = con.execute("""select case when trim(t.business_address) in ('','NULL','<NULL>') then 'empty' when regexp_matches(t.business_address,'\\(\\d+\\)') then 'paren' else 'text' end k,
   count(*)/259452 per_s1, avg((o.sx is not null)::int) assigned
   from 'data/work/rf_test_sx_France.parquet' r join (select entity_id, business_address from 'data/work/test_source2.parquet' where country='France' union all select entity_id, business_address from 'data/work/test_source3.parquet' where country='France') t on t.entity_id=r.id
   left join own o on o.sx=r.id where coalesce(r.hn,'')='' group by 1""").df()
print("France no-hn SX by address kind"); print(r.round(4))
