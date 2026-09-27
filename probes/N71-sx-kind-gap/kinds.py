"""N71: per SX kind (label-free), train truth rate vs test assignment rate (sub76) per country."""
import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='3500MB'")
con.execute("""create temp table own as select unnest(string_split(matched_entity_ids,',')) sx from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
KIND = """case when trim(business_address) in ('','NULL','<NULL>') then 'a_noaddr'
  when business_name like '%|%' then 'b_pipe'
  when regexp_matches(lower(business_name), '(www\\.|\\.com|\\.in\\b|\\.fr\\b|\\.net|\\.org|https?:)') then 'c_web'
  when business_name = upper(business_name) and business_name <> lower(business_name) then 'd_caps'
  when business_name = lower(business_name) and business_name <> upper(business_name) then 'e_lower'
  when regexp_matches(business_name, '[0-9]') then 'f_digit'
  else 'g_plain' end"""
for split in ("train", "test"):
    for c in (("India", "US") if split == "train" else ("India", "US", "France")):
        q = f"""select {KIND} k, count(*) n, {'avg((g.sx is not null)::int)' if split=='train' else 'avg((o.sx is not null)::int)'} rate from
          (select entity_id, business_name, business_address from '{'data/work/'+split}_source2.parquet' where country='{c}' union all
           select entity_id, business_name, business_address from '{'data/work/'+split}_source3.parquet' where country='{c}') t
          {"left join 'data/work/gt_pairs.parquet' g on g.sx=t.entity_id" if split=='train' else 'left join own o on o.sx=t.entity_id'} group by 1 order by 1"""
        r = con.execute(q).df()
        print(split, c, "  ".join(f"{k}:{n}/{v:.3f}" for k, n, v in r.itertuples(index=False)), flush=True)
