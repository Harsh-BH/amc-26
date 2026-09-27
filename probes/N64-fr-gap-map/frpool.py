import duckdb, numpy as np, pandas as pd
con=duckdb.connect(); con.execute("SET memory_limit='4000MB'")

con.execute("""create temp table m as select source1_entity_id s1, unnest(string_split(matched_entity_ids,',')) sx
 from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
con.execute("create temp table own as select sx, s1 own1 from m")
con.execute("create temp table k as select s1, count(*) k from m group by 1")
con.execute("""create temp table a as select p.s1, p.sx, p.p, p.r, p.hn_cls, p.argmax, o.own1, coalesce(k.k,0) k1
 from 'data/work/p31/pairs_France.parquet' p left join own o on p.sx=o.sx left join k on p.s1=k.s1""")
print(con.execute("""select argmax, case when own1 is null then 'free' when own1=s1 then 'acc' else 'other' end st,
 count(*) n from a group by 1,2 order by 1,2""").df())
print(con.execute("""select case when p>=.99 then 'a.99+' when p>=.9 then 'b.9-.99' when p>=.7 then 'c.7-.9' when p>=.5 then 'd.5-.7' when p>=.2 then 'e.2-.5' else 'f<.2' end pb,
 sum((own1 is null)::int) free, sum((own1=s1)::int) acc, sum((own1 is null and k1=0)::int) free_emptyS1,
 sum((own1 is null and r is not null)::int) free_has_r, avg(case when own1 is null then r end) r_free, avg(case when own1=s1 then r end) r_acc
 from a where argmax group by 1 order by 1""").df().to_string())
