import duckdb, pandas as pd, sys
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
con=duckdb.connect(); con.execute("SET memory_limit='4000MB'")
con.execute("""create temp table m as select source1_entity_id s1, unnest(string_split(matched_entity_ids,',')) sx
 from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
con.execute("create temp table own as select sx, s1 o from m")
con.execute("create temp table k as select s1, count(*) k from m group by 1")
con.execute("""create temp table f as select p.*, coalesce(k.k,0) k1 from 'data/work/p31/pairs_France.parquet' p
 left join own on p.sx=own.sx left join k on p.s1=k.s1 where p.argmax and own.o is null""")
con.execute("copy f to '/tmp/claude-1000/-home-epitaph-repos-ml-challenge/b6bd9027-bde0-42a2-bbff-74d428d2a055/scratchpad/frfree.parquet'")
print(con.execute("select name_rel, hn_cls, count(*) n, avg(p) p, avg(r) r from f where p>=.7 group by 1,2 order by n desc limit 25").df().to_string())
x=con.execute("select name_1,name_x,addr_1,addr_x,round(p,3) p,round(r,3) r,k1 from f where p>=.9 using sample 40 rows").df()
print(x.to_string())
