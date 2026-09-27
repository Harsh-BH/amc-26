import duckdb, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
con = duckdb.connect(); con.execute("SET memory_limit='3500MB'")
con.execute("""create temp table m as select source1_entity_id s1, unnest(string_split(matched_entity_ids,',')) sx from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
con.execute("create temp table k as select s1, count(*) k from m group by 1")
d = con.execute("""select p.s1, p.sx, p.p, p.r, p.p_best_x, p.p_2nd_x, p.n_x, p.name_rel, p.name_1, p.name_x, p.addr_1, p.core_1, p.core_x, m.s1 own, coalesce(k.k,0) k1
  from 'data/work/p31/pairs_France.parquet' p left join m on m.sx=p.sx left join k on k.s1=p.s1
  where p.argmax and trim(p.addr_x) in ('','NULL','<NULL>')""").df()
d["st"] = np.select([d.own.isna(), d.own == d.s1], ["free", "acc"], "other")
print("France empty-addr SX argmax pairs:", len(d), d.st.value_counts().to_dict())
d["pb"] = pd.cut(d.p, [0, .05, .2, .4, .6, .8, .9868, 1.01]); d["gb"] = pd.cut(d.p_2nd_x.fillna(0), [-.01, .01, .05, .2, 1.01])
print(d.pivot_table(index="pb", columns="gb", values="sx", aggfunc="size", observed=True).fillna(0).astype(int).to_string())
print("acc share by cell"); print(d.assign(a=d.st == "acc").pivot_table(index="pb", columns="gb", values="a", aggfunc="mean", observed=True).round(2).to_string())
f = d[(d.st == "free") & (d.p >= .2) & (d.p_2nd_x.fillna(0) < .05)]
print("free p>=.2 & p2<.05:", len(f), "| name_rel", f.name_rel.value_counts().head(6).to_dict(), "| r median", f.r.median(), "r>=.5", (f.r >= .5).mean().round(3), "| S1 k1=0", (f.k1 == 0).sum())
print(f.sample(25, random_state=3)[["name_1", "name_x", "addr_1", "p", "p_2nd_x", "r", "k1"]].round(3).to_string())
d.to_parquet("data/work/n69/fr_noaddr.parquet", index=False)
