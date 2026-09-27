# N67: do '(n)' France SX get accepted less than same-shaped pairs with a plain address? (label-free, sub76)
import sys, re; sys.path[:0] = ["code/business_entity_resolution/src"]
import duckdb, numpy as np, pandas as pd
import france as FR, normalize as N
con = duckdb.connect(); con.execute("SET memory_limit='4000MB'")
con.execute("""create temp table m as select source1_entity_id s1, unnest(string_split(matched_entity_ids,',')) sx
 from read_csv('data/work/n63/sub76_matching.tsv',delim='\t',header=true,quote='',all_varchar=true) where coalesce(matched_entity_ids,'')<>''""")
d = con.execute("""select p.s1, p.sx, p.p, p.r, p.hn_1, p.hn_x, p.name_rel, p.addr_1, p.addr_x, p.name_1, p.name_x, p.city_eq,
  m.s1 own from 'data/work/p31/pairs_France.parquet' p left join m on m.sx=p.sx where p.argmax""").df()
d["paren"] = d.addr_x.str.contains(r"(?:^|,\s*)\(\d+\)", regex=True)
fx = FR.paren_hn(d.addr_x[d.paren])
d.loc[d.paren, "hn_x"] = N.house_number(fx).to_numpy()
d["hn_eq"] = (d.hn_1 != "") & (d.hn_1 == d.hn_x)
d["st1"] = FR.street_key(d.addr_1).to_numpy(); d["stx"] = FR.street_key(FR.paren_hn(d.addr_x)).to_numpy()
d["st_eq"] = (d.st1 != "") & (d.st1 == d.stx)
d["acc"] = d.own == d.s1; d["free"] = d.own.isna()
d["pb"] = pd.cut(d.p, [0, .1, .3, .5, .7, .9, .98, .9868, 1.0001], right=False)
d.drop(columns=["addr_1", "name_1"]).assign(pb=d.pb.astype(str)).to_parquet("data/work/n64/fr_argmax_fixed.parquet", index=False)
z = d[d.hn_eq]
print("hn-eq argmax pairs: paren", z.paren.sum(), "plain", (~z.paren).sum())
t = z.groupby(["pb", "paren"], observed=True).acc.agg(["size", "mean"]).unstack()
print(t.round(3).to_string())
t2 = z[z.p >= .5].groupby(["name_rel", "paren"]).acc.agg(["size", "mean"]).unstack()
print(t2[t2[("size", True)] >= 30].round(3).to_string())
