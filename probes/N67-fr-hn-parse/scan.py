import re, numpy as np, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
f = pd.read_parquet("data/work/n64/frfree_lf.parquet")
a = pd.read_parquet("data/work/n64/frfree.parquet", columns=["s1","sx","addr_1","addr_x","hn_cls","city_eq"])
f = f.merge(a, on=["s1","sx"])
num = lambda s: tuple(re.findall(r"\d+", s))
f["n1"] = f.addr_1.map(num); f["nx"] = f.addr_x.map(num)
f["first_eq"] = [bool(a) and bool(b) and a[0] == b[0] for a, b in zip(f.n1, f.nx)]
f["set_eq"] = [bool(a) and set(a) == set(b) for a, b in zip(f.n1, f.nx)]
x = f[(f.hr != "eq") & f.first_eq & (f.p >= .5)]
print("free p>=.5, parsed hn not equal but first raw number equal:", len(x), "| set_eq", x.set_eq.sum(), "| by name_rel", x.name_rel.value_counts().head(6).to_dict())
print("hn pairs:", pd.Series(list(zip(x.hn_1, x.hn_x))).value_counts().head(15).to_dict())
print(x.sample(20, random_state=2)[["name_1","name_x","addr_1","addr_x","hn_1","hn_x","p","r"]].round(3).to_string())
