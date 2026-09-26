import sys; sys.path[:0]=["."]
import pandas as pd, numpy as np, p49
fr = pd.read_parquet("../../data/work/p31/pairs_France.parquet", columns=["s1","sx","p","acc","argmax","hn_cls","hn"])
m = p49.rule_p49(fr[["s1","sx","p"]], "test", "France", 0.9868)
nonempty = fr.groupby("s1").acc.transform("any")
x = fr[m["arm_eq"] & ~fr.acc & fr.argmax.astype(bool)].copy(); x["nonemp"] = nonempty[x.index]
print("rejected am", len(x)); x["hnx"] = (x.p >= 0.8) & x.nonemp
print("hn-extra would take (p>=.8 & nonempty):", int(x.hnx.sum()), " gc-ish (p>=.7):", int((x.p>=0.7).sum()))
r = x[~x.hnx]; print("left", len(r), "fills empty S1:", int((~r.nonemp).sum()))
print(pd.cut(r.p,[0,.1,.3,.5,.7,.8,.9,.99,1]).value_counts().sort_index())
