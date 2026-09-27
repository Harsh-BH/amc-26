"""N68: per-S1 class mix of TRUE pairs (train truth, India/US) vs France's ACCEPTED pairs (sub76). Classes are label-free
(house-number relation x source x SX address-less). Where France sits below the train law = its recall deficit."""
import sys, numpy as np, pandas as pd, duckdb
sys.path[:0] = ["code/business_entity_resolution/src"]
SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
def hcls(h1, hx):
    out = []
    for a, b in zip(h1, hx):
        if not a and not b: out.append("none_both")
        elif not b: out.append("sx_none")
        elif not a: out.append("s1_none")
        elif a == b: out.append("eq")
        else:
            try: d = int(b) - int(a)
            except ValueError: out.append("other"); continue
            if d in SIB: out.append("sib+")
            elif -d in SIB: out.append("sib-")
            elif len(a) != len(b) and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)): out.append("digit")
            else: out.append("other")
    return out
con = duckdb.connect(); con.execute("SET memory_limit='2500MB'")
res = {}
for c in ("India", "US"):
    t = con.execute(f"""select g.s1, g.sx, a.hn h1, b.hn hx from 'data/work/gt_pairs.parquet' g
        join 'data/work/rf_train_s1_{c}.parquet' a on a.id=g.s1 join 'data/work/rf_train_sx_{c}.parquet' b on b.id=g.sx""").df()
    n1 = con.execute(f"select count(*) from 'data/work/rf_train_s1_{c}.parquet'").fetchone()[0]
    t["cls"] = hcls(t.h1.fillna(""), t.hx.fillna("")); t["src"] = t.sx.str[:2]
    res[f"truth_{c}"] = t.groupby(["src", "cls"]).size() / n1
    print(c, "truth pairs/S1", round(len(t) / n1, 3), flush=True)
f = pd.read_parquet("data/work/n64/fr_argmax_fixed.parquet", columns=["s1", "sx", "hn_1", "hn_x", "acc"])
a = f[f.acc]; n1 = 259452
a = a.assign(cls=hcls(a.hn_1.fillna(""), a.hn_x.fillna("")), src=a.sx.str[:2])
res["France_acc"] = a.groupby(["src", "cls"]).size() / n1
print("France accepted (argmax, in p31 frame)/S1", round(len(a) / n1, 3))
free = f[f.acc == False].merge(pd.read_parquet("data/work/n64/frfree.parquet", columns=["s1","sx","p"]), on=["s1","sx"])
free = free.assign(cls=hcls(free.hn_1.fillna(""), free.hn_x.fillna("")), src=free.sx.str[:2])
res["France_free_p>=.5"] = free[free.p >= .5].groupby(["src", "cls"]).size() / n1
R = pd.DataFrame(res).fillna(0)
R["truth_mean"] = (R.truth_India + R.truth_US) / 2
R["deficit"] = R.truth_mean - R.France_acc
print(R.round(4).sort_values("deficit", ascending=False).to_string())
print("sum deficit", R.deficit.sum().round(4))
