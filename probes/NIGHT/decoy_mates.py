# Decoy-word same-hn swap cell: does "other SX sharing this SX's (core, city)" separate true variants from phantom decoys?
import sys; import numpy as np, pandas as pd, duckdb
sys.path[:0] = ["probes/P46-noise-word-tier"]
import p46, r05 as R
W = p46.W
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'")
def mates(split, c, ids):
    q = pd.DataFrame({"qid": pd.unique(ids)}); con.register("q", q)
    t = con.execute(f"""WITH x0 AS (SELECT id, core, city FROM 'data/work/rf_{split}_sx_{c}.parquet'),
        x AS (SELECT x0.* FROM x0 JOIN q ON q.qid = x0.id),
        k AS (SELECT core, city, count(*) m FROM x0 WHERE (core, city) IN (SELECT core, city FROM x) GROUP BY 1,2),
        s AS (SELECT core, city, count(*) k1 FROM 'data/work/rf_{split}_s1_{c}.parquet' GROUP BY 1,2)
        SELECT x.id, k.m - 1 mates, coalesce(s.k1,0) s1_same FROM x JOIN k USING(core, city) LEFT JOIN s USING(core, city)""").df()
    return t.set_index("id").reindex(ids)
def band(m): return pd.cut(m, [-1, 0, 1, 3, 1e9], labels=["0", "1", "2-3", "4+"])
for c in ["US", "India"]:
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "am", "y", "pred"]); d = d[d.holdout & d.am].reset_index(drop=True)
    lt = p46.lens_table(d[["s1", "sx"]], "train", c)
    d["w"] = p46.swap_words(d[["s1", "sx"]], "train", c); x = d[d.w != ""].merge(lt[["s", "n"]], left_on="w", right_index=True)
    r1 = R._rf(W / f"rf_train_s1_{c}.parquet", pd.unique(x.s1), ["hn"]); rx = R._rf(W / f"rf_train_sx_{c}.parquet", pd.unique(x.sx), ["hn"])
    h1, hx = r1.hn.reindex(x.s1).fillna("").to_numpy(), rx.hn.reindex(x.sx).fillna("").to_numpy()
    x = x[(h1 != "") & (h1 == hx) & (x.s > 0.2) & (x.n >= 20)]
    mt = mates("train", c, x.sx.to_numpy()); x["mates"] = mt.mates.to_numpy(); x["s1_same"] = mt.s1_same.to_numpy()
    print(c, len(x)); print(x.groupby(band(x.mates), observed=True).agg(n=("y", "size"), prec=("y", "mean"), acc=("pred", "mean")).round(3).to_string())
fr = pd.read_parquet("data/work/p54_sub1_hneq.parquet"); fr = fr[(fr.s > 0.2) & (fr.n >= 20)]
mt = mates("test", "France", fr.sx.to_numpy()); fr["mates"] = mt.mates.to_numpy()
print("France", len(fr)); print(pd.crosstab(band(fr.mates), fr.state))
