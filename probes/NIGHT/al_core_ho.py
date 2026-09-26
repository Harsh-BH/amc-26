# Holdout (model out of fold, P23 dump): address-less SX whose core equals exactly one country S1 core (full train S1),
# pair (that S1, SX) in the dump. Precision among pairs the #8c-style decision left unaccepted, by p band.
import duckdb, pandas as pd
con = duckdb.connect(); con.execute("SET memory_limit='2500MB'; SET threads=4")
for c in ["US", "India"]:
    df = con.execute(f"""
    WITH al AS (SELECT entity_id id FROM (SELECT * FROM 'data/work/train_source2.parquet' UNION ALL SELECT * FROM 'data/work/train_source3.parquet')
                WHERE country='{c}' AND trim(business_address) IN ('','NULL','<NULL>')),
    x AS (SELECT r.id, r.core FROM 'data/work/rf_train_sx_{c}.parquet' r JOIN al USING(id)),
    s AS (SELECT id, core FROM 'data/work/rf_train_s1_{c}.parquet'),
    k AS (SELECT core, count(*) k FROM s GROUP BY 1 HAVING count(*)=1),
    u AS (SELECT x.id sx, s.id s1 FROM x JOIN k USING(core) JOIN s USING(core)),
    p AS (SELECT * FROM 'data/work/p23/pairs_{c}.parquet' WHERE holdout),
    taken AS (SELECT DISTINCT sx FROM p WHERE pred)
    SELECT p.p, p.am, p.pred, p.y, (p.sx IN (SELECT sx FROM taken)) sx_taken FROM u JOIN p USING(s1, sx)""").df()
    print(c, len(df), "pred", df.pred.mean().round(3), "y", df.y.mean().round(4))
    r = df[~df.pred]
    r = r.assign(pb=pd.cut(r.p, [0, .1, .3, .5, .7, .9, 1.0]))
    print(r.groupby(["sx_taken", "pb"], observed=True).agg(n=("y", "size"), y=("y", "mean"), am=("am", "mean")))
