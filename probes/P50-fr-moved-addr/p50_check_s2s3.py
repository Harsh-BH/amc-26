import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_s2s3(country):
    con = duckdb.connect()
    
    q_s1 = f"SELECT COUNT(DISTINCT s1) FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' WHERE holdout = True"
    holdout_s1 = con.execute(q_s1).fetchone()[0]
    
    q = f"""
    WITH tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
        WHERE core != '' GROUP BY core
    ),
    pairs AS (
        SELECT 
            p.y,
            LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name) as name_raw_eq,
            tc.k,
            CASE WHEN p.sx LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as sx_source
        FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' p
        JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON p.s1 = s1.id
        JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON p.sx = sx.id
        JOIN tc ON s1.core = tc.core
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}') s1_raw 
          ON s1.id = s1_raw.entity_id
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' UNION ALL SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw 
          ON sx.id = sx_raw.entity_id
        WHERE p.holdout = True
          AND s1.core = sx.core
          AND s1.city = sx.city
          AND s1.core != '' 
          AND s1.city != ''
          AND s1.hn != sx.hn
          AND tc.k = 1
          AND LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name)
          AND NOT (
            abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21)
          )
    )
    SELECT 
        sx_source,
        COUNT(*) as total,
        SUM(CASE WHEN NOT y THEN 1 ELSE 0 END) as falses
    FROM pairs
    GROUP BY 1
    """
    df = con.execute(q).df()
    print(f"\n--- {country} ---")
    print(df)
    
    for _, row in df.iterrows():
        src = row['sx_source']
        f = row['falses']
        n_neg = (f / holdout_s1) * 259452
        print(f"Source {src}: falses = {f}, N_neg = {n_neg:.2f}")

if __name__ == '__main__':
    run_s2s3('India')
    run_s2s3('US')
