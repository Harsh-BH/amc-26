import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_task1(country):
    print(f'\n--- {country} (k <= 10) ---')
    con = duckdb.connect()
    
    q = f"""
    WITH tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
        WHERE core != '' GROUP BY core
    ),
    pairs AS (
        SELECT 
            LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name) as name_raw_eq,
            tc.k,
            CASE WHEN sx.id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as sx_source,
            CASE WHEN g.s1 IS NOT NULL THEN 1 ELSE 0 END as is_true
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1
        JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx 
          ON s1.core = sx.core 
          AND s1.city = sx.city
          AND s1.core != '' 
          AND s1.city != ''
          AND s1.hn != sx.hn
        JOIN tc ON s1.core = tc.core
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}') s1_raw 
          ON s1.id = s1_raw.entity_id
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' UNION ALL SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw 
          ON sx.id = sx_raw.entity_id
        LEFT JOIN '{ROOT}/data/work/gt_pairs.parquet' g ON s1.id = g.s1 AND sx.id = g.sx
        WHERE tc.k <= 10
          AND NOT (
            abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21)
          )
    )
    SELECT 
        name_raw_eq,
        k,
        sx_source,
        COUNT(*) as total,
        SUM(is_true) as trues
    FROM pairs
    GROUP BY 1, 2, 3
    """
    df = con.execute(q).df()
    
    print("Overall:")
    print(df[['total', 'trues']].sum())
    
    print("\nBy name_raw_eq:")
    print(df.groupby('name_raw_eq')[['total', 'trues']].sum().assign(prec=lambda x: x['trues']/x['total']))
    
    print("\nBy k (name_raw_eq=True):")
    df_req = df[df['name_raw_eq'] == True]
    print(df_req.groupby('k')[['total', 'trues']].sum().assign(prec=lambda x: x['trues']/x['total']).head(10))

if __name__ == '__main__':
    run_task1('India')
    run_task1('US')
