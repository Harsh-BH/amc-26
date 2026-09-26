import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_task1(country):
    print(f'\n--- {country} ---')
    con = duckdb.connect()
    
    # Get all s1
    s1_df = con.execute(f"SELECT id, core, city, hn FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' WHERE core != '' AND city != ''").df()
    
    # Twin counts
    core_counts = s1_df.groupby('core').size().reset_index(name='k')
    s1_df = s1_df.merge(core_counts, on='core', how='left')
    
    # To prevent massive memory usage, limit to k <= 10 or do it iteratively?
    # Let's just do it directly using a duckdb query with group by
    
    q = f"""
    SELECT 
        LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name) as name_raw_eq,
        tc.k,
        CASE WHEN sx.id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as sx_source,
        COUNT(*) as total,
        SUM(CASE WHEN g.s1 IS NOT NULL THEN 1 ELSE 0 END) as trues
    FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1
    JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx 
      ON s1.core = sx.core 
      AND s1.city = sx.city
      AND s1.core != '' 
      AND s1.city != ''
      AND s1.hn != sx.hn
    JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}') s1_raw 
      ON s1.id = s1_raw.entity_id
    JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' UNION ALL SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw 
      ON sx.id = sx_raw.entity_id
    LEFT JOIN '{ROOT}/data/work/gt_pairs.parquet' g ON s1.id = g.s1 AND sx.id = g.sx
    LEFT JOIN (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
        WHERE core != '' GROUP BY core
    ) tc ON s1.core = tc.core
    WHERE NOT (
        abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21)
    )
    GROUP BY 1, 2, 3
    """
    df = con.execute(q).df()
    df['prec'] = df['trues'] / df['total']
    
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
