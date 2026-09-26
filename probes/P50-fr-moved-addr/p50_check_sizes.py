import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_sizes(country):
    con = duckdb.connect()
    
    q_s1 = f"SELECT COUNT(*) FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet'"
    s1_count = con.execute(q_s1).fetchone()[0]
    
    q_pairs = f"""
    WITH tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
        WHERE core != '' GROUP BY core
    )
    SELECT COUNT(*) 
    FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1
    JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx 
      ON s1.core = sx.core 
      AND s1.city = sx.city
    JOIN tc ON s1.core = tc.core
    JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}') s1_raw 
      ON s1.id = s1_raw.entity_id
    JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' UNION ALL SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw 
      ON sx.id = sx_raw.entity_id
    WHERE s1.core != '' 
      AND s1.city != ''
      AND s1.hn != sx.hn
      AND tc.k = 1
      AND LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name)
      AND NOT (
        abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21)
      )
    """
    pairs = con.execute(q_pairs).fetchone()[0]
    print(f"{country} cell size: {pairs}")
    print(f"{country} cell size per S1: {pairs / s1_count:.6f}")

if __name__ == '__main__':
    run_sizes('India')
    run_sizes('US')
