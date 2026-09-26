import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_france():
    con = duckdb.connect()
    con.execute("SET memory_limit='1200MB'")
    
    q = f"""
    WITH fr_tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_test_s1_France.parquet' 
        WHERE core != '' GROUP BY core
    ),
    p31 AS (
        SELECT 
            p.*,
            LOWER(p.name_1) = LOWER(p.name_x) as name_raw_eq,
            t.k,
            CASE WHEN p.src LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as sx_source
        FROM '{ROOT}/data/work/p31/pairs_France.parquet' p
        JOIN fr_tc t ON p.core_1 = t.core
        WHERE p.argmax = True
          AND p.acc = False
          AND p.core_1 = p.core_x
          AND p.city_1 = p.city_x
          AND p.core_1 != ''
          AND p.city_1 != ''
          AND p.hn_cls != 1
          AND p.hn_cls != 0
          AND LOWER(p.name_1) = LOWER(p.name_x)
          AND t.k = 1
    )
    SELECT sx_source, COUNT(*) as adds FROM p31 GROUP BY 1
    """
    df = con.execute(q).df()
    print(df)
    
if __name__ == '__main__':
    run_france()
