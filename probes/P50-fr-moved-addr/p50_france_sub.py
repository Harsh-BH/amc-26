import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_france():
    con = duckdb.connect()
    
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
            t.k
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
    SELECT id FROM p31
    """
    df = con.execute(q).df()
    print("Base pairs to add:", len(df))
    
    # Check s1_has_same_addr for these
    # Actually, for France, we don't have gt_pairs (it's test set). We can't check s1_has_same_addr by looking at gt_pairs!
    # Ah!! s1_has_same_addr is "whether the S1 has other members at its own address" in TRUE pairs.
    # In the test set, we don't have true pairs!
    # So we CANNOT USE `s1_has_same_addr` AS A RULE in the test set!
