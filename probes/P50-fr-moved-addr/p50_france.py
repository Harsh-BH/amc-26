import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_france():
    print('\n--- France ---')
    con = duckdb.connect()
    con.execute("SET memory_limit='1200MB'")
    
    # 1. France S1 count
    q_s1 = f"SELECT COUNT(*) FROM '{ROOT}/data/work/rf_test_s1_France.parquet'"
    france_s1 = con.execute(q_s1).fetchone()[0]
    print(f"France S1 count: {france_s1}")
    
    # We need twin counts for France
    con.execute(f"""
        CREATE TABLE fr_tc AS
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_test_s1_France.parquet' 
        WHERE core != '' GROUP BY core
    """)
    
    # France P31 pairs:
    # "only SX not assigned to any S1 in the base" -> on P31, this means acc == False for this SX globally.
    # We can just look at acc == False and argmax == True.
    # The columns: hn_cls (1 = sibling, 0 = equal, 2/3/4 = different), name_1, name_x (raw), city_1, city_x, core_1, core_x
    q = f"""
    WITH p31 AS (
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
    )
    SELECT 
        name_raw_eq,
        k,
        COUNT(*) as pairs_to_add
    FROM p31
    GROUP BY 1, 2
    """
    df = con.execute(q).df()
    
    df_req = df[(df['name_raw_eq'] == True) & (df['k'] == 1)]
    adds = df_req['pairs_to_add'].sum() if not df_req.empty else 0
    
    print(f"\nFor name_raw_eq=True, k=1, hn_cls != 1 (and != 0):")
    print(f"France pairs to ADD: {adds}")
    
    N_neg_india = 56.90
    N_neg_us = 4.00
    
    if adds > 0:
        prec_india = 1 - N_neg_india / adds
        prec_us = 1 - N_neg_us / adds
        print(f"India-rate bound: {prec_india:.4f}")
        print(f"US-rate bound: {prec_us:.4f}")
    
    # Label-free France stats: share of such pairs vs US/India share among argmax pairs
    # Wait, we need to compare the size of the cell in France vs US/India to make sure France doesn't have way more of these.
    # Let's count ALL argmax pairs for this cell in France vs US vs India.
    # In France, argmax pairs:
    q_fr_all = f"""
        SELECT COUNT(*)
        FROM '{ROOT}/data/work/p31/pairs_France.parquet' p
        JOIN fr_tc t ON p.core_1 = t.core
        WHERE p.argmax = True
          AND p.core_1 = p.core_x
          AND p.city_1 = p.city_x
          AND p.core_1 != ''
          AND p.city_1 != ''
          AND p.hn_cls != 1
          AND p.hn_cls != 0
          AND LOWER(p.name_1) = LOWER(p.name_x)
          AND t.k = 1
    """
    fr_all = con.execute(q_fr_all).fetchone()[0]
    print(f"France cell size (all argmax): {fr_all}")
    print(f"France cell size per S1: {fr_all / france_s1:.6f}")

if __name__ == '__main__':
    run_france()
