import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run_revised(country):
    con = duckdb.connect()
    
    q_s1 = f"SELECT COUNT(DISTINCT s1) FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' WHERE holdout = True"
    holdout_s1 = con.execute(q_s1).fetchone()[0]
    
    q = f"""
    WITH tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
        WHERE core != '' GROUP BY core
    ),
    s1_same_addr AS (
        SELECT DISTINCT g.s1
        FROM '{ROOT}/data/work/gt_pairs.parquet' g
        JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON g.s1 = s1.id
        JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON g.sx = sx.id
        WHERE s1.hn = sx.hn AND s1.hn != ''
    ),
    pairs AS (
        SELECT 
            p.y,
            LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name) as name_raw_eq,
            tc.k,
            CASE WHEN ssa.s1 IS NOT NULL THEN True ELSE False END as s1_has_same_addr
        FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' p
        JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON p.s1 = s1.id
        JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON p.sx = sx.id
        JOIN tc ON s1.core = tc.core
        LEFT JOIN s1_same_addr ssa ON s1.id = ssa.s1
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
          AND tc.k <= 10
          AND NOT (
            abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21)
          )
    )
    SELECT 
        s1_has_same_addr,
        COUNT(*) as total_holdout_pairs,
        SUM(CASE WHEN y THEN 1 ELSE 0 END) as trues,
        SUM(CASE WHEN NOT y THEN 1 ELSE 0 END) as falses
    FROM pairs
    WHERE name_raw_eq = True AND k = 1
    GROUP BY 1
    """
    df = con.execute(q).df()
    
    print(f"\n--- {country} ---")
    print(df)
    
    # Calculate for s1_has_same_addr == False
    df_req = df[df['s1_has_same_addr'] == False]
    falses = df_req['falses'].sum() if not df_req.empty else 0
    neg_per_s1 = falses / holdout_s1 if holdout_s1 > 0 else 0
    n_neg = neg_per_s1 * 259452
    print(f"\nFor name_raw_eq=True, k=1, s1_has_same_addr=False:")
    print(f"False pairs: {falses}")
    print(f"neg_per_s1 = {neg_per_s1:.6f}")
    print(f"Expected France negatives (N_neg): {n_neg:.2f}")

if __name__ == '__main__':
    run_revised('India')
    run_revised('US')
