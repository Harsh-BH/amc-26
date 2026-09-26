import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run():
    con = duckdb.connect()
    
    q_fr_cell = f"""
    WITH fr_tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_test_s1_France.parquet' 
        WHERE core != '' GROUP BY core
    )
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
      AND NOT p.src LIKE 'S2%'
    """
    fr_cell = con.execute(q_fr_cell).fetchone()[0]
    
    q_fr_total = f"SELECT COUNT(*) FROM '{ROOT}/data/work/p31/pairs_France.parquet' WHERE argmax = True AND NOT src LIKE 'S2%'"
    fr_total = con.execute(q_fr_total).fetchone()[0]
    
    print(f"France S3 cell / total argmax S3: {fr_cell} / {fr_total} = {fr_cell/fr_total:.4f}")
    
    for country in ['India', 'US']:
        q_cell = f"""
        WITH tc AS (
            SELECT core, COUNT(*) as k 
            FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' 
            WHERE core != '' GROUP BY core
        )
        SELECT COUNT(*)
        FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' p
        JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON p.s1 = s1.id
        JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON p.sx = sx.id
        JOIN tc ON s1.core = tc.core
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}') s1_raw 
          ON s1.id = s1_raw.entity_id
        JOIN (SELECT entity_id, business_name FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw 
          ON sx.id = sx_raw.entity_id
        WHERE p.holdout = True
          AND p.am = True
          AND NOT p.sx LIKE 'S2-%'
          AND s1.core = sx.core
          AND s1.city = sx.city
          AND s1.core != '' 
          AND s1.city != ''
          AND s1.hn != sx.hn
          AND tc.k = 1
          AND LOWER(s1_raw.business_name) = LOWER(sx_raw.business_name)
          AND NOT (abs(TRY_CAST(sx.hn as FLOAT) - TRY_CAST(s1.hn as FLOAT)) IN (3, 4, 5, 7, 9, 11, 13, 21))
        """
        cell = con.execute(q_cell).fetchone()[0]
        
        q_total = f"SELECT COUNT(*) FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' WHERE holdout = True AND am = True AND NOT sx LIKE 'S2-%'"
        total = con.execute(q_total).fetchone()[0]
        print(f"{country} S3 cell / total argmax S3: {cell} / {total} = {cell/total:.4f}")

if __name__ == '__main__':
    run()
