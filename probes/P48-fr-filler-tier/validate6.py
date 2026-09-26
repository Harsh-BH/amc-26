import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import gc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
W = ROOT / "data/work"

def run(country):
    print(f"[{country}] Starting...", flush=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    
    # 1. Filter p23 directly in duckdb
    print("Filtering p23...", flush=True)
    con.execute(f"""
        CREATE TABLE pairs AS 
        SELECT s1, sx, p, y, am, pred, holdout 
        FROM '{W}/p23/pairs_{country}.parquet'
        WHERE holdout = true OR am = true
    """)
    
    # 2. Get accepted SX
    con.execute(f"""
        CREATE TABLE acc_sx AS
        SELECT DISTINCT sx 
        FROM '{W}/p23/pairs_{country}.parquet'
        WHERE pred = true
    """)
    
    # 3. Create a unified source table for this country (lazy)
    sources = [f"'{W}/train_source{i}.parquet'" for i in (1,2,3)]
    union_q = " UNION ALL ".join([f"SELECT entity_id, business_name, business_address FROM {s} WHERE country='{country}'" for s in sources])
    con.execute(f"CREATE VIEW raw AS {union_q}")
    
    # 4. Join pairs with raw strings and rf features
    print("Joining...", flush=True)
    q = f"""
        SELECT 
            p.*,
            p.sx NOT IN (SELECT sx FROM acc_sx) AS sx_is_free,
            r1.business_name AS name_1,
            rx.business_name AS name_x,
            r1.business_address AS addr_1,
            rx.business_address AS addr_x,
            f1.hn AS hn_1,
            fx.hn AS hn_x,
            f1.city AS city_1,
            fx.city AS city_x
        FROM pairs p
        LEFT JOIN raw r1 ON p.s1 = r1.entity_id
        LEFT JOIN raw rx ON p.sx = rx.entity_id
        LEFT JOIN '{W}/rf_train_s1_{country}.parquet' f1 ON p.s1 = f1.id
        LEFT JOIN '{W}/rf_train_sx_{country}.parquet' fx ON p.sx = fx.id
    """
    
    df = con.execute(q).df()
    df.to_parquet(f"/tmp/p48_joined_{country}.parquet")
    print(f"Saved {country} joined data, shape: {df.shape}")
    
if __name__ == "__main__":
    run("US")
    run("India")
