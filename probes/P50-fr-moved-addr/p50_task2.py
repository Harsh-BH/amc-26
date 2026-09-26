import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/src'))
sys.path.insert(0, str(ROOT / 'probes/P28-addr-anchor'))

from anchor import street_key

SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]

def run_task2(country):
    print(f'\n--- Task 2: {country} ---')
    con = duckdb.connect()
    con.execute("SET memory_limit='1200MB'")
    
    query = f"""
    SELECT 
        p.s1,
        p.sx,
        p.y,
        p.p,
        p.am,
        s1.core as core1,
        sx.core as corex,
        s1.city as city,
        s1.hn as hn1,
        sx.hn as hnx
    FROM '{ROOT}/data/work/p23/pairs_{country}.parquet' p
    JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON p.s1 = s1.id
    JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON p.sx = sx.id
    WHERE p.holdout = True 
      AND p.pred = False
      AND s1.core = sx.core
      AND s1.city = sx.city
      AND s1.core != '' 
      AND s1.city != ''
      AND s1.hn != sx.hn
    """
    df = con.execute(query).df()
    
    if len(df) == 0:
        print("No pairs found.")
        return
        
    def is_sib(h1, hx):
        try:
            return abs(float(hx) - float(h1)) in SIB3
        except:
            return False
            
    df['is_sib'] = df.apply(lambda row: is_sib(row['hn1'], row['hnx']), axis=1)
    df = df[~df['is_sib']].copy()
    
    s1_ids = tuple(df['s1'].unique().tolist())
    sx_ids = tuple(df['sx'].unique().tolist())
    if not s1_ids:
        return
        
    s1_ids_sql = str(s1_ids) if len(s1_ids) > 1 else str(s1_ids).replace(',', '')
    sx_ids_sql = str(sx_ids) if len(sx_ids) > 1 else str(sx_ids).replace(',', '')
    
    q_s1_raw = f"SELECT entity_id as s1, business_name as n1_raw, business_address as a1_raw FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}' AND entity_id IN {s1_ids_sql}"
    q_s2 = f"SELECT entity_id as sx, business_name as nx_raw, business_address as ax_raw FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' AND entity_id IN {sx_ids_sql}"
    q_s3 = f"SELECT entity_id as sx, business_name as nx_raw, business_address as ax_raw FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}' AND entity_id IN {sx_ids_sql}"
    
    s1_raw = con.execute(q_s1_raw).df()
    sx_raw2 = con.execute(q_s2).df()
    sx_raw3 = con.execute(q_s3).df()
    sx_raw = pd.concat([sx_raw2, sx_raw3])
    
    df = df.merge(s1_raw, on='s1', how='left')
    df = df.merge(sx_raw, on='sx', how='left')
    
    df['street1'] = street_key(df['a1_raw'])
    df['streetx'] = street_key(df['ax_raw'])
    df = df[df['street1'] != df['streetx']].copy()
    
    print(f"Total holdout candidate pairs (pred==False): {len(df)}")
    
    df['name_raw_eq'] = df['n1_raw'].str.lower() == df['nx_raw'].str.lower()
    core_counts = con.execute(f"SELECT core, COUNT(*) as k FROM '{ROOT}/data/work/rf_train_s1_{country}.parquet' GROUP BY core").df()
    df = df.merge(core_counts, left_on='core1', right_on='core', how='left')
    
    gt_all_for_s1_query = f"""
    SELECT g.s1, g.sx, s1.hn as hn1, sx.hn as hnx, s1_raw.business_address as a1_raw, sx_raw.business_address as ax_raw
    FROM '{ROOT}/data/work/gt_pairs.parquet' g
    JOIN '{ROOT}/data/work/rf_train_s1_{country}.parquet' s1 ON g.s1 = s1.id
    JOIN '{ROOT}/data/work/rf_train_sx_{country}.parquet' sx ON g.sx = sx.id
    JOIN (SELECT entity_id, business_address FROM '{ROOT}/data/work/train_source1.parquet' WHERE country='{country}' AND entity_id IN {s1_ids_sql}) s1_raw ON s1.id = s1_raw.entity_id
    JOIN (SELECT entity_id, business_address FROM '{ROOT}/data/work/train_source2.parquet' WHERE country='{country}' UNION ALL SELECT entity_id, business_address FROM '{ROOT}/data/work/train_source3.parquet' WHERE country='{country}') sx_raw ON sx.id = sx_raw.entity_id
    WHERE g.s1 IN {s1_ids_sql}
    """
    gt_s1 = con.execute(gt_all_for_s1_query).df()
    gt_s1['street1'] = street_key(gt_s1['a1_raw'])
    gt_s1['streetx'] = street_key(gt_s1['ax_raw'])
    gt_s1['same_addr'] = (gt_s1['hn1'] == gt_s1['hnx']) & (gt_s1['street1'] == gt_s1['streetx'])
    s1_has_same_addr = set(gt_s1[gt_s1['same_addr']]['s1'].unique())
    df['s1_has_same_addr'] = df['s1'].isin(s1_has_same_addr)
    
    df['sx_source'] = df['sx'].apply(lambda x: 'S2' if x.startswith('S2') else 'S3')
    
    df['p_band'] = pd.cut(df['p'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0])
    
    print("--- By p_band overall ---")
    print(df.groupby('p_band')['y'].agg(['count', 'mean']))
    
    print("\n--- By p_band for name_raw_eq == True ---")
    print(df[df['name_raw_eq'] == True].groupby('p_band')['y'].agg(['count', 'mean']))
    
    print("\n--- By p_band for name_raw_eq == True AND k == 1 ---")
    print(df[(df['name_raw_eq'] == True) & (df['k'] == 1)].groupby('p_band')['y'].agg(['count', 'mean']))
    
    print("\n--- By p_band for name_raw_eq == True AND k == 1 AND NOT s1_has_same_addr ---")
    print(df[(df['name_raw_eq'] == True) & (df['k'] == 1) & (~df['s1_has_same_addr'])].groupby('p_band')['y'].agg(['count', 'mean']))
    
    print("\n--- By p_band for am == True AND name_raw_eq == True AND k == 1 AND NOT s1_has_same_addr ---")
    print(df[(df['am'] == True) & (df['name_raw_eq'] == True) & (df['k'] == 1) & (~df['s1_has_same_addr'])].groupby('p_band')['y'].agg(['count', 'mean']))

if __name__ == '__main__':
    run_task2('India')
    run_task2('US')
