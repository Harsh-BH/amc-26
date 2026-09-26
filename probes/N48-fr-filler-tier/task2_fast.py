import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anchor import street_key
from patterns import name_rel

duckdb.execute("SET memory_limit='1200MB'")

for c in ["US", "India"]:
    print(f"\n--- {c} ---")
    
    # get holdout pairs
    pairs = pd.read_parquet(ROOT / f"data/work/p23/pairs_{c}.parquet", columns=['s1', 'sx', 'holdout', 'y', 'p', 'pred'])
    d = pairs[pairs['holdout'] == True].copy()
    
    # get rf data
    r1 = pd.read_parquet(ROOT / f"data/work/rf_train_s1_{c}.parquet", columns=['id', 'core', 'hn'])
    rx = pd.read_parquet(ROOT / f"data/work/rf_train_sx_{c}.parquet", columns=['id', 'core', 'hn', 'is_web'])
    
    d = d.merge(r1, left_on='s1', right_on='id', how='left').rename(columns={'core': 'c1', 'hn': 'h1'})
    d = d.merge(rx, left_on='sx', right_on='id', how='left').rename(columns={'core': 'cx', 'hn': 'hx'})
    
    # get addresses using duckdb
    s1_ids = tuple(set(d['s1']))
    sx_ids = tuple(set(d['sx']))
    
    q_s1 = f"SELECT entity_id as s1, business_address as a1 FROM '{ROOT}/data/work/train_source1.parquet' WHERE entity_id IN {s1_ids}"
    a1_df = duckdb.query(q_s1).df()
    
    q_s2 = f"SELECT entity_id as sx, business_address as ax FROM '{ROOT}/data/work/train_source2.parquet' WHERE entity_id IN {sx_ids}"
    q_s3 = f"SELECT entity_id as sx, business_address as ax FROM '{ROOT}/data/work/train_source3.parquet' WHERE entity_id IN {sx_ids}"
    ax_df = duckdb.query(f"{q_s2} UNION ALL {q_s3}").df().drop_duplicates('sx')
    
    d = d.merge(a1_df, on='s1', how='left')
    d = d.merge(ax_df, on='sx', how='left')
    
    k1 = street_key(d['a1']).fillna("").to_numpy()
    kx = street_key(d['ax']).fillna("").to_numpy()
    d['st_x'] = (k1 != "") & (k1 == kx)
    
    d['hn_eq'] = (d['h1'] != "") & (d['h1'] == d['hx'])
    d['same_addr'] = d['st_x'] & d['hn_eq']
    
    nr = [name_rel(a, b, w) for a, b, w in zip(d['c1'].fillna(""), d['cx'].fillna(""), d['is_web'].fillna(False))]
    d['kind'] = [k for k, _ in nr]
    d['word'] = [w for _, w in nr]
    
    d_drop1 = d[(d['same_addr']) & (d['kind'] == 'drop1')]
    print(f"drop1 overall (same addr): precision {d_drop1['y'].mean():.4f} ({d_drop1['y'].sum()}/{len(d_drop1)})")
    
    d_drop1_not_acc = d_drop1[d_drop1['pred'] == False]
    print(f"drop1 not accepted: precision {d_drop1_not_acc['y'].mean():.4f} ({d_drop1_not_acc['y'].sum()}/{len(d_drop1_not_acc)})")
    
    print("by p band (not accepted):")
    for lo, hi in [(0, 0.3), (0.3, 0.71), (0.71, 0.95), (0.95, 1.0)]:
        b = d_drop1_not_acc[(d_drop1_not_acc['p'] >= lo) & (d_drop1_not_acc['p'] < hi)]
        if len(b):
            print(f"  band {lo}-{hi}: precision {b['y'].mean():.4f} ({b['y'].sum()}/{len(b)})")

    # Train truth in full (gt_pairs) for "same exact address, S1 word dropped" SX: fraction true for that S1
    # Actually wait. The prompt says: "Also check train truth in full (gt_pairs) for 'same exact address, S1 word dropped' SX: fraction true for that S1."
    # Meaning: among all candidate pairs that are same addr + drop1, what fraction are in gt_pairs? That's what precision measures on the holdout.
    # What does "fraction true for that S1" mean? "fraction true for that S1" probably means: of all S1s that have a "same exact address, S1 word dropped" SX candidate, how often is that candidate the true one? Or how often is it true?
    # Wait, the problem is macro F0.5 per-S1. "fraction true for that S1" could mean: if we find such a pair for an S1, what is the probability it is true?
    
