import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii
from anchor import street_key

def fold(s): return anyascii(s).lower() if isinstance(s, str) else ""

fillers = {"&", "+", "et", "fils", "cie", "associes", "freres", "frs", "developpement", "groupe", "compagnie", "(france)"}
legals = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei"} # US/India analogues: LLC, INC, LTD, PVT, CO, CORP, etc.
# Wait, for US/India analogues, what are the fillers/legals?
# "filler/legal token". I can just look at `add1` or `drop1` or any gained token that is in a small set, or just use the same logic: pure drop-one-word or drop-one-word + any gained token.
# Let's just look at the exact difference: dropped == 1 word, and gained tokens are ALL in some set.
# BUT wait! The prompt doesn't ask to constrain the gained token for US/India, it says: "same-address + drop-one-word (+ any gained filler/legal token)". 
# Actually, if I just measure "same address + drop exactly one word + any gained token", that might be too broad.
# Let's use `name_rel` from P20 for US and India! P20 `name_rel` has "drop1" (one dropped, 0 gained) and "sub1" (one dropped, 1 gained). But wait, P20's `name_rel` normalizes legal words out using `normalize.name_core`!
# Ah! `normalize.name_core` ALREADY strips legal words like LLC, INC in US, and Pvt Ltd in India!
# So if we use `name_rel` on `name_core`, "drop1" means exactly one core word dropped, and ANY legal words gained/dropped are already ignored!
# Let's check `name_rel` == 'drop1' and `name_rel` == 'sub1' where gained word is filler.

for c in ["US", "India"]:
    print(f"\n--- {c} ---")
    d = pd.read_parquet(ROOT / f"data/work/p23/pairs_{c}.parquet")
    d = d[d['holdout'] == True].copy()
    
    # same-address:
    # use rf_train_s1/sx for hn, city, places, and anchor.street_key
    r1 = pd.read_parquet(ROOT / f"data/work/rf_train_s1_{c}.parquet", columns=['id', 'core', 'hn', 'city'])
    rx = pd.read_parquet(ROOT / f"data/work/rf_train_sx_{c}.parquet", columns=['id', 'core', 'hn', 'city', 'is_web'])
    
    # need addresses for street_key
    s1_addr = pd.read_parquet(ROOT / f"data/work/train_source1.parquet", columns=['entity_id', 'business_address'])
    sx2 = pd.read_parquet(ROOT / f"data/work/train_source2.parquet", columns=['entity_id', 'business_address'])
    sx3 = pd.read_parquet(ROOT / f"data/work/train_source3.parquet", columns=['entity_id', 'business_address'])
    sx_addr = pd.concat([sx2, sx3]).drop_duplicates('entity_id')
    
    d = d.merge(r1, left_on='s1', right_on='id', how='left').rename(columns={'core': 'c1', 'hn': 'h1', 'city': 'city1'})
    d = d.merge(rx, left_on='sx', right_on='id', how='left').rename(columns={'core': 'cx', 'hn': 'hx', 'city': 'cityx'})
    
    d = d.merge(s1_addr, left_on='s1', right_on='entity_id', how='left').rename(columns={'business_address': 'a1'})
    d = d.merge(sx_addr, left_on='sx', right_on='entity_id', how='left').rename(columns={'business_address': 'ax'})
    
    k1 = street_key(d['a1']).fillna("").to_numpy()
    kx = street_key(d['ax']).fillna("").to_numpy()
    d['st_x'] = (k1 != "") & (k1 == kx)
    
    # hn equal
    d['hn_eq'] = (d['h1'] != "") & (d['h1'] == d['hx'])
    
    # city overlap? actually let's just use city1 == cityx or something?
    # the prompt says "same-address" -> let's use hn_eq & st_x. P28 also uses city. 
    # For US/India, let's just use st_x & hn_eq as same address.
    d['same_addr'] = d['st_x'] & d['hn_eq']
    
    from patterns import name_rel
    nr = [name_rel(a, b, w) for a, b, w in zip(d['c1'].fillna(""), d['cx'].fillna(""), d['is_web'].fillna(False))]
    d['kind'] = [k for k, _ in nr]
    d['word'] = [w for _, w in nr]
    
    # drop-one-word (+ any gained filler/legal token)
    # Since core ignores legals, kind == 'drop1' means exactly one core word dropped, 0 core words gained (legal differences ignored).
    # What if a filler was gained? In P20, fillers are NOT ignored in core, they are in the core.
    # So if 'cie' is gained, kind is 'sub1' if 1 dropped, or 'add1' if 0 dropped.
    
    d_drop1 = d[(d['same_addr']) & (d['kind'] == 'drop1')]
    print("drop1 overall:", d_drop1['y'].mean(), f"({d_drop1['y'].sum()}/{len(d_drop1)})")
    
    d_drop1_not_acc = d_drop1[d_drop1['pred'] == False]
    print("drop1 not accepted:", d_drop1_not_acc['y'].mean(), f"({d_drop1_not_acc['y'].sum()}/{len(d_drop1_not_acc)})")
    
    # by p band
    for lo, hi in [(0, 0.3), (0.3, 0.71), (0.71, 0.95), (0.95, 1.0)]:
        b = d_drop1[(d_drop1['p'] >= lo) & (d_drop1['p'] < hi)]
        if len(b):
            print(f"  band {lo}-{hi}: {b['y'].mean():.3f} ({len(b)})")
