import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/N48-fr-filler-tier")]
from p48 import get_diffs_vec, fold_series, lens_table

W = ROOT / "data/work"
FR = W / "p31/pairs_France.parquet"

def check_overlap():
    d = pd.read_parquet(FR, columns=['s1', 'sx', 'p', 'argmax', 'acc', 'hn_cls', 'name_1', 'name_x', 'addr_1', 'addr_x', 'city_eq', 'hn_1', 'hn_x'])
    am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
    d_am = d[am].copy()
    
    stats, d_am = lens_table(d_am)
    
    strict_allowed = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.05) & (stats['e'] >= 0.8)].index) - {""}
    loose_allowed = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.25) & (stats['e'] >= 0.75)].index) - {""}
    
    d['acc_bool'] = d['acc'] == 'True' if d['acc'].dtype == 'O' else d['acc'] == True
    owned_sx = set(d[d['acc_bool']]['sx'])
    s1_has_acc = set(d[d['acc_bool']]['s1'])
    
    # We apply the rule to argmax pairs
    d_am['sx_free'] = ~d_am['sx'].isin(owned_sx)
    
    from anchor import street_key
    k1 = street_key(d_am['addr_1']).fillna("").to_numpy()
    kx = street_key(d_am['addr_x']).fillna("").to_numpy()
    st_x = (k1 != "") & (k1 == kx)
    
    if d_am['city_eq'].dtype == bool:
        city_ok = d_am['city_eq']
    else:
        city_ok = d_am['city_eq'].astype(str).isin(['True', '1.0', '1', 'True '])
        
    hn_eq = d_am['hn_cls'].astype(str) == '0'
    same_addr = st_x & city_ok & hn_eq
    
    is_drop1 = d_am['dropped'].str.split().str.len() == 1
    
    # Overlap conditions
    # (a) P46: gained in {"developpement", "groupe"}
    a_mask = d_am['gained'].isin({"developpement", "groupe"})
    
    # (b) P42/P37 hn-extra: free argmax non-sibling SX with hn equal to the S1's and p >= 0.80 in non-empty groups
    # Wait, my pairs are already free argmax SX. Are they non-sibling? Yes, hn equal implies hn_cls == '0'. 
    # And p >= 0.80. And S1 non-empty group.
    d_am['p_float'] = d_am['p'].astype(float)
    b_mask = (d_am['p_float'] >= 0.80) & (d_am['s1'].isin(s1_has_acc))
    
    # Now for each variant
    for name, allowed in [("STRICT", strict_allowed), ("LOOSE", loose_allowed)]:
        rule = d_am['sx_free'] & same_addr & is_drop1 & d_am['gained'].isin(allowed)
        adds = d_am[rule]
        n_adds = len(adds)
        
        overlap_a = adds[a_mask].shape[0]
        overlap_b = adds[b_mask].shape[0]
        overlap_ab = adds[a_mask | b_mask].shape[0]
        
        new_pairs = n_adds - overlap_ab
        
        print(f"--- {name} ---")
        print(f"Total adds: {n_adds}")
        print(f"Overlap with P46 (a): {overlap_a}")
        print(f"Overlap with P42 (b): {overlap_b}")
        print(f"Overlap with both (a) and (b): {adds[a_mask & b_mask].shape[0]}")
        print(f"TRULY NEW PAIRS: {new_pairs}")
        
        # Expected LB gain ≈ new_pairs × (2·prec − 1) × ~0.08 / 1.73M
        # We don't have precision on France, but we'll print the formula.

if __name__ == "__main__":
    check_overlap()
