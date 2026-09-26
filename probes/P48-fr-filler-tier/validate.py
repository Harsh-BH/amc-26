import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii
from anchor import street_key, SIB3

W = ROOT / "data/work"

def get_diffs_vec(n1, nx):
    res_g, res_d = [], []
    for a, b in zip(n1, nx):
        t1 = set(x for x in re.split(r'[\s]+', str(a)) if x)
        tx = set(x for x in re.split(r'[\s]+', str(b)) if x)
        res_g.append(" ".join(sorted(tx - t1)))
        res_d.append(" ".join(sorted(t1 - tx)))
    return pd.Series(res_g, index=n1.index), pd.Series(res_d, index=n1.index)

def fold_series(s):
    u = pd.unique(s)
    u_map = {x: (anyascii(x).lower() if isinstance(x, str) else "") for x in u}
    return s.map(u_map)

def get_raw_data(country, needed_ids=None):
    # Apply AGY memory limit (~1.2GB) to avoid OOM.
    con = duckdb.connect()
    con.execute("SET memory_limit='1200MB'")
    # If a subset of entity IDs is known, filter to those only.
    ids_clause = ""
    if needed_ids is not None:
        ids_list = ", ".join(str(i) for i in needed_ids)
        ids_clause = f" AND entity_id IN ({ids_list})"
    q = []
    for s in ["train_source1.parquet", "train_source2.parquet", "train_source3.parquet"]:
        q.append(f"SELECT entity_id, business_name, business_address FROM '{W}/{s}' WHERE country='{country}'{ids_clause}")
    return con.execute(" UNION ALL ".join(q)).df().set_index("entity_id")

def run(country):
    print(f"Loading {country}...", flush=True)
    d = pd.read_parquet(W / f"p23/pairs_{country}.parquet", columns=["s1", "sx", "p", "y", "am", "pred", "holdout"]).copy()
    # Determine which raw entity IDs are needed for this country
    needed_ids = set(d["s1"]).union(set(d["sx"]))
    raw = get_raw_data(country, needed_ids)
    d["name_1"] = d["s1"].map(raw["business_name"]).fillna("")
    d["name_x"] = d["sx"].map(raw["business_name"]).fillna("")
    d["addr_1"] = d["s1"].map(raw["business_address"]).fillna("")
    d["addr_x"] = d["sx"].map(raw["business_address"]).fillna("")
    del raw
    # Duplicate block removed - using optimized loading above
    
    # load rf features
    r1 = pd.read_parquet(W / f"rf_train_s1_{country}.parquet", columns=["id", "hn", "city"]).set_index("id")
    rx = pd.read_parquet(W / f"rf_train_sx_{country}.parquet", columns=["id", "hn", "city"]).set_index("id")
    
    d['hn_1'] = d['s1'].map(r1['hn']).fillna("")
    d['hn_x'] = d['sx'].map(r1['hn']).fillna("") # wait, sx maps to rx!
    d['hn_x'] = d['sx'].map(rx['hn']).fillna("")
    d['city_1'] = d['s1'].map(r1['city']).fillna("")
    d['city_x'] = d['sx'].map(rx['city']).fillna("")
    del r1, rx
    
    print("Computing string features...", flush=True)
    n1 = fold_series(d['name_1'])
    nx = fold_series(d['name_x'])
    gained, dropped = get_diffs_vec(n1, nx)
    d['gained'] = gained
    d['dropped'] = dropped
    d['is_drop1'] = d['dropped'].str.split().str.len() == 1
    
    k1 = street_key(d['addr_1']).fillna("").to_numpy()
    kx = street_key(d['addr_x']).fillna("").to_numpy()
    d['st_x'] = (k1 != "") & (k1 == kx)
    
    h1 = pd.to_numeric(d['hn_1'], errors='coerce').to_numpy(float)
    hx = pd.to_numeric(d['hn_x'], errors='coerce').to_numpy(float)
    both = (d['hn_1'] != "") & (d['hn_x'] != "")
    dl = np.full(len(h1), np.nan)
    dl[both] = hx[both] - h1[both]
    
    d['is_sib'] = np.isin(dl, SIB3)
    d['is_eq'] = both & (dl == 0)
    d['both_hn'] = both
    
    # city_eq
    d['city_ok'] = (d['city_1'] != "") & (d['city_1'] == d['city_x'])
    d['same_addr'] = d['st_x'] & d['city_ok'] & d['is_eq']
    
    # Lens on argmax
    d_am = d[d['am'] == True]
    d_drop1_am = d_am[d_am['is_drop1']]
    d_hn_am = d_drop1_am[d_drop1_am['both_hn']]
    
    stats = d_hn_am.groupby('gained').agg(
        n_hn=('both_hn', 'sum'),
        n_sib=('is_sib', 'sum'),
        n_eq=('is_eq', 'sum')
    )
    stats['s'] = stats['n_sib'] / stats['n_hn']
    stats['e'] = stats['n_eq'] / stats['n_hn']
    
    strict_phrases = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.05) & (stats['e'] >= 0.8)].index) - {""}
    loose_phrases = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.25) & (stats['e'] >= 0.75)].index) - {""}
    mid_phrases = loose_phrases - strict_phrases # 0.05-0.25
    
    print(f"Strict phrases: {strict_phrases}")
    print(f"Mid phrases: {mid_phrases}")
    
    # Now evaluate on holdout
    h = d[d['holdout'] == True]
    n_holdout_s1 = h['s1'].nunique()
    
    def eval_rule(phrases, name):
        adds = h[(h['pred'] == False) & h['is_drop1'] & h['same_addr'] & h['gained'].isin(phrases)]
        
        # negative bound
        # (# FALSE pairs of the cell on the c holdout, all pairs, accepted or not)
        cell = h[h['is_drop1'] & h['same_addr'] & h['gained'].isin(phrases)]
        false_pairs = (cell['y'] == False).sum()
        neg_per_s1 = false_pairs / n_holdout_s1
        expected_neg_fr = neg_per_s1 * 259452
        
        adds_true = adds['y'].sum()
        adds_total = len(adds)
        adds_prec = adds_true / adds_total if adds_total > 0 else np.nan
        
        print(f"--- {name} ---")
        print(f"Holdout adds: {adds_total} (Prec: {adds_prec:.4f})")
        print(f"Cell total: {len(cell)}, False: {false_pairs}")
        print(f"Neg per S1: {neg_per_s1:.6f}")
        print(f"Expected France negatives: {expected_neg_fr:.1f}")
        return expected_neg_fr
        
    eval_rule(strict_phrases, "STRICT (s<=0.05, e>=0.8)")
    eval_rule(loose_phrases, "LOOSE (s<=0.25, e>=0.75)")
    eval_rule(mid_phrases, "MID (0.05 < s <= 0.25, e>=0.75)")
    
    # by s bin: s <= 0.05 / 0.05–0.15 / 0.15–0.25
    bin1 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.05) & (stats['e'] >= 0.75)].index) - {""}
    bin2 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] > 0.05) & (stats['s'] <= 0.15) & (stats['e'] >= 0.75)].index) - {""}
    bin3 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] > 0.15) & (stats['s'] <= 0.25) & (stats['e'] >= 0.75)].index) - {""}
    
    eval_rule(bin1, "Bin 1: s <= 0.05, e>=0.75")
    eval_rule(bin2, "Bin 2: 0.05 < s <= 0.15, e>=0.75")
    eval_rule(bin3, "Bin 3: 0.15 < s <= 0.25, e>=0.75")
    
    return strict_phrases, loose_phrases

if __name__ == "__main__":
    run("US")
    run("India")
