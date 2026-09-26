import sys
from pathlib import Path
import pandas as pd
import numpy as np
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
    return res_g, res_d

def fold_series(s):
    u = pd.unique(s)
    u_map = {x: (anyascii(x).lower() if isinstance(x, str) else "") for x in u}
    return s.map(u_map)

def run(country):
    print(f"Loading {country} p23...", flush=True)
    d = pd.read_parquet(W / f"p23/pairs_{country}.parquet", columns=["s1", "sx", "p", "y", "am", "pred", "holdout"])
    d_filt = d[(d['holdout'] == True) | (d['am'] == True)].copy()
    accepted_sx = set(d[d['pred'] == True]['sx'])
    del d
    d_filt['sx_is_free'] = ~d_filt['sx'].isin(accepted_sx)
    ids = set(d_filt['s1']) | set(d_filt['sx'])
    
    print(f"Loading {country} raw...", flush=True)
    dfs = []
    for s in ["train_source1.parquet", "train_source2.parquet", "train_source3.parquet"]:
        df = pd.read_parquet(W / s, columns=["entity_id", "business_name", "business_address", "country"], filters=[('country', '=', country)])
        df = df[df['entity_id'].isin(ids)]
        dfs.append(df)
    
    raw = pd.concat(dfs).set_index("entity_id")
    del dfs
    
    d_filt['name_1'] = d_filt['s1'].map(raw['business_name']).fillna("")
    d_filt['name_x'] = d_filt['sx'].map(raw['business_name']).fillna("")
    d_filt['addr_1'] = d_filt['s1'].map(raw['business_address']).fillna("")
    d_filt['addr_x'] = d_filt['sx'].map(raw['business_address']).fillna("")
    del raw
    
    print(f"Loading {country} features...", flush=True)
    r1 = pd.read_parquet(W / f"rf_train_s1_{country}.parquet", columns=["id", "hn", "city"]).set_index("id")
    rx = pd.read_parquet(W / f"rf_train_sx_{country}.parquet", columns=["id", "hn", "city"]).set_index("id")
    
    d_filt['hn_1'] = d_filt['s1'].map(r1['hn']).fillna("")
    d_filt['hn_x'] = d_filt['sx'].map(rx['hn']).fillna("")
    d_filt['city_1'] = d_filt['s1'].map(r1['city']).fillna("")
    d_filt['city_x'] = d_filt['sx'].map(rx['city']).fillna("")
    del r1, rx
    
    print("Computing string features...", flush=True)
    n1 = fold_series(d_filt['name_1'])
    nx = fold_series(d_filt['name_x'])
    gained, dropped = get_diffs_vec(n1, nx)
    d_filt['gained'] = gained
    d_filt['dropped'] = dropped
    d_filt['is_drop1'] = d_filt['dropped'].str.split().str.len() == 1
    
    k1 = street_key(d_filt['addr_1']).fillna("").to_numpy()
    kx = street_key(d_filt['addr_x']).fillna("").to_numpy()
    d_filt['st_x'] = (k1 != "") & (k1 == kx)
    
    h1 = pd.to_numeric(d_filt['hn_1'], errors='coerce').to_numpy(float)
    hx = pd.to_numeric(d_filt['hn_x'], errors='coerce').to_numpy(float)
    both = (d_filt['hn_1'] != "") & (d_filt['hn_x'] != "")
    dl = np.full(len(h1), np.nan)
    dl[both] = hx[both] - h1[both]
    
    d_filt['is_sib'] = np.isin(dl, SIB3)
    d_filt['is_eq'] = both & (dl == 0)
    d_filt['both_hn'] = both
    
    d_filt['city_ok'] = (d_filt['city_1'] != "") & (d_filt['city_1'] == d_filt['city_x'])
    d_filt['same_addr'] = d_filt['st_x'] & d_filt['city_ok'] & d_filt['is_eq']
    
    print("Evaluating...", flush=True)
    d_am = d_filt[d_filt['am'] == True]
    d_drop1_am = d_am[d_am['is_drop1']]
    d_hn_am = d_drop1_am[d_drop1_am['both_hn']]
    
    stats = d_hn_am.groupby('gained').agg(
        n_hn=('both_hn', 'sum'),
        n_sib=('is_sib', 'sum'),
        n_eq=('is_eq', 'sum')
    )
    stats['s'] = stats['n_sib'] / stats['n_hn']
    stats['e'] = stats['n_eq'] / stats['n_hn']
    
    bin1 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.05) & (stats['e'] >= 0.75)].index) - {""}
    bin2 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] > 0.05) & (stats['s'] <= 0.15) & (stats['e'] >= 0.75)].index) - {""}
    bin3 = set(stats[(stats['n_hn'] >= 20) & (stats['s'] > 0.15) & (stats['s'] <= 0.25) & (stats['e'] >= 0.75)].index) - {""}
    
    h = d_filt[d_filt['holdout'] == True]
    n_holdout_s1 = h['s1'].nunique()
    
    def eval_rule(phrases, name):
        adds = h[(h['pred'] == False) & (h['sx_is_free'] == True) & h['is_drop1'] & h['same_addr'] & h['gained'].isin(phrases)]
        cell = h[h['is_drop1'] & h['same_addr'] & h['gained'].isin(phrases)]
        false_pairs = (cell['y'] == False).sum()
        neg_per_s1 = false_pairs / n_holdout_s1
        expected_neg_fr = neg_per_s1 * 259452
        
        adds_true = adds['y'].sum()
        adds_total = len(adds)
        adds_prec = adds_true / adds_total if adds_total > 0 else np.nan
        
        print(f"\n--- {name} ---")
        print(f"Holdout adds: {adds_total} (Prec: {adds_prec:.4f})")
        print(f"Cell total: {len(cell)}, False: {false_pairs}")
        print(f"Neg per S1: {neg_per_s1:.6f}")
        print(f"Expected France negatives: {expected_neg_fr:.1f}")
        
        if adds_total > 0:
            print("Top phrases in adds:")
            phrases_stats = adds.groupby('gained').agg(
                adds=('y', 'count'),
                true=('y', 'sum')
            ).sort_values('adds', ascending=False).head(10)
            for phrase, row in phrases_stats.iterrows():
                print(f"  {phrase}: {row['adds']} adds (Prec: {row['true']/row['adds']:.4f})")
                
        return expected_neg_fr, adds_prec
        
    eval_rule(bin1, "Bin 1: s <= 0.05, e>=0.75")
    eval_rule(bin2, "Bin 2: 0.05 < s <= 0.15, e>=0.75")
    eval_rule(bin3, "Bin 3: 0.15 < s <= 0.25, e>=0.75")
    
    strict = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.05) & (stats['e'] >= 0.8)].index) - {""}
    loose = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= 0.25) & (stats['e'] >= 0.75)].index) - {""}
    
    eval_rule(strict, "STRICT (s <= 0.05, e>=0.8)")
    eval_rule(loose, "LOOSE (s <= 0.25, e>=0.75)")

if __name__ == "__main__":
    print("=== US ===")
    run("US")
    print("\n=== India ===")
    run("India")
