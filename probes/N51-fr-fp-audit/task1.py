import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns")]
from stack import hn_cls
from patterns import name_rel

W = ROOT / "data/work"

def get_feats(split, c, s1_arr, sx_arr):
    import pyarrow.parquet as pq
    r1 = pq.read_table(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "hn"]).to_pandas().set_index("id")
    rx = pq.read_table(W / f"rf_{split}_sx_{c}.parquet", columns=["id", "core", "hn", "is_web"]).to_pandas().set_index("id")
    c1 = r1.core.reindex(s1_arr).fillna("").to_numpy()
    h1 = r1.hn.reindex(s1_arr).fillna("").to_numpy()
    cx = rx.core.reindex(sx_arr).fillna("").to_numpy()
    hx = rx.hn.reindex(sx_arr).fillna("").to_numpy()
    wx = rx.is_web.reindex(sx_arr).fillna(False).to_numpy()
    return c1, h1, cx, hx, wx

def build_lens(c1, h1, cx, hx, wx):
    print("Building lens...")
    kw = [name_rel(a, b, x) for a, b, x in zip(c1, cx, wx)]
    kinds = np.array([k for k, w in kw])
    words = np.array([w for k, w in kw])
    
    cls = hn_cls(np.array(h1, object), np.array(hx, object))
    both_hn = (h1 != "") & (hx != "")
    
    df = pd.DataFrame({'w': words, 'sib': (cls == 1) & both_hn, 'eq': (cls == 0) & both_hn, 'both': both_hn})
    df_words = df[df.w != ""]
    g = df_words.groupby('w').sum()
    g['s'] = g['sib'] / g['both'].clip(lower=1)
    g['e'] = g['eq'] / g['both'].clip(lower=1)
    g = g[g['both'] >= 20]
    return g, kinds, words, cls, both_hn

def scan(country, split):
    print(f"\n--- {country} ---")
    if country == "France":
        d = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "argmax", "acc", "city_eq", "name_rel", "hn_cls"])
        am_mask = d.argmax == True
        acc_mask = d.acc == True
    else:
        d = pd.read_parquet(W / f"p23/pairs_{country}.parquet", columns=["s1", "sx", "holdout", "am", "pred", "y"])
        d = d[d.holdout == True].reset_index(drop=True)
        am_mask = d.am == True
        acc_mask = d.pred == True

    d_am = d[am_mask].reset_index(drop=True)
    c1, h1, cx, hx, wx = get_feats("test" if country == "France" else "train", country, d_am.s1, d_am.sx)
    
    lens_df, kinds, words, cls, both_hn = build_lens(c1, h1, cx, hx, wx)
    
    def assign_band(w):
        if w == "": return "no_gained"
        if w not in lens_df.index: return "low_support"
        s = lens_df.loc[w, 's']
        if s <= 0.05: return "<=0.05"
        if s <= 0.25: return "0.05-0.25"
        if s <= 0.5: return "0.25-0.5"
        return ">0.5"
    
    bands = np.array([assign_band(w) for w in words])
    
    address_less = ~both_hn

    hc_str = np.full(len(cls), "other", dtype=object)
    hc_str[cls == 0] = "equal"
    hc_str[cls == 1] = "sibling"
    hc_str[~both_hn] = "missing"
    
    d_am['kind'] = kinds
    d_am['hc'] = hc_str
    d_am['band'] = bands
    d_am['addr_less'] = address_less
    d_am['acc'] = acc_mask[am_mask].values
    if country != "France":
        d_am['y'] = d.y[am_mask].values
        
    n_s1 = d.s1.nunique() if country != "France" else 259452
    print(f"Total S1: {n_s1}")
    
    if country != "France":
        grouped = d_am.groupby(['kind', 'hc', 'band', 'addr_less']).agg(
            n_all=('s1', 'size'),
            n_acc=('acc', 'sum'),
            n_true=('y', 'sum'),
            n_false=('y', lambda x: (~x).sum())
        )
        grouped['acc_per_s1'] = grouped['n_acc'] / n_s1
        grouped['cell_prec'] = grouped['n_true'] / grouped['n_all']
        grouped['neg_per_s1'] = grouped['n_false'] / n_s1
    else:
        grouped = d_am.groupby(['kind', 'hc', 'band', 'addr_less']).agg(
            n_all=('s1', 'size'),
            n_acc=('acc', 'sum')
        )
        grouped['acc_per_s1'] = grouped['n_acc'] / n_s1
        
    return grouped

fr = scan("France", "test")
us = scan("US", "train")
ind = scan("India", "train")

idx = fr.index.union(us.index).union(ind.index)

df = pd.DataFrame(index=idx)
df['fr_acc'] = fr['n_acc']
df['fr_acc_per_s1'] = fr['acc_per_s1']

for c, d in [('us', us), ('ind', ind)]:
    df[f'{c}_acc'] = d['n_acc']
    df[f'{c}_acc_per_s1'] = d['acc_per_s1']
    df[f'{c}_cell_prec'] = d['cell_prec']
    df[f'{c}_neg_per_s1'] = d['neg_per_s1']

df = df.fillna(0)

# Re-fill precisions to 1.0 where there are no pairs


df.loc[df['us_cell_prec'] == 0, 'us_cell_prec'] = 1.0  # maybe 0 means no true pairs, which is precision 0! wait. 0 precision is valid.
# Wait, if n_all was 0, cell_prec is NaN, which fillna(0) made 0. We should make it 1.0.
# So:
df['us_cell_prec'] = us['cell_prec'].fillna(1.0)
df['ind_cell_prec'] = ind['cell_prec'].fillna(1.0)

df['us_ind_max_acc_per_s1'] = df[['us_acc_per_s1', 'ind_acc_per_s1']].max(axis=1)
df['us_ind_min_cell_prec'] = df[['us_cell_prec', 'ind_cell_prec']].min(axis=1)
df['excess'] = df['fr_acc_per_s1'] - df['us_ind_max_acc_per_s1']

suspects = df[(df['excess'] > 0.001) & (df['us_ind_min_cell_prec'] < 0.7)]
print("\n--- SUSPECTS ---")
suspects_print = suspects.reset_index().sort_values('excess', ascending=False)
print(suspects_print.to_string())

df.to_csv("probes/N51-fr-fp-audit/cells.csv")
print("Saved all cells to cells.csv")
