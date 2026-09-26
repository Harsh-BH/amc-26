import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii

print("loading filtered argmax rows...")
table = pq.read_table(ROOT / "data/work/p31/pairs_France.parquet", 
                      columns=['s1', 'sx', 'hn_cls', 'name_1', 'name_x', 'hn_1', 'hn_x', 'argmax'])
# argmax is boolean or string 'True'. Let's filter in pandas since pyarrow string filter might be tricky if mixed.
d = table.to_pandas()
am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
d_am = d[am].copy()
del d, table
import gc; gc.collect()

d_am['hn_cls_str'] = d_am['hn_cls'].astype(str)

def has_both_hns(row):
    h1 = row['hn_1']
    hx = row['hn_x']
    return bool(h1 and hx and isinstance(h1, str) and isinstance(hx, str) and h1.strip() and hx.strip())

d_am['has_hn'] = d_am.apply(has_both_hns, axis=1)

def fold(s):
    return anyascii(s).lower() if isinstance(s, str) else ""

print("folding...")
n1 = d_am['name_1'].apply(fold)
nx = d_am['name_x'].apply(fold)

gained_list = []
for a, b in zip(n1, nx):
    t1 = set(x for x in re.split(r'[\s]+', a) if x)
    tx = set(x for x in re.split(r'[\s]+', b) if x)
    gained_list.append(" ".join(sorted(tx - t1)))

d_am['gained'] = gained_list

# Compute lens for each phrase
stats = []
phrases = set(d_am['gained'])
for phrase in phrases:
    if not phrase: continue
    subset = d_am[d_am['gained'] == phrase]
    n_all = len(subset)
    subset_hn = subset[subset['has_hn']]
    n_hn = len(subset_hn)
    if n_hn == 0: continue
    s = (subset_hn['hn_cls_str'] == '1').sum() / n_hn
    e = (subset_hn['hn_cls_str'] == '0').sum() / n_hn
    stats.append({
        'gained': phrase,
        'n_all': n_all,
        'n_hn': n_hn,
        's': s,
        'e': e
    })

stats_df = pd.DataFrame(stats).set_index('gained')

# Join with d_ident counts
d_ident = pd.read_parquet("d_ident.parquet", columns=['name_1', 'name_x'])
def get_diffs(row):
    a = fold(row['name_1'])
    b = fold(row['name_x'])
    t1 = set(x for x in re.split(r'[\s]+', a) if x)
    tx = set(x for x in re.split(r'[\s]+', b) if x)
    gained = " ".join(sorted(tx - t1))
    dropped = " ".join(sorted(t1 - tx))
    return pd.Series([gained, dropped])

d_ident[['gained', 'dropped']] = d_ident.apply(get_diffs, axis=1)
counts = d_ident.groupby(['gained', 'dropped']).size().reset_index(name='n')
counts = counts.sort_values('n', ascending=False)
counts = counts[counts['gained'] != '']

res = counts.join(stats_df, on='gained', how='inner')
print("Top lens stats:")
print(res[['gained', 'dropped', 'n', 'n_hn', 's', 'e']].head(40).to_string())

