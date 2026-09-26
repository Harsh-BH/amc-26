import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii
import pyarrow.parquet as pq

print("Loading...")
d = pq.read_table(ROOT / "data/work/p31/pairs_France.parquet", columns=['hn_cls', 'name_1', 'name_x', 'argmax']).to_pandas()
am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
d = d[am].copy()
d['hn_cls_str'] = d['hn_cls'].astype(str)

print("Vectorized fold...")
# Vectorized fold using pandas string methods
n1 = d['name_1'].str.lower().str.replace(r'[^\w\s\&\+]', ' ', regex=True).fillna("")
nx = d['name_x'].str.lower().str.replace(r'[^\w\s\&\+]', ' ', regex=True).fillna("")
# But wait, anyascii folds accents. e.g. é -> e. pandas string replace doesn't.
# We can just use anyascii on the unique strings to save time!
u1 = pd.unique(d['name_1'])
ux = pd.unique(d['name_x'])
u_all = np.unique(np.concatenate([u1, ux]))
u_map = {s: (anyascii(s).lower() if isinstance(s, str) else "") for s in u_all}
n1 = d['name_1'].map(u_map)
nx = d['name_x'].map(u_map)

print("Tokenizing...")
import re
def get_diff(a, b):
    t1 = set(x for x in re.split(r'[\s]+', a) if x)
    tx = set(x for x in re.split(r'[\s]+', b) if x)
    gained = " ".join(sorted(tx - t1))
    dropped = " ".join(sorted(t1 - tx))
    return gained, dropped

diffs = [get_diff(a, b) for a, b in zip(n1, nx)]
d['gained'] = [x[0] for x in diffs]
d['dropped'] = [x[1] for x in diffs]

# compute stats
d_hn = d[d['hn_cls_str'].isin(['0', '1', '2', '3'])].copy()
grouped = d_hn.groupby('gained')
d_hn_drop1 = d_hn[d_hn["dropped"].str.split().str.len() == 1]
stats = d_hn_drop1.groupby("gained").agg(
    n_hn=('hn_cls_str', 'count'),
    n_sib=('hn_cls_str', lambda x: (x == '1').sum()),
    n_eq=('hn_cls_str', lambda x: (x == '0').sum())
)
stats['s'] = stats['n_sib'] / stats['n_hn']
stats['e'] = stats['n_eq'] / stats['n_hn']

print(stats[stats['n_hn'] >= 20].sort_values('n_hn', ascending=False).head(40).to_string())

