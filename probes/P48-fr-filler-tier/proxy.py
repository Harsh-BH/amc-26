import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
import p48

print("running proxy...")
d = pd.read_parquet(ROOT / "data/work/p31/pairs_France.parquet", columns=['s1', 'sx', 'p', 'argmax', 'acc', 'hn_cls', 'name_1', 'name_x', 'addr_1', 'addr_x', 'city_eq'])
am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
d_am = d[am].copy()
stats, d_am = p48.lens_table(d_am)
allowed = set(stats[(stats['n_hn'] >= 20) & (stats['s'] <= p48.S_MAX) & (stats['e'] >= p48.E_MIN)].index) - {""}

acc = d_am['acc'] == 'True' if d_am['acc'].dtype == 'O' else d_am['acc'] == True
owned = set(d_am.loc[acc, 'sx'])
sx_free = ~d_am['sx'].isin(owned)

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
has_allowed_gained = d_am['gained'].isin(allowed)

rule = sx_free & same_addr & is_drop1 & has_allowed_gained
flips = d_am[rule].copy()

acc_pairs = d_am[acc]
s1_owned = acc_pairs.groupby('s1')['sx'].apply(list).to_dict()

added = 0
p46_overlap = 0
for s1, sx, gained in flips.sort_values('p', ascending=False)[['s1', 'sx', 'gained']].itertuples(index=False):
    if sx in owned: continue
    cur = s1_owned.setdefault(s1, [])
    if sum(x.startswith(sx[:2]) for x in cur) >= p48.CAP.get(sx[:2], 5): continue
    cur.append(sx)
    owned.add(sx)
    added += 1
    if gained in {"developpement", "groupe"}:
        p46_overlap += 1

print(f"Added on P31 proxy base: {added}")
print(f"P46 overlap: {p46_overlap}")
