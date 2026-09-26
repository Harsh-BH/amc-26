import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anchor import street_key
import r05 as R
from anyascii import anyascii

print("loading...")
d = pd.read_parquet(ROOT / "data/work/p31/pairs_France.parquet")

# SX free, argmax, not acc
# pairs_France has `acc`, `argmax`.
am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
ac = d['acc'] == 'True' if d['acc'].dtype == 'O' else d['acc'] == True
h1 = d['hn_cls'].astype(str) != '1'

sx_free = ~d['sx'].isin(set(d.loc[ac, 'sx']))

mask = am & sx_free & h1
d_sub = d[mask].copy()

# identical hn+street+city
print("address matching...")
k1 = street_key(d_sub['addr_1']).fillna("").to_numpy()
kx = street_key(d_sub['addr_x']).fillna("").to_numpy()
d_sub['st_x'] = (k1 != "") & (k1 == kx)

if d_sub['city_eq'].dtype == bool:
    d_sub['city_ok'] = d_sub['city_eq']
else:
    d_sub['city_ok'] = d_sub['city_eq'].astype(str).isin(['True', '1.0', '1', 'True '])

d_sub['hn_eq'] = d['hn_cls'].astype(str)[mask] == '0'

d_ident = d_sub[d_sub['st_x'] & d_sub['city_ok'] & d_sub['hn_eq']].copy()

print("ident size:", len(d_ident))

def normalize_name(names):
    return [anyascii(n).lower() if isinstance(n, str) else "" for n in names]

# wait, we want to just tokenize and do set difference
n1 = normalize_name(d_ident['name_1'])
nx = normalize_name(d_ident['name_x'])

gained_list = []
dropped_list = []
for a, b in zip(n1, nx):
    t1 = set(x for x in re.split(r'[\s]+', a) if x)
    tx = set(x for x in re.split(r'[\s]+', b) if x)
    gained_list.append(" ".join(sorted(tx - t1)))
    dropped_list.append(" ".join(sorted(t1 - tx)))

d_ident['gained'] = gained_list
d_ident['dropped'] = dropped_list

counts = d_ident.groupby(['gained', 'dropped']).size().reset_index(name='n')
counts = counts.sort_values('n', ascending=False)
print("Top relations:")
print(counts.head(40).to_string())
