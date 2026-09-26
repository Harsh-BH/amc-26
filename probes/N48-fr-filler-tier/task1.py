import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anchor import street_key
import r05 as R

d = pd.read_parquet(ROOT / "data/work/p31/pairs_France.parquet")

argmax_mask = d['argmax'] == 'True' if d['argmax'].dtype == 'O' or d['argmax'].dtype.name == 'string' else d['argmax'] == True
acc_mask = d['acc'] == 'True' if d['acc'].dtype == 'O' or d['acc'].dtype.name == 'string' else d['acc'] == True
hn_cls_str = d['hn_cls'].astype(str)
hn_cls_mask = hn_cls_str != '1'
acc_sx = set(d.loc[acc_mask, 'sx'])
sx_free = ~d['sx'].isin(acc_sx)

mask = argmax_mask & sx_free & hn_cls_mask
d_sub = d[mask].copy()

k1 = street_key(d_sub['addr_1']).fillna("").to_numpy()
kx = street_key(d_sub['addr_x']).fillna("").to_numpy()
d_sub['st_x'] = (k1 != "") & (k1 == kx)

city_eq_str = d_sub['city_eq'].astype(str)
d_sub['city_ok'] = city_eq_str.isin(['True', '1.0', '1', 'True '])
if d_sub['city_eq'].dtype == bool:
    d_sub['city_ok'] = d_sub['city_eq']

d_sub['hn_eq'] = hn_cls_str[mask] == '0'

d_sub['identical_addr'] = d_sub['st_x'] & d_sub['city_ok'] & d_sub['hn_eq']
d_ident = d_sub[d_sub['identical_addr']].copy()

print("Total rows:", len(d))
print("Filtered rows:", len(d_sub))
print("Identical addr rows:", len(d_ident))
print(d_ident['name_rel'].value_counts())

