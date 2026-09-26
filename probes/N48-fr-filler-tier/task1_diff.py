import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii
import re

d_ident = pd.read_parquet("d_ident.parquet") if Path("d_ident.parquet").exists() else None
if d_ident is None:
    # re-create d_ident
    d = pd.read_parquet(ROOT / "data/work/p31/pairs_France.parquet")
    am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
    ac = d['acc'] == 'True' if d['acc'].dtype == 'O' else d['acc'] == True
    h1 = d['hn_cls'].astype(str) != '1'
    sx_free = ~d['sx'].isin(set(d.loc[ac, 'sx']))
    mask = am & sx_free & h1
    d_sub = d[mask].copy()
    from anchor import street_key
    k1 = street_key(d_sub['addr_1']).fillna("").to_numpy()
    kx = street_key(d_sub['addr_x']).fillna("").to_numpy()
    d_sub['st_x'] = (k1 != "") & (k1 == kx)
    d_sub['city_ok'] = d_sub['city_eq'].astype(str).isin(['True', '1.0', '1', 'True '])
    if d_sub['city_eq'].dtype == bool: d_sub['city_ok'] = d_sub['city_eq']
    d_sub['hn_eq'] = d['hn_cls'].astype(str)[mask] == '0'
    d_ident = d_sub[d_sub['st_x'] & d_sub['city_ok'] & d_sub['hn_eq']].copy()
    d_ident.to_parquet("d_ident.parquet")

def fold(s):
    if not isinstance(s, str): return ""
    return anyascii(s).lower()

import normalize as N

def get_diffs(row):
    # Use name_core so we don't pick up legal form differences if they exist?
    # "name relation = drop-one-word / swap-one-word / add-filler-phrase"
    # Let's just look at core_1 and core_x
    c1, cx = row['core_1'], row['core_x']
    if not isinstance(c1, str): c1 = ""
    if not isinstance(cx, str): cx = ""
    # wait, the prompt says raw, lowercased, accents folded...
    # maybe we should fold raw name?
    n1 = fold(row['name_1'])
    nx = fold(row['name_x'])
    t1 = set(x for x in re.split(r'[\s]+', n1) if x)
    tx = set(x for x in re.split(r'[\s]+', nx) if x)
    # let's just do list difference to preserve some sort of order if needed, or sorted
    gained = " ".join(sorted(tx - t1))
    dropped = " ".join(sorted(t1 - tx))
    return pd.Series([gained, dropped])

d_ident[['gained', 'dropped']] = d_ident.apply(get_diffs, axis=1)
counts = d_ident.groupby(['gained', 'dropped']).size().reset_index(name='n')
counts = counts.sort_values('n', ascending=False)
print(counts.head(40).to_string())

