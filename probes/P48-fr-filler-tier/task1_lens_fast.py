import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
import pyarrow.parquet as pq
from anyascii import anyascii

# We only need to check stats for the top gained phrases:
top_phrases = ["developpement", "groupe", "et fils", "cie", "sci", "& associes", "& fils", "sas", "sarl", "compagnie", "eurl", "sasu"]

print("reading pairs...")
table = pq.read_table(ROOT / "data/work/p31/pairs_France.parquet", columns=['hn_cls', 'name_1', 'name_x', 'hn_1', 'hn_x', 'argmax'])
d = table.to_pandas()
am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
d = d[am].copy()
d['hn_cls_str'] = d['hn_cls'].astype(str)

def has_both_hns(row):
    h1, hx = row['hn_1'], row['hn_x']
    return bool(h1 and hx and isinstance(h1, str) and isinstance(hx, str) and h1.strip() and hx.strip())

d['has_hn'] = d.apply(has_both_hns, axis=1)
d_hn = d[d['has_hn']].copy()

print("filtering for phrases...")
# A fast way to find if a phrase is gained: check if it's in name_x but not in name_1
# Then accurately fold and check
stats = []
for phrase in top_phrases:
    # fast pre-filter
    phrase_clean = phrase.replace('&', '').strip()
    # just fold all names? No, too slow.
    # actually we only need to fold names for pairs where the word might be present
    # but wait, let's just fold the whole column using a vectorized/faster approach.
    pass

def fold(s):
    return anyascii(s).lower() if isinstance(s, str) else ""

# apply is slow. let's just use it on a sample? No, we need it on all.
# Is anyascii really that slow? Let's write a multiprocessing or just use it.
