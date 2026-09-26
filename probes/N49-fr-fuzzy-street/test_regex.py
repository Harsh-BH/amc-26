import sys
import pandas as pd
import re
from pathlib import Path

df = pd.read_parquet("../../data/work/p31/pairs_France.parquet", columns=["addr_1", "addr_x", "hn_1", "hn_x", "core_1", "core_x", "city_1", "city_x", "argmax"])
df = df[df.argmax == True]
df = df[(df.core_1 == df.core_x) & (df.core_1 != "")]
df = df[(df.hn_1 == df.hn_x) & (df.hn_1 != "")]
df = df[df.city_1 == df.city_x]

sys.path[:0] = ["../../code/business_entity_resolution/src", "../../probes/P28-addr-anchor"]
from anchor import street_key

s1 = street_key(df.addr_1)
sx = street_key(df.addr_x)
unequal = s1 != sx
df_u = df[unequal].copy()

a1 = df_u.addr_1.str.lower()
ax = df_u.addr_x.str.lower()

pat = r"\b[ld]\s+"
m1 = a1.str.contains(pat, regex=True)
mx = ax.str.contains(pat, regex=True)

diff = df_u[m1 ^ mx]
print("Different in l space vs largonne:", len(diff))
for i, r in diff.head(10).iterrows():
    print(r.addr_1, "|||", r.addr_x)

