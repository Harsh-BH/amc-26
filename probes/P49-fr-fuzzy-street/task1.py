import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("../../")
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P28-addr-anchor")]
import normalize as N
from anchor import street_key

df = pd.read_parquet("../../data/work/p31/pairs_France.parquet")
df = df[df.argmax == True]
df = df[(df.core_1 == df.core_x) & (df.core_1 != "")]
df = df[(df.hn_1 == df.hn_x) & (df.hn_1 != "")]
# the brief says equal city, let's use city_1 == city_x or city_eq
df = df[df.city_1 == df.city_x]

df["street_1"] = street_key(df.addr_1)
df["street_x"] = street_key(df.addr_x)

df_diff = df[df.street_1 != df.street_x]
print(f"Total such pairs: {len(df)}")
print(f"Unequal normalised street: {len(df_diff)}")

df_diff[["addr_1", "addr_x", "street_1", "street_x"]].to_csv("diff.csv", index=False)
