import sys
sys.path[:0] = ["../../code/business_entity_resolution/src", "../../probes/P28-addr-anchor"]
from anchor import street_key
import pandas as pd

a1 = pd.Series(["10 de l' Argonne", "10 l'Hôpital", "10 l' Hôpital", "10 d' Agen", "10 de l argonne"])
ax = pd.Series(["10 DE LARGONNE", "10 lhopital", "10 lhopital", "10 DAGEN", "10 de largonne"])
print("s1:")
print(street_key(a1).tolist())
print("sx:")
print(street_key(ax).tolist())
