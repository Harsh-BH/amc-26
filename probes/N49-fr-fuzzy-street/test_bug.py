import sys
sys.path[:0] = ["../../code/business_entity_resolution/src", "../../probes/P28-addr-anchor"]
from anchor import street_key
import pandas as pd

a1 = pd.Series(["10 de l'Argonne", "10 d’Agen", "10 Rue", "10 Avenue", "10 BD", "10 Général", "10 ST"])
ax = pd.Series(["10 DE LARGONNE", "10 DAGEN", "10 R.", "10 AV", "10 Boulevard", "10 GEN", "10 Saint"])
print("s1:")
print(street_key(a1))
print("sx:")
print(street_key(ax))
