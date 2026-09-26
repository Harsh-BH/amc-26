import sys
from pathlib import Path
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anchor import street_key
from patterns import name_rel

duckdb.execute("SET memory_limit='1200MB'")

for c in ["US", "India"]:
    print(f"\n--- {c} ---")
    
    # We load all pairs from the full blocked list... but it's huge!
    # Instead, let's just query the gt_pairs directly and see how many of them are drop1 + same addr?
    # No, the prompt asks: for a given "same exact address, S1 word dropped" SX, what is the "fraction true for that S1".
    # This means: Precision of the "drop1 + same addr" rule over ALL candidate pairs.
    # To get ALL candidate pairs, we need `p3/cands_train_full_{c}.parquet`
    pass
