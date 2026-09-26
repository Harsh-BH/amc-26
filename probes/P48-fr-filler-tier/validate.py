import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]

W = ROOT / "data/work"
def get_raw_data(country):
    con = duckdb.connect()
    q = []
    for s in ["train_source1.parquet", "train_source2.parquet", "train_source3.parquet"]:
        q.append(f"SELECT entity_id, business_name, business_address FROM '{W}/{s}' WHERE country='{country}'")
    return con.execute(" UNION ALL ".join(q)).df().set_index("entity_id")

df = get_raw_data("US")
print(df.head())
