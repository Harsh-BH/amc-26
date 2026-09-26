import sys
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
duckdb.execute("SET memory_limit='2000MB'")

print("Querying stats with duckdb...")

# identical hn+street+city, SX free, argmax, hn_cls != 1
# city_eq = 'True' or '1.0' or '1'
# st_x: street_key(addr_1) == street_key(addr_x) -> duckdb doesn't have street_key, so we have to do it in python or approximate it.
# Actually, we already have d_ident.parquet from task-37 ! Wait, task-37 only printed the counts, it didn't save d_ident.
