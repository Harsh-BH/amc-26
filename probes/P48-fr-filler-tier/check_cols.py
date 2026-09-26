import sys
from pathlib import Path
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"

con = duckdb.connect()
print("train_source1 schema:", con.execute(f"DESCRIBE SELECT * FROM '{W}/train_source1.parquet'").df())
print("rf_train_s1_US schema:", con.execute(f"DESCRIBE SELECT * FROM '{W}/rf_train_s1_US.parquet'").df())

