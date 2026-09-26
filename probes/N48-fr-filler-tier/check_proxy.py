import pandas as pd
import duckdb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT / "data/work/p31/pairs_France.parquet")
print(df.columns)
