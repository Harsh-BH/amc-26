import pandas as pd
import numpy as np

d = pd.read_parquet("d_ident.parquet") if Path("d_ident.parquet").exists() else None
# Actually we have all the info in memory from task1_exact if I just modified it!
