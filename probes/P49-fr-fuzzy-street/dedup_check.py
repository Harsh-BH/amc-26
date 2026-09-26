import sys, os
import pandas as pd, numpy as np

W = pd.Series(["data/work"]).iloc[0]

def _get_base_s1_sizes():
    # p31 base sizes (number of accepted SX per S1)
    d = pd.read_parquet(f"{W}/p31/pairs_France.parquet", columns=["s1", "acc"])
    return d[d.acc].groupby("s1").size()

def check_dedup(exact_mask_path, fuzzy_mask_path):
    pass
