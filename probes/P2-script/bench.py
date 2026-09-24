"""Full-scale cost of to_latin + fold on the test set, one source at a time.

    .venv/bin/python probes/P2-script/bench.py
"""
import sys, time, resource
sys.path[:0] = ["probes/P2-script"]
import pandas as pd
from translit import to_latin, fold, table, INDIC

rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
table()
for src in ("test_source2", "test_source3"):
    df = pd.read_parquet(f"data/work/{src}.parquet", columns=["business_name", "business_address"])
    for col in ("business_name", "business_address"):
        t = time.time(); r = to_latin(df[col]); t1 = time.time() - t
        n = int(df[col].str.contains(INDIC, regex=True).sum())
        left = int(r.str.contains(INDIC, regex=True).sum())
        sub = r[df[col].str.contains(INDIC, regex=True).to_numpy()]
        t = time.time(); fold(sub); t2 = time.time() - t
        print(f"{src} {col}: {len(df):,} rows, {n:,} Indic, to_latin {t1:.1f}s, fold(Indic rows) {t2:.1f}s, Indic left {left}, peak {rss()} GB")
    del df, r, sub
