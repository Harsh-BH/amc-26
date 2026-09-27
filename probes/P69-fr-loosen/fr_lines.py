"""stdin = a matching_results.tsv; stdout = its France lines (ids from rf_test_s1_France.parquet)."""
import sys
import pandas as pd
fr = set(pd.read_parquet("data/work/rf_test_s1_France.parquet", columns=["id"]).id)
next(sys.stdin)
for ln in sys.stdin:
    if ln.split("\t", 1)[0] in fr:
        sys.stdout.write(ln)
