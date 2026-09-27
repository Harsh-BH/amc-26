"""P50: lines over the 5 S2 / 6 S3 cap per country in each output/<folder> (matching_results.tsv)."""
import sys, collections
import pandas as pd
s1 = pd.read_parquet("data/work/test_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country.to_dict()
for fo in sys.argv[1:]:
    c = collections.Counter()
    with open(f"output/{fo}/matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, v = line.rstrip("\n").split("\t", 1)
            if v:
                x = v.split(",")
                n2 = sum(t.startswith("S2-") for t in x)
                if n2 > 5 or len(x) - n2 > 6:
                    c[s1[a]] += 1
    print(fo, dict(c), flush=True)
