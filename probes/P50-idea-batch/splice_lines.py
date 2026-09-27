"""P50: output/<OUT> = output/<BASE> with the lines of COUNTRIES taken from output/<SRC> (any row order); every other
line byte-identical; candidate_pairs.tsv / code.zip hard links from BASE.
    splice_lines.py OUT BASE SRC India,US"""
import os, sys
from pathlib import Path
import pandas as pd
out, base, src, cs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
O = Path("output")
s1 = pd.read_parquet("data/work/test_source1.parquet", columns=["entity_id", "country"])
ids = set(s1.entity_id[s1.country.isin(cs)])
del s1
with open(O / src / "matching_results.tsv", encoding="utf-8") as f:
    next(f)
    take = {k: line for k, line in ((line.split("\t", 1)[0], line) for line in f) if k in ids}
assert len(take) == len(ids), (len(take), len(ids))
(O / out).mkdir(parents=True, exist_ok=True)
n = 0
with open(O / base / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
        open(O / out / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for line in fi:
        k = line.split("\t", 1)[0]
        if k in take:
            line, n = take[k], n + 1
        fo.write(line)
assert n == len(ids)
for name in ("candidate_pairs.tsv", "code.zip"):
    (O / out / name).unlink(missing_ok=True)
    if (O / base / name).exists():
        os.link(O / base / name, O / out / name)
print(f"{out}: {n:,} {'/'.join(cs)} lines from {src}, the rest from {base}", flush=True)
