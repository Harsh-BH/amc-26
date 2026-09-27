"""P50: output/<OUT> = output/<BASE> minus the (s1, sx) pairs in PAIRS.parquet (France or any country); every other line
byte-identical; candidate_pairs.tsv / code.zip hard links. Reports pairs present / absent and S1 made empty.
    rm_pairs.py BASE OUT PAIRS.parquet"""
import os, sys
from pathlib import Path
import pandas as pd
base, out, pp = sys.argv[1:4]
O = Path("output")
d = pd.read_parquet(pp, columns=["s1", "sx"])
minus = d.groupby("s1").sx.agg(set).to_dict()
(O / out).mkdir(parents=True, exist_ok=True)
n_rm = n_chg = n_empty = 0
with open(O / base / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
        open(O / out / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for line in fi:
        k, v = line.rstrip("\n").split("\t", 1)
        if k in minus and v:
            cur = set(v.split(","))
            new = cur - minus[k]
            if new != cur:
                n_rm += len(cur) - len(new)
                n_chg += 1
                n_empty += not new
                fo.write(f"{k}\t{','.join(sorted(new))}\n")
                continue
        fo.write(line)
for name in ("candidate_pairs.tsv", "code.zip"):
    (O / out / name).unlink(missing_ok=True)
    if (O / base / name).exists():
        os.link(O / base / name, O / out / name)
print(f"{out}: removed {n_rm:,} of {len(d):,} listed pairs ({len(d) - n_rm:,} not in {base}); {n_chg:,} S1 changed, {n_empty:,} made empty", flush=True)
