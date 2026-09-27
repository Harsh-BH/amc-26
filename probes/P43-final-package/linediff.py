"""Line diff of two submission dirs by country: python linediff.py A B [file] [sets]. Same row order assumed (asserted).
sets: compare each row's ID set (P40 / P43 builders append new IDs unsorted; run.py sorts them)."""
import sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
rp = lambda x: x if x.startswith(("s3:", "gdrive:")) else Path(x)
a, b = rp(sys.argv[1]), rp(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else "matching_results.tsv"
sets = "sets" in sys.argv[4:]
norm = (lambda l: ",".join(sorted(l.rstrip("\n").split("\t")[1].split(",")))) if sets else (lambda l: l)
t = pq.read_table(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"])
cty = dict(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))
del t
diff, rows, order = {}, {}, 0
def opener(d):
    """A folder, or an rclone remote folder (streamed with rclone cat: the local disk is full)."""
    if str(d).startswith(("s3:", "gdrive:")):
        import os
        import subprocess
        pr = subprocess.Popen(["rclone", "cat", f"{d}/{name}"], stdout=subprocess.PIPE, env={**os.environ, "AWS_PROFILE": "harsh"})
        return __import__("io").TextIOWrapper(pr.stdout, encoding="utf-8")
    return open(d / name, encoding="utf-8")


with opener(a) as f, opener(b) as g:
    assert next(f) == next(g)
    for x, y in zip(f, g):
        s, s2 = x.split("\t", 1)[0], y.split("\t", 1)[0]
        assert s == s2, ("row order differs", s, s2)
        c = cty[s]
        rows[c] = rows.get(c, 0) + 1
        if norm(x) != norm(y):
            diff[c] = diff.get(c, 0) + 1
        order += x != y
    assert next(f, None) is None and next(g, None) is None, "different row counts"
print(f"{str(a).rstrip('/').split('/')[-1]} vs {str(b).rstrip('/').split('/')[-1]} [{name}{', as ID sets' if sets else ''}]: rows {rows}, differing {diff or 0}"
      + (f"; raw lines differing {order:,}" if sets else ""))
