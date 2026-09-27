"""Generator vocabulary discovery by S1-vs-SX token-frequency asymmetry (label-free, aggregates only).
Tokens far more frequent per record in S2/S3 than in S1 = words the corruptor ADDS (honorifics, generic words, alias
markers, abbreviations, branch words); tokens far rarer = words it drops / abbreviates.
    vocab_asym.py <country> <split> [first]   # 'first' = first-token (prefix) asymmetry of names
"""
import re, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from anyascii import anyascii

ROOT = Path("/home/harsh1/github-repos/amc-26")
c, split = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "all"
TOK = re.compile(r"[a-z0-9@#&+/'.]+")


def load(src, col, cap=800_000):
    t = ds.dataset(ROOT / f"data/work/{split}_{src}.parquet").to_table(columns=[col], filter=ds.field("country") == c)
    s = t.column(col).to_pylist()[:cap]
    return s


def count(strings, first=False):
    cnt, n = Counter(), 0
    for x in strings:
        if not x:
            continue
        n += 1
        f = anyascii(x).lower()
        toks = TOK.findall(f)
        toks = [t.strip(".'") for t in toks]
        toks = [t for t in toks if t]
        if first:
            if toks:
                cnt[toks[0]] += 1
        else:
            cnt.update(set(toks))
    return cnt, n


for col in ("business_name", "business_address"):
    if mode == "first" and col != "business_name":
        continue
    c1, n1 = count(load("source1", col), mode == "first")
    cx, nx = Counter(), 0
    for src in ("source2", "source3"):
        k, n = count(load(src, col), mode == "first")
        cx.update(k); nx += n
    rows = []
    for t in set(c1) | set(cx):
        f1, fx = c1[t] / n1, cx[t] / nx
        if max(c1[t], cx[t]) < 200:
            continue
        rows.append((t, f1, fx, (fx + 1e-5) / (f1 + 1e-5)))
    print(f"== {c} {split} {col} {mode}: n1 {n1:,} nx {nx:,}")
    rows.sort(key=lambda r: -r[3])
    print(" SX-over (added by corruptor):", ", ".join(f"{t} {fx*100:.2f}%/{f1*100:.2f}%" for t, f1, fx, r in rows[:45]))
    rows.sort(key=lambda r: r[3])
    print(" S1-over (dropped/abbreviated):", ", ".join(f"{t} {f1*100:.2f}%/{fx*100:.2f}%" for t, f1, fx, r in rows[:30]))
