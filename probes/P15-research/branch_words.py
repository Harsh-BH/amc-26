"""P15 label-free: which single token does an SX core ADD to its argmax S1's core (S1 core tokens a strict subset,
exactly one extra token)? The generator's orphan siblings add a branch word from a short fixed list (P12: US
partners/holdings/group..., India ventures/group/public...), so a flat top of near-equal counts is the list's
fingerprint. Prints token counts only (vocabulary, no rows).
    .venv/bin/python probes/P15-research/branch_words.py France [US]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

W = Path(__file__).resolve().parents[2] / "data/work"


def cores(path, ids):
    rf = pq.read_table(path, columns=["id", "core"])
    pos = pc.index_in(pa.array(ids), value_set=rf.column("id")).to_numpy(zero_copy_only=False).astype(np.int64)
    return rf.column("core").to_numpy(zero_copy_only=False)[pos]


def extra(c1, cx):
    """The one token cx adds to c1 ('' otherwise)."""
    a, b = set(c1.split()), cx.split()
    if not a or len(b) != len(a) + 1 or not a.issubset(b):
        return ""
    e = [t for t in b if t not in a]
    return e[0] if len(e) == 1 else ""


def main(c, pmin=0.5):
    f = W / ("p10/prob_test_France.parquet" if c == "France" else f"p11/prob_test_{c}_blend.parquet")
    d = pd.read_parquet(f, columns=["s1", "sx", "p"])
    d = d[(d.p.to_numpy() == d.groupby("sx").p.transform("max").to_numpy()) & (d.p >= pmin)]
    c1 = cores(W / f"rf_test_s1_{c}.parquet", d.s1.to_numpy())
    cx = cores(W / f"rf_test_sx_{c}.parquet", d.sx.to_numpy())
    ex = [extra(a, b) for a, b in zip(c1, cx)]
    cnt = Counter(t for t in ex if t)
    n = sum(cnt.values())
    top = cnt.most_common(30)
    print(f"{c}: {len(d):,} argmax pairs p>={pmin}; {n:,} add exactly one token ({n / len(d):.3f}); "
          f"top-10 tokens hold {sum(v for _, v in top[:10]) / max(n, 1):.3f} of them")
    print("  ", ", ".join(f"{t}:{v}" for t, v in top))
    return n / len(d), top


if __name__ == "__main__":
    assert extra("ram marketing", "ram marketing ventures") == "ventures" and extra("a b", "a c") == "" \
        and extra("a", "a") == ""
    for c in sys.argv[1:] or ["France"]:
        main(c)
