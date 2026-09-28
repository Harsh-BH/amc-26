"""P15 feasibility check: does the SIGNED house-number delta separate true pairs from generated orphan siblings?

P12 measured the generator: orphan siblings take hn + d, d in {1,2,3,4,5,7,9,11,13,21} (always positive), while noise on
true pairs is a digit drop/add (|d| >= 50, ~70-80%) or a small +-1/+-2. The matcher only sees hn_eq / hn_diff.
Run on the P10 holdout rows (m2_big p, fold-0 S1, every competing train pair):
    .venv/bin/python probes/P15-research/hn_check.py India
Prints aggregates only (no raw rows).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
import features as F  # noqa: E402

SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def hn_cls(h1: np.ndarray, hx: np.ndarray) -> np.ndarray:
    """0 none/equal, 1 sibling-like (+d, d in SIB), 2 digit drop/add (one is a suffix/prefix of the other),
    3 small negative / other small, 4 other."""
    out = np.zeros(len(h1), np.int8)
    for k, (a, b) in enumerate(zip(h1, hx)):
        if not a or not b or a == b:
            continue
        d = int(b) - int(a)
        if d in SIB:
            out[k] = 1
        elif len(a) != len(b) and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
            out[k] = 2
        elif abs(d) <= 21:
            out[k] = 3
        else:
            out[k] = 4
    return out


def main(c: str):
    t0 = time.time()
    h = pd.read_parquet(ROOT / f"data/work/p10/hold_{c}.parquet",
                        columns=["q1", "qx", "fold", "is_true", "hn_diff", "p"])
    s1, sx = F.ids("train", c)
    need1, needx = np.unique(h.q1.to_numpy()), np.unique(h.qx.to_numpy())

    def hn_of(ids_q: pa.Array, need: np.ndarray, side: str) -> pd.Series:
        rf = pq.read_table(ROOT / f"data/work/rf_train_{side}_{c}.parquet", columns=["id", "hn"])
        want = ids_q.take(pa.array(need))
        pos = pc.index_in(want, value_set=rf.column("id")).to_numpy(zero_copy_only=False)
        return pd.Series(rf.column("hn").to_numpy(zero_copy_only=False)[pos], index=need)

    H1, HX = hn_of(s1, need1, "s1"), hn_of(sx, needx, "sx")
    h["h1"] = H1.reindex(h.q1.to_numpy()).to_numpy()
    h["hx"] = HX.reindex(h.qx.to_numpy()).to_numpy()
    del H1, HX
    d = h[h.hn_diff > 0].copy()
    d["cls"] = hn_cls(d.h1.to_numpy(), d.hx.to_numpy())
    # shipped-rule proxy on m2_big p: argmax per SX over the rows here, p >= 0.69
    h["am"] = h.p.to_numpy() == h.groupby("qx").p.transform("max").to_numpy()
    h["acc"] = h.am & (h.p >= 0.69)
    d["acc"] = h.loc[d.index, "acc"]
    ho = h.fold == 0
    tru = h.is_true > 0
    print(f"{c}: {len(h):,} rows, holdout true pairs {int((ho & tru).sum()):,}, "
          f"missed {int((ho & tru & ~h.acc).sum()):,} ({(ho & tru & ~h.acc).sum() / (ho & tru).sum():.4f}); "
          f"false accepted {int((ho & ~tru & h.acc).sum()):,}")
    mt = ho & tru & ~h.acc
    print(f"  missed true pairs with hn_diff: {(mt & (h.hn_diff > 0)).sum() / mt.sum():.3f}; "
          f"all holdout true with hn_diff: {(ho & tru & (h.hn_diff > 0)).sum() / (ho & tru).sum():.3f}")
    fa = ho & ~tru & h.acc
    print(f"  false-accepted with hn_diff: {(fa & (h.hn_diff > 0)).sum() / max(fa.sum(), 1):.3f}")
    names = {0: "eq/none", 1: "+SIB", 2: "digit drop/add", 3: "small other", 4: "far"}
    dd = d[d.fold == 0]
    tab = pd.crosstab(dd.cls.map(names), [dd.is_true.gt(0).map({True: "true", False: "false"}), dd.acc],
                      margins=True)
    print("  hn_diff rows by class x (truth, accepted):")
    print(tab.to_string())
    # the +SIB true pairs: which d?
    tr = dd[(dd.cls == 1)]
    dv = tr.hx.astype(np.int64) - tr.h1.astype(np.int64)
    print("  +SIB rows: share true by d:",
          pd.crosstab(dv, tr.is_true > 0, normalize="index").round(3)[True].to_dict())
    print(f"  time {time.time() - t0:.0f}s")
    h[["q1", "qx", "fold", "is_true", "p", "am", "acc", "h1", "hx"]].to_parquet(
        ROOT / f"data/work/p15/_hn_{c}.parquet", index=False)


if __name__ == "__main__":
    assert list(hn_cls(np.array(["10", "10", "5402", "10", ""]), np.array(["13", "8", "402", "10", "3"]))) == [1, 3, 2, 0, 0]
    main(sys.argv[1] if len(sys.argv) > 1 else "India")
