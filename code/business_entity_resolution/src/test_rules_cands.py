"""Guard for the bounded-memory rules used on the full test set: the same exact-key join as `match`, partitioned by
state (the unpartitioned join reached 10.6 GB on test name chains). Dense holdout, all 2.2M train S1 competing.
It may give up at most ~0.5 pt against the unpartitioned join (0.8201) and must keep pair precision >= 0.98.
(An earlier variant judged uniqueness only among P3 candidates; P3's holdout export keeps holdout S1 only, so it
could not be validated honestly — 0.764 / precision 0.83 on that file — and was dropped.)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rules import evaluate

if __name__ == "__main__":          # forkserver workers re-import this file
    m = evaluate("holdout", dense=True, by_state=True)
    assert m["f05"] >= 0.815, m
    assert m["pair_precision"] >= 0.98, m
    print(f"state-partitioned rules test ok  dense f05={m['f05']:.4f}  pair_precision={m['pair_precision']:.4f}")
