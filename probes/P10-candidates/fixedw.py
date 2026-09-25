"""The deliverable's own number: #5's blend weights kept (refit=False), D3 still tuned on half 0, at the chosen taus.

    fixedw.py [tau ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from prep import OUT, peak_gb
from evalp import Bench, W_BIG


def main(taus):
    from ledger import log
    t0 = time.time()
    b = Bench(cols=["rule_acc"])
    q = np.concatenate([pd.read_parquet(OUT / f"hold_q_{c}.parquet").q.to_numpy() for c in ("India", "US")])
    ra = b.h.rule_acc.to_numpy() > 0
    b.baseline()
    for tau in taus:
        m, _ = b.run((q >= tau) | ra, refit=False, name=f"prune-tau{tau}-w#5")
        log("exp", probe="P10-candidates", subproblem="pruning", name=f"prune-tau{tau}-fixed-w", world="holdout-dense-half1",
            metrics={k: round(float(v), 5) for k, v in m.items() if not isinstance(v, list)},
            config={"pruner": "pruner.txt", "w": W_BIG, "refit": "D3 only (w fixed = #5)"},
            hypothesis="keeping #5's blend weights (the deliverable's p) loses no more than the refit yardstick",
            verdict="keep" if abs(tau - 0.03) < 1e-9 else "inconclusive", runtime_s=round(time.time() - t0),
            peak_gb=peak_gb())


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [0.02, 0.03, 0.05])
