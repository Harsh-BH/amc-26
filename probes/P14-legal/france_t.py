"""P14-legal: France's threshold for the legal model, label-free. #7 ships France at t 0.95 on m2_big (the
leaderboard optimum, D4). The legal model's p is on another scale, so its France t is the one whose predicted
empty-S1 rate equals m2_big's at 0.95: the same answer volume as #7's France, the new model choosing the pairs.
Both on the pruned test pairs (M2's p: France is not reranked).

    .venv/bin/python probes/P14-legal/france_t.py     # -> data/work/p14/france_t.json
"""
import json
import sys

sys.path[:0] = ["code/business_entity_resolution/src", "."]

import numpy as np

import features as F
import model as M
import run
from worlds import WORK

if __name__ == "__main__":
    run.use("sub7")
    n = len(F.ids("test", "France")[0])
    ts = np.round(np.arange(0.50, 0.99, 0.01), 2)
    curves = {}
    for b in ("m2_big.txt", "m2_legal_big.txt"):
        M.BOOSTER, M._M = WORK / "p4" / b, None
        d = M.split_probs("test", "France")
        curves[b] = run.empty_curve(d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32), n, ts)
    ref = float(curves["m2_big.txt"].loc[0.95])
    t = float((curves["m2_legal_big.txt"] - ref).abs().idxmin())
    out = {"t_france": t, "empty_rate_m2_big_at_0.95": ref, "empty_rate_legal_at_t": float(curves["m2_legal_big.txt"].loc[t]),
           "legal_empty_rate_at_0.95": float(curves["m2_legal_big.txt"].loc[0.95])}
    (WORK / "p14" / "france_t.json").write_text(json.dumps(out, indent=1))
    print(out)
