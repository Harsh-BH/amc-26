"""P56 arm 1b: fit-free group-state bars. A pair whose S1 already has k accepted higher-p'' pairs (both sources, greedy)
is worth adding iff p'' >= q*(k), the exact per-S1 F0.5 break-even when the k members are true and complete:
  q*(k) = (1 - Fa) / ((1 - Fn) + (1 - Fa)),  Fn = 1.25k / (1.25k + 0.25),  Fa = 1.25k / (1.25k + 1)
  -> 0.5, 0.727, 0.759, 0.771, 0.778, 0.782, ...  (k = 0 is the empty-S1 case).
theory: q*(k) for every k; theory_hi: k = 0 keeps 0.71 (raise-only, no empty fill: P54 owns that); theory_lo: only k = 0
moves (to 0.5). No fitting: both halves are honest reports vs the one-t 0.71 base.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/arm1b.py"""
import sys
import time
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
import arm1 as A1  # noqa: E402


def qstar(k):
    k = np.asarray(k, float)
    fn, fa = 1.25 * k / (1.25 * k + 0.25), 1.25 * k / (1.25 * k + 1)
    return (1 - fa) / ((1 - fn) + (1 - fa))


assert np.allclose(qstar([0, 1, 2]), [0.5, 0.4444 / 0.6111, 0.2857 / 0.3766], atol=1e-3)


def rank_any(d):
    key = d["g"]
    o2, st = A1.order(d["pa"], key)
    r = np.empty(len(key), np.int64)
    r[o2] = np.arange(len(key)) - st
    return r


if __name__ == "__main__":
    from ledger import log
    D = {c: A1.load(c) for c in A1.LB_W}
    for name in ("theory", "theory_hi", "theory_lo"):
        ts = time.time()
        out = {}
        for c, d in D.items():
            k = rank_any(d)
            q = qstar(np.minimum(k, 50))
            if name == "theory_hi":
                q = np.where(k == 0, A1.T0, q)
            if name == "theory_lo":
                q = np.where(k == 0, 0.5, A1.T0)
            x = A1.score(d, q, d["g"]) - d["base"]
            out[c] = {h: (x[d["half"] == h].mean(), x[d["half"] == h].std(ddof=1) / np.sqrt((d["half"] == h).sum()),
                          int((x[d["half"] == h] > 0).sum()), int((x[d["half"] == h] < 0).sum())) for h in (0, 1)}
        lb = {h: sum(A1.LB_W[c] * out[c][h][0] for c in D) for h in (0, 1)}
        se = {h: float(np.sqrt(sum((A1.LB_W[c] * out[c][h][1]) ** 2 for c in D))) for h in (0, 1)}
        print(name, {h: f"LB {lb[h]:+.6f} ± {se[h]:.6f}" for h in (0, 1)},
              {c: {h: f"{v[0]:+.6f}±{v[1]:.6f} up{v[2]} dn{v[3]}" for h, v in out[c].items()} for c in D}, flush=True)
        ok = min(lb.values()) >= 1e-4 and all(lb[h] >= 3 * se[h] for h in (0, 1))
        log("exp", probe="P56-thresh-grain", subproblem="decision", name=f"arm1b-{name}", world="holdout",
            metrics={"lbeq_h0": lb[0], "se_h0": se[0], "lbeq_h1": lb[1], "se_h1": se[1],
                     **{f"d_{c}_h{h}": out[c][h][0] for c in D for h in (0, 1)}},
            config={"q_star": [round(float(v), 4) for v in qstar(range(6))], "greedy": "rank in S1 by p'' (both sources)",
                    "variant": name, "fit": "none"},
            hypothesis="the exact F0.5 break-even per group state beats one t 0.71 on sub44r's p''",
            verdict="keep" if ok else "drop", runtime_s=round(time.time() - ts, 1), peak_gb=0.5)
