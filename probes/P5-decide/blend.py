"""MERGE-2 blend refit check: (a) threshold fallback on the blended p (P4 M2 + P4b cross-encoder) vs (b) the full CFG
(D7 stack + excl + EF top-k + gate) refit on the blended p by `bench_stack.py blend D7-stack-s1only`.
Same half-split protocol: tune = hash half 0 of the holdout S1, report = half 1; paired SE over report S1.

    .venv/bin/python probes/P5-decide/blend.py
"""
import json, sys, time

import numpy as np

from bench import HERE, load, metrics, per_s1, sweep
from decide import FALLBACK, select
from proxy import peak

if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1]))
    from ledger import log
    t0 = time.time()
    d, S = load("blend")
    m = S.half.to_numpy() == 1
    T = np.round(np.arange(0.5, 0.91, 0.05), 2)
    c, _ = sweep(d, S, {"excl": True}, {"t": T, "t0": T})
    c, _ = sweep(d, S, {"excl": True}, {"t": np.round(c["t"] + np.arange(-0.03, 0.031, 0.01), 2),
                                        "t0": np.round(c["t0"] + np.arange(-0.03, 0.031, 0.01), 2)})
    A = {"a-t0.70": {"excl": True, "t": 0.70}, "a-FALLBACK": FALLBACK, "a-retuned": c}
    b = json.load(open(HERE / "cfg_blend_s1only.json"))
    m2 = {**json.load(open(HERE / "cfg_p4_s1only.json"))}          # M2's stack applied unrefit, for contrast
    cfgs = {**A, "b-CFG-refit": b, "b-M2-stack-unrefit": m2}
    f, res = {}, {}
    for k, cfg in cfgs.items():
        sel = select(d, cfg)
        f[k] = per_s1(d, S, sel)[0]
        res[k] = (metrics(d, S, sel, 0)["f05"], metrics(d, S, sel, 1))
    best_a = max(A, key=lambda k: res[k][0])                   # the strictest baseline: best (a) on the TUNE half
    for k in cfgs:
        tune, rep = res[k]
        df = (f[k] - f[best_a])[m]
        line = f"{k:20s} tune {tune:.4f} report {rep['f05']:.4f}  vs {best_a}: {df.mean():+.4f} ± " \
               f"{df.std() / np.sqrt(len(df)):.4f}  " + "  ".join(f"{n}={v:.4f}" for n, v in rep.items() if n != "f05")
        print(line, flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"blend-{k}", world="holdout-dense-half1",
            metrics={"f05_tune": round(tune, 4), **{n: round(float(v), 4) for n, v in rep.items()},
                     f"diff_vs_{best_a}": round(float(df.mean()), 4), "diff_se": round(float(df.std() / np.sqrt(len(df))), 4)},
            config={n: v for n, v in cfg.items() if n not in ("stack_cols", "report_f05", "fallback_thresh")},
            hypothesis="MERGE-2: the D7 stack refit on blended p beats the threshold fallback by >= 0.001",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
    print("retuned (a):", c, f"done {time.time() - t0:.0f}s peak {peak():.2f} GB")
