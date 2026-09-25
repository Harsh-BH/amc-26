"""Leave-one-country-out transfer of the decision layer (France proxy): fit on country A's tune half, score on
country B's report half. Compares the D3 threshold rule with D7 stack + EF gate.

    .venv/bin/python probes/P5-decide/loco.py proxy|p4 [D7-stack-all|D7-stack-s1only]
"""
import sys, time

import lightgbm as lgb
import numpy as np

from bench import HERE, load, metrics, sweep
from bench_stack import PRM, VARIANTS
from decide import CTX, _gmax, ctx, select
from proxy import peak

if __name__ == "__main__":
    src = sys.argv[1]
    sys.path.insert(0, str(HERE.parents[1]))
    from ledger import log
    t0 = time.time()
    d, S = load(src)
    g, x = d.s1.to_numpy(), d.sx.to_numpy()
    var = sys.argv[2] if len(sys.argv) > 2 else "D7-stack-all"
    X, y = ctx(d.p.to_numpy(np.float64), g, x)[:, [CTX.index(c) for c in VARIANTS[var]]], d.is_true.to_numpy()
    pr = d.p.to_numpy()                           # exclusivity is taken on the matcher's p, as in select()
    am = d.am.to_numpy(bool) if "am" in d else pr >= _gmax(x, pr, x.max() + 1)[x]
    tune, sub, cty = S.half.to_numpy()[g] == 0, g % 2, d.country.to_numpy()
    T = np.round(np.arange(0.3, 0.86, 0.05), 2)
    for A, B in (("India", "US"), ("US", "India")):
        dA, dB = cty == A, cty == B
        c3, _ = sweep(d[dA], S, {"excl": True}, {"t": T, "t0": T})         # other-country S1: constant offset
        p2 = np.zeros(len(d))
        for k in (0, 1):
            tr, te = tune & dA & (sub != k), tune & dA & (sub == k)
            p2[te] = lgb.train(PRM, lgb.Dataset(X[tr], y[tr]), 300).predict(X[te])
        m = lgb.train(PRM, lgb.Dataset(X[tune & dA], y[tune & dA]), 300)
        p2[dB] = m.predict(X[dB])
        e = d.assign(p=p2, am=am)
        ce, _ = sweep(e[dA & tune], S, {"excl": True, "mode": "ef"}, {"t0": T})
        rB3 = metrics(d[dB], S, select(d[dB], c3), 1)[f"f05_{B}"]
        rB7 = metrics(e[dB], S, select(e[dB], ce), 1)[f"f05_{B}"]
        print(f"fit {A} -> score {B}: D3 {rB3:.4f} (t {c3['t']}, t0 {c3['t0']})  D7+ef-gate {rB7:.4f} (t0 {ce['t0']})",
              flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"{src}-loco-{A}-to-{B}-{var[9:]}", world="holdout-dense-half1",
            metrics={f"f05_{B}_D3": round(rB3, 4), f"f05_{B}_D7efgate": round(rB7, 4)},
            config={"D3": c3, "D7_t0": ce["t0"], "fit_on": A, "score_on": B},
            hypothesis="the D7 stack + gate transfers across countries (France proxy)", verdict="inconclusive",
            runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
