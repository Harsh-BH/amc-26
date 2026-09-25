"""D7: learned second stage on ctx(s1, sx, p) features. Fit on the tune half (2-fold OOF inside it for the
threshold sweep), report on the report half. Writes probes/P5-decide/stack_<src>.txt (LightGBM model text).

    .venv/bin/python probes/P5-decide/bench_stack.py proxy|p4 [variant ...]
"""
import json, sys, time

import lightgbm as lgb
import numpy as np

from bench import HERE, load, metrics, sweep, WORK
from decide import CTX, S1_CTX, _gmax, ctx, select
from proxy import peak

PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
S1_ONLY = S1_CTX
V1 = CTX[:14]                                     # first run: without the S1's 2nd..5th p
VARIANTS = {"D7-stack-all": CTX, "D7-stack-s1only": S1_ONLY, "D7-stack-v1": V1}

if __name__ == "__main__":
    src = sys.argv[1]
    sys.path.insert(0, str(HERE.parents[1]))
    from ledger import log
    t0 = time.time()
    d, S = load(src)
    g, x = d.s1.to_numpy(), d.sx.to_numpy()
    Xall = ctx(d.p.to_numpy(np.float64), g, x)
    y = d.is_true.to_numpy()
    pr = d.p.to_numpy()                           # exclusivity is taken on the matcher's p, as in select()
    am = d.am.to_numpy(bool) if "am" in d else pr >= _gmax(x, pr, x.max() + 1)[x]
    tune = S.half.to_numpy()[g] == 0
    sub = g % 2                                   # S1 row order is random w.r.t. matches (random ids)
    for name in sys.argv[2:] or VARIANTS:
        cols = VARIANTS[name]
        X = Xall[:, [CTX.index(c) for c in cols]]
        p2 = np.zeros(len(d))
        for k in (0, 1):                          # OOF on the tune half, for an honest threshold sweep
            tr, te = tune & (sub != k), tune & (sub == k)
            p2[te] = lgb.train(PRM, lgb.Dataset(X[tr], y[tr]), 300).predict(X[te])
        m = lgb.train(PRM, lgb.Dataset(X[tune], y[tune]), 300)
        p2[~tune] = m.predict(X[~tune])
        e = d.assign(p=p2, am=am)
        c, _ = sweep(e, S, {"excl": True}, {"t": np.round(np.arange(0.3, 0.86, 0.05), 2),
                                            "t0": np.round(np.arange(0.3, 0.86, 0.05), 2)})
        c, _ = sweep(e, S, {"excl": True}, {"t": c["t"] + np.arange(-0.03, 0.031, 0.01),
                                            "t0": c["t0"] + np.arange(-0.03, 0.031, 0.01)})
        tune_m, rep = metrics(e, S, select(e, c), 0), metrics(e, S, select(e, c), 1)
        cfg = {**c, "stack": m.model_to_string(), "stack_cols": cols}
        chk = metrics(d, S, select(d, cfg), 1)["f05"]      # the cfg alone reproduces the report number
        assert abs(chk - rep["f05"]) < 1e-9, (chk, rep["f05"])
        (HERE / f"stack_{src}_{name[9:]}.txt").write_text(cfg["stack"])
        imp = dict(zip(cols, np.round(m.feature_importance("gain") / m.feature_importance("gain").sum(), 3)))
        print(f"{name:20s} t={c['t']:.2f} t0={c['t0']:.2f} tune {tune_m['f05']:.4f}  report {rep['f05']:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in rep.items() if k != "f05"), "\n  imp", imp, flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"{src}-{name}", world="holdout-dense-half1",
            metrics={"f05_tune": round(tune_m["f05"], 4), **{k: round(float(v), 4) for k, v in rep.items()}},
            config={"excl": True, "t": c["t"], "t0": c["t0"], "stack": f"lgbm 300x31 on ctx {cols}",
                    "model_file": f"probes/P5-decide/stack_{src}_{name[9:]}.txt"},
            hypothesis="a second stage on the pair's standing among its S1/SX candidates fixes per-S1 set size",
            verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
        ce, _ = sweep(e, S, {"excl": True, "mode": "ef"}, {"t0": np.round(np.arange(0.3, 0.86, 0.05), 2)})
        tu2, rep2 = metrics(e, S, select(e, ce), 0), metrics(e, S, select(e, ce), 1)
        print(f"{name + '+ef-gate':20s} t0={ce['t0']:.2f} tune {tu2['f05']:.4f}  report {rep2['f05']:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in rep2.items() if k != "f05"), flush=True)
        log("exp", probe="P5-decide", subproblem="decision", name=f"{src}-{name}+ef-gate", world="holdout-dense-half1",
            metrics={"f05_tune": round(tu2["f05"], 4), **{k: round(float(v), 4) for k, v in rep2.items()}},
            config={"excl": True, "mode": "ef", "t0": ce["t0"], "stack": f"lgbm 300x31 on ctx {cols}"},
            hypothesis="EF (k>=1) + gate on the second-stage p", verdict="inconclusive",
            runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
        out = {"excl": True, "mode": "ef", "t0": float(ce["t0"]), "cap": 24,
               "stack": f"probes/P5-decide/stack_{src}_{name[9:]}.txt", "stack_cols": cols,
               "report_f05": round(float(rep2["f05"]), 4), "fallback_thresh": {"t": float(c["t"]), "t0": float(c["t0"])}}
        (HERE / f"cfg_{src}_{name[9:]}.json").write_text(json.dumps(out, indent=1))
    print(f"done {time.time() - t0:.0f}s peak {peak():.2f} GB")
