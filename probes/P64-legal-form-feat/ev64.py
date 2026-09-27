"""P64 gate: sub54's India/US combiner recipe refit on P57's honest sw world, with vs without legal-form columns.

Both arms: te44.FINAL + zoo, te44 seeds, fit on half h, scored on half 1-h (both directions), acc = p >= E.T.
Paired per-S1 F0.5 delta (lf arm - base arm) on the report half; LB-eq = E.LB_W weights.
    .venv/bin/python probes/P64-legal-form-feat/ev64.py [world]     -> data/work/p64/ev64_<world>.json
"""
import gc
import json
import os
import sys
import time

os.environ["P44_VARIANT"] = "r"
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "probes/P64-legal-form-feat", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev as E  # noqa: E402
import te44  # noqa: E402
import zoo50 as Z  # noqa: E402
import ev57  # noqa: E402
from attach import OUT  # noqa: E402

LF = ["lf_same", "lf_m1", "lf_mx", "lf_conf", "lf_ov"]           # lf_cls 1, 2, 3, 4, 5 (0 = neither side has a form)
COLS = te44.FINAL + ev57.ZOO
DERIVED = {"has_hn", "exact_hn", "lq", "tw_any", "l_m2hn", "l_hn21", "l_r_all"} | set(ev57.ZOO)
RAW = sorted((set(COLS) - DERIVED) | {"hn_lev", "hn_cls", "q", "n_twins", "p_m2hn", "p_hn21", "p_r_all", "hc", "p", "pa",
                                      "y", "g", "half", "lp"} | {f"p_{m}" for m in Z.MEM + Z.RANK})


def load(w: str) -> dict:
    D = {}
    for c in E.C:
        A = pd.read_parquet(E.W / "p57" / f"A_{w}_{c}.parquet", columns=RAW)
        A = te44.derive(A)
        for m in Z.MEM:
            A[f"z_{m}"] = E.lg(A.pop(f"p_{m}").to_numpy())
        for m in Z.RANK:
            A[f"z_{m}"] = A.pop(f"p_{m}").to_numpy(np.float64)
        k = pd.read_parquet(OUT / f"lf_{w}_{c}.parquet", columns=["lf_cls"]).lf_cls.to_numpy()
        for i, f in enumerate(LF, 1):
            A[f] = (k == i).astype(np.float32)
        A["lf_cls"] = k
        f64 = [x for x in A.columns if A[x].dtype == np.float64 and x not in ("p", "pa", "lp")]
        A[f64] = A[f64].astype(np.float32)
        D[c] = A
        gc.collect()
    return D


def main(w: str = "sw"):
    t0 = time.time()
    S = ev57.load_s()
    D = load(w)
    print("loaded", {c: D[c].shape for c in E.C}, f"{time.time() - t0:.0f}s", flush=True)
    res = {}
    for fit_h in (0, 1):
        rep = 1 - fit_h
        s = {}
        for arm, cols in (("base", COLS), ("lf", COLS + LF)):
            bs = ev57.fit(D, fit_h, cols, te44.SEEDS)
            s[arm] = {c: ev57.score(bs, D[c], cols) for c in E.C}
            if arm == "lf":
                imp = dict(zip(bs[0].feature_name(), bs[0].feature_importance("gain")))
                res[f"imp_h{fit_h}"] = {f: float(imp[f]) for f in LF}
                res[f"imp_h{fit_h}_share"] = float(sum(imp[f] for f in LF) / sum(imp.values()))
            del bs
            gc.collect()
            print(f"fit h{fit_h} {arm} {time.time() - t0:.0f}s", flush=True)
        r = {}
        for c in E.C:
            A, nt, h = D[c], S[c].n_true.to_numpy(), S[c].half.to_numpy()
            fb, fl = (ev57.per_s1(A, nt, s[a][c] >= E.T) for a in ("base", "lf"))
            x = (fl - fb)[h == rep]
            hr = A.half.to_numpy() == rep
            ob, nl = s["base"][c] >= E.T, s["lf"][c] >= E.T
            add, drop = nl & ~ob & hr, ob & ~nl & hr
            y, k = A.y.to_numpy(), A.lf_cls.to_numpy()
            r[c] = dict(d=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), adds=int(add.sum()),
                        drops=int(drop.sum()), add_prec=float(y[add].mean()) if add.any() else None,
                        drop_false=float(1 - y[drop].mean()) if drop.any() else None,
                        flips_by_cls={int(i): int(((add | drop) & (k == i)).sum()) for i in range(6)})
        r["lb"] = sum(E.LB_W[c] * r[c]["d"] for c in E.C)
        r["lbse"] = float(np.sqrt(sum((E.LB_W[c] * r[c]["se"]) ** 2 for c in E.C)))
        res[f"h{fit_h}->h{rep}"] = r
        print(f"h{fit_h}->h{rep}: LB-eq {r['lb']:+.6f} ± {r['lbse']:.6f} | "
              + " | ".join(f"{c} {r[c]['d']:+.6f} ± {r[c]['se']:.6f} (+{r[c]['adds']}/-{r[c]['drops']})" for c in E.C),
              flush=True)
        (OUT / f"ev64_{w}.json").write_text(json.dumps(res, indent=1))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
