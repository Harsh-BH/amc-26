"""P45 stage C (reranker): a LEARNED combiner of the two cross-encoders (bge-reranker-v2-m3 + Qwen3-Reranker-4B LoRA)
under P38's France procedure (LOCO both directions; in-domain blend on the source, out-of-domain weights on the target,
shipped rule, P6's label-free threshold), vs bge alone, Qwen alone and P38's logit mean.

The combiner replaces the 3-weight blend p' = σ(a·logit p + b·logit r + c) with a logistic stack over several CE
columns: 'stack' = [logit p, logit r_bge, logit r_qwen]; 'stack_d' adds |logit r_bge − logit r_qwen| (disagreement).
    ce.py gate [qwen_US_tag qwen_India_tag]        # default: P38's LOCO files
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P38-qwen4b-fr"), str(ROOT)]
import p38                                              # noqa: E402  (P13 machinery, R dir = data/work/p38/r)

p13 = p38.p13
OUT = ROOT / "data/work/p45"


def cols(p, rs, extra):
    L = [p13.rr.logit(p)] + [p13.rr.logit(r) for r in rs]
    if extra:
        L.append(np.abs(L[1] - L[2]))
    return np.column_stack(L)


def fit_k(p, rs, y, extra):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(cols(p, rs, extra), y)
    return list(lr.coef_[0]) + [float(lr.intercept_[0])]


def blend_k(p, rs, w, extra, lo, hi):
    b = (p > lo) & (p < hi) & ~np.any([np.isnan(r) for r in rs], 0)
    out = p.astype(np.float32).copy()
    z = cols(p[b], [r[b] for r in rs], extra) @ np.asarray(w[:-1]) + w[-1]
    out[b] = 1 / (1 + np.exp(-z))
    return out


def gate(q_us="qwen_locoUS_holdout", q_in="qwen_locoIndia_holdout", b_us="bge_locoUS_holdout", b_in="bge_locoIndia_holdout"):
    from ledger import log
    sys.path[:0] = [str(ROOT / "probes/P5-decide"), str(ROOT / "probes/P6-france")]
    import decide as D
    import fr_diag
    LO, HI = p13.LO, p13.HI
    big, t0 = p13._big(), time.time()
    B = big._bench()
    B.SRC["_x"] = big.PROB
    d, S = B.load("_x")
    ids = pd.read_parquet(big.PROB, columns=["s1", "sx"])
    p, y = d.p.to_numpy(), d.is_true.to_numpy()
    h, cty = S.half.to_numpy()[d.s1.to_numpy()], d.country.to_numpy()
    Sc, m1 = S.country.to_numpy(), S.half.to_numpy() == 1
    band = (p > LO) & (p < HI)
    TS = np.round(np.arange(0.5, 0.99, 0.01), 2)

    def kept(pp, c):
        dd = d.assign(p=pp)[cty == c]
        q = dd.p.to_numpy()
        x = D._codes(dd.sx)
        return dd, q, q >= D._gmax(x, q, x.max() + 1)[x]

    def empty(pp, c):
        dd, q, k = kept(pp, c)
        top = np.full(len(S), -1.0)
        np.maximum.at(top, dd.s1.to_numpy()[k], q[k])
        return pd.Series([(top[Sc == c] < t).mean() for t in TS], index=TS)

    def f_at(pp, c, t):
        dd, q, k = kept(pp, c)
        f, _, _ = B.per_s1(dd, S, k & (q >= t))
        return f[m1 & (Sc == c)]

    def proc(a, b, pa, pb):
        ts = p13._plain(B, d.assign(p=pa)[cty == a], S)["t"]
        tb = fr_diag.france_threshold(empty(pb, b), float(empty(pa, a).loc[ts]), ts)
        return f_at(pb, b, tb), ts, tb

    R = {t: p13._r_of(t, ids, big._r) for t in (b_us, b_in, q_us, q_in, f"{b_us}+{q_us}", f"{b_in}+{q_in}")}
    fam = {"bge": ([b_us], [b_in], False), "qwen": ([q_us], [q_in], False),
           "ens": ([f"{b_us}+{q_us}"], [f"{b_in}+{q_in}"], False),              # P38's logit mean
           "stack": ([b_us, q_us], [b_in, q_in], False), "stack_d": ([b_us, q_us], [b_in, q_in], True)}
    base = {(a, b): proc(a, b, p, p)[0] for a, b in (("US", "India"), ("India", "US"))}
    res, F = {}, {}
    for k, (tu, ti, ex) in fam.items():
        rs = {"US": [R[t] for t in tu], "India": [R[t] for t in ti]}          # rs[X] = the CE trained on X only

        def fit(rl, c):
            m = band & ~np.any([np.isnan(r) for r in rl], 0) & (h == 0) & (cty == c)
            return fit_k(p[m], [r[m] for r in rl], y[m], ex)
        w_x = {"India": fit(rs["India"], "US"), "US": fit(rs["US"], "India")}   # OOD weights for target b
        w_mean = list(np.mean([w_x["India"], w_x["US"]], 0))
        for mode in ("mean", "xfit"):
            for a, b in (("US", "India"), ("India", "US")):
                ra = rs[a]
                pa = blend_k(p, ra, fit(ra, a), ex, LO, HI)
                pb = blend_k(p, ra, w_mean if mode == "mean" else w_x[b], ex, LO, HI)
                f, ts, tb = proc(a, b, pa, pb)
                F[(k, mode, a, b)] = f
                dif = f - base[(a, b)]
                res[f"{k}_{mode}_{a}>{b}"] = {"gain": float(dif.mean()), "se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                                              "t_src": ts, "t_dst": tb, "w": [round(float(x), 4) for x in (w_mean if mode == "mean" else w_x[b])]}
                print(f"{k:7s} {mode} {a}>{b}", res[f"{k}_{mode}_{a}>{b}"], f"{time.time() - t0:.0f}s", flush=True)
    for k in ("stack", "stack_d", "ens"):
        for ref in ("qwen", "bge"):
            for mode in ("mean", "xfit"):
                for a, b in (("US", "India"), ("India", "US")):
                    dif = F[(k, mode, a, b)] - F[(ref, mode, a, b)]
                    res[f"{k}-{ref}_{mode}_{a}>{b}"] = {"gain": float(dif.mean()), "se": float(dif.std(ddof=1) / np.sqrt(len(dif))),
                                                        "s1_up": int((dif > 0).sum()), "s1_down": int((dif < 0).sum())}
                    print(f"{k} - {ref} {mode} {a}>{b}", res[f"{k}-{ref}_{mode}_{a}>{b}"], flush=True)
    json.dump(res, open(OUT / "ce_gate.json", "w"), indent=1)
    for k in ("stack", "stack_d"):
        g = [res[f"{k}-qwen_mean_{a}>{b}"] for a, b in (("US", "India"), ("India", "US"))]
        ok = all(x["gain"] > 0 and x["gain"] >= 3 * x["se"] for x in g)
        log("exp", probe="P45-ens-stages", subproblem="rerank-france-combiner", name=f"C-{k}-vs-qwen-loco", world="holdout-dense-half1",
            metrics={f"{kk}": round(v["gain"], 6) for kk, v in res.items() if kk.startswith(k + "-") or kk.startswith(k + "_")}
            | {f"{kk}_se": round(v["se"], 6) for kk, v in res.items() if kk.startswith(k + "-")},
            config={"combiner": "logistic on [logit p, logit r_bge, logit r_qwen]" + (" + |diff|" if k == "stack_d" else ""),
                    "procedure": "P38 gate (P13 France procedure, OOD weights mean/xfit, P6 threshold)",
                    "files": [b_us, b_in, q_us, q_in]},
            hypothesis="a learned bge+Qwen combiner beats Qwen alone for an unseen country in both LOCO directions (>= 3 SE)",
            verdict="keep" if ok else "drop", runtime_s=round(time.time() - t0), peak_gb=round(p38.peak(), 2))


if __name__ == "__main__":
    gate(*sys.argv[2:]) if sys.argv[1] == "gate" else None
