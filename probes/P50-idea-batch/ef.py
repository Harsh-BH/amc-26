"""P50 E + F on P44's holdout frames (sub44r's model = te44.FINAL + l_r_all, a10 params + bagging, guard, t 0.71).
E: learning curve - fit on 25/50/75/100% of half `fit`'s S1 (5 seeds each), report on the other half vs #34.
F: 5 vs 10 vs 20 seeds on the full fit half, paired per-S1 differences on the report half.
Both directions. -> data/work/p50/ef.json
    .venv/bin/python probes/P50-idea-batch/ef.py"""
import os, sys, json, time
os.environ["P44_VARIANT"] = "r"
from pathlib import Path
import numpy as np
sys.path.insert(0, "probes/P44-meta-stack")
import ev as E, meta as MT  # noqa: E402
import te44  # noqa: E402

COLS = te44.FINAL
assert COLS[-1] == "l_r_all"
FR = (0.25, 0.5, 0.75, 1.0)
NSEED = 20


def per_lb(D, p, rep):
    """per-country per-S1 F0.5 arrays (report half) for scores p."""
    return {c: E.per_s1(D[c][0], D[c][1], p[c] >= E.T)[D[c][1].half.to_numpy() == rep] for c in E.C}


def stat(fa, fb):
    """LB-eq mean / SE of per-S1 fa - fb (paired), + per country."""
    r = {c: ((fa[c] - fb[c]).mean(), (fa[c] - fb[c]).std(ddof=1) / np.sqrt(len(fa[c]))) for c in E.C}
    lb = sum(E.LB_W[c] * r[c][0] for c in E.C)
    se = float(np.sqrt(sum((E.LB_W[c] * r[c][1]) ** 2 for c in E.C)))
    return {"lbeq": lb, "se": se, **{f"d_{c}": r[c][0] for c in E.C}, **{f"se_{c}": r[c][1] for c in E.C}}


if __name__ == "__main__":
    t0 = time.time()
    D = MT.prep(["m2hn", "hn21", "r_all"])
    f34 = {rep: {c: D[c][1].f34.to_numpy()[D[c][1].half.to_numpy() == rep] for c in E.C} for rep in (0, 1)}
    res = {}
    for fit in (0, 1):
        rep = 1 - fit
        rng = np.random.default_rng(50 + fit)
        u = {c: rng.random(len(D[c][1])) for c in E.C}           # S1-level draw: frac f keeps u < f
        bags = {}
        for f in FR:
            tr = {c: (D[c][0].half.to_numpy() == fit) & (u[c][D[c][0].g.to_numpy()] < f) for c in E.C}
            ns = NSEED if f == 1.0 else 5
            acc = {c: np.zeros(len(D[c][0])) for c in E.C}
            for s in range(ns):
                o = MT.gbdt(COLS, seed=s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True)(tr, D)
                for c in E.C:
                    acc[c] += MT.lg(o[c])
                if s + 1 in (5, 10, 20):
                    bags[(f, s + 1)] = {c: MT.sig(acc[c] / (s + 1)) for c in E.C}
                    print(f"fit h{fit} frac {f} seeds {s + 1}: {time.time() - t0:.0f}s", flush=True)
        if fit == 0:                                              # the shipped sub44r boosters = frac 1, seeds 0-4
            for c in E.C:
                ref = np.load(E.OUT / f"ho_p44r_{c}.npy")
                print(c, "bag5 vs shipped ho_p44r max|d|", float(np.abs(bags[(1.0, 5)][c] - ref).max()), flush=True)
        F = {k: per_lb(D, v, rep) for k, v in bags.items()}
        r = {}
        for f in FR:
            r[f"E_frac{f}_vs34"] = stat(F[(f, 5)], f34[rep])
        r["E_frac1.0_vs_frac0.5"] = stat(F[(1.0, 5)], F[(0.5, 5)])
        r["E_frac1.0_vs_frac0.75"] = stat(F[(1.0, 5)], F[(0.75, 5)])
        r["F_seed10_vs_seed5"] = stat(F[(1.0, 10)], F[(1.0, 5)])
        r["F_seed20_vs_seed5"] = stat(F[(1.0, 20)], F[(1.0, 5)])
        r["F_seed20_vs34"] = stat(F[(1.0, 20)], f34[rep])
        for k, v in r.items():
            print(f"h{fit}->h{rep} {k:24s} LB-eq {v['lbeq']:+.6f} ± {v['se']:.6f} | India {v['d_India']:+.6f} US {v['d_US']:+.6f}", flush=True)
        res[f"h{fit}h{rep}"] = r
        np.savez_compressed(f"data/work/p50/ef_bag20_fit{fit}.npz", **{c: bags[(1.0, 20)][c].astype(np.float32) for c in E.C})
    json.dump(res, open("data/work/p50/ef.json", "w"), indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s", flush=True)
