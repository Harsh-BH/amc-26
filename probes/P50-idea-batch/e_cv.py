"""P50 E2: does more training data help the sub44r combiner at real scale? 4 random S1 folds over the whole holdout.
For report fold k: BIG = fit on the other 3 folds (1.5x a half), SMALL = fit on 2 of them (1x a half, the shipped size;
the pair rotates with k). 5 seeds each, te44.FINAL + l_r_all, a10 params + bagging, guard, t 0.71. Paired per-S1
F0.5 BIG - SMALL on fold k, pooled over k; also split by run.half so neither half hides a loss.
    .venv/bin/python probes/P50-idea-batch/e_cv.py  -> data/work/p50/e_cv.json"""
import os, sys, json, time
os.environ["P44_VARIANT"] = "r"
import numpy as np
sys.path.insert(0, "probes/P44-meta-stack")
import ev as E, meta as MT  # noqa: E402
import te44  # noqa: E402

COLS = te44.FINAL
K = 4


def bag(tr, D, seeds=range(5)):
    acc = {c: np.zeros(len(D[c][0])) for c in E.C}
    for s in seeds:
        o = MT.gbdt(COLS, seed=s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True)(tr, D)
        for c in E.C:
            acc[c] += MT.lg(o[c])
    return {c: MT.sig(acc[c] / len(seeds)) for c in E.C}


if __name__ == "__main__":
    t0 = time.time()
    D = MT.prep(["m2hn", "hn21", "r_all"])
    fold = {c: np.random.default_rng(500 + i).integers(0, K, len(D[c][1])) for i, c in enumerate(E.C)}
    d = {c: np.full(len(D[c][1]), np.nan) for c in E.C}           # per-S1 BIG - SMALL on its report fold
    fb = {c: np.full(len(D[c][1]), np.nan) for c in E.C}          # per-S1 BIG - #34
    for k in range(K):
        others = [j for j in range(K) if j != k]
        small = others[:2] if k % 2 == 0 else others[1:]
        rowf = {c: fold[c][D[c][0].g.to_numpy()] for c in E.C}
        pb = bag({c: np.isin(rowf[c], others) for c in E.C}, D)
        ps = bag({c: np.isin(rowf[c], small) for c in E.C}, D)
        for c in E.C:
            A, S = D[c]
            m = fold[c] == k
            f_b, f_s = E.per_s1(A, S, pb[c] >= E.T), E.per_s1(A, S, ps[c] >= E.T)
            d[c][m] = (f_b - f_s)[m]
            fb[c][m] = (f_b - S.f34.to_numpy())[m]
            x = (f_b - f_s)[m]
            print(f"fold {k} {c}: BIG-SMALL {x.mean():+.6f} ± {x.std(ddof=1) / np.sqrt(len(x)):.6f}; BIG-#34 {fb[c][m].mean():+.6f}  {time.time() - t0:.0f}s", flush=True)
    out = {}
    def pool(v, sel):
        r = {c: (v[c][sel[c]].mean(), v[c][sel[c]].std(ddof=1) / np.sqrt(sel[c].sum())) for c in E.C}
        return {"lbeq": sum(E.LB_W[c] * r[c][0] for c in E.C), "se": float(np.sqrt(sum((E.LB_W[c] * r[c][1]) ** 2 for c in E.C))),
                **{f"d_{c}": r[c][0] for c in E.C}, **{f"se_{c}": r[c][1] for c in E.C}}
    allm = {c: np.ones(len(D[c][1]), bool) for c in E.C}
    out["big_vs_small_all"] = pool(d, allm)
    out["big_vs_34_all"] = pool(fb, allm)
    for h in (0, 1):
        out[f"big_vs_small_half{h}"] = pool(d, {c: D[c][1].half.to_numpy() == h for c in E.C})
    for k in range(K):
        out[f"big_vs_small_fold{k}"] = pool(d, {c: fold[c] == k for c in E.C})
    for k, v in out.items():
        print(f"{k:22s} LB-eq {v['lbeq']:+.6f} ± {v['se']:.6f} ({v['lbeq'] / v['se']:+.1f} SE) | India {v['d_India']:+.6f} US {v['d_US']:+.6f}", flush=True)
    json.dump(out, open("data/work/p50/e_cv.json", "w"), indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s", flush=True)
