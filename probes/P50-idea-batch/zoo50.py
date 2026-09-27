"""P50 max-stack: P45's zoo members as extra columns of the sub44r combiner (te44.FINAL + l_r_all), holdout halves,
5 seeds, both directions, vs the same combiner without them.   -> data/work/p50/zoo50.json"""
import os, sys, json, time
os.environ["P44_VARIANT"] = "r"
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P44-meta-stack")
import ev as E, meta as MT  # noqa: E402
import te44  # noqa: E402

MEM = ["bag1", "bag2", "bag3", "bag4", "cat", "dart", "deep", "et2", "extra", "fsub", "goss", "mlp", "mlp2", "noctx"]
RANK = ["rank_s1", "rank_sx2"]


def attach(D):
    for c in E.C:
        A = D[c][0]
        z = pd.read_parquet(f"data/work/p45/oof/holdout_{c}.parquet", columns=["s1", "sx"] + [f"p_{m}" for m in MEM + RANK])
        z = A[["s1", "sx"]].merge(z.drop_duplicates(["s1", "sx"]), on=["s1", "sx"], how="left")
        assert len(z) == len(A) and z[f"p_{MEM[0]}"].notna().all(), z.isna().mean()
        for m in MEM:
            A[f"z_{m}"] = MT.lg(z[f"p_{m}"].to_numpy())
        for m in RANK:
            A[f"z_{m}"] = z[f"p_{m}"].to_numpy(np.float64)
        L = A[[f"z_{m}" for m in MEM]].to_numpy()
        A["z_mean"], A["z_std"], A["z_min"], A["z_max"] = L.mean(1), L.std(1), L.min(1), L.max(1)
        A["z_dev"] = A.z_mean - A.lp                              # zoo consensus vs #8c's logit
        del z, L


def bag(cols, tr, D):
    acc = {c: np.zeros(len(D[c][0])) for c in E.C}
    for s in range(5):
        o = MT.gbdt(cols, seed=s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, deterministic=True)(tr, D)
        for c in E.C:
            acc[c] += MT.lg(o[c])
    return {c: MT.sig(acc[c] / 5) for c in E.C}


if __name__ == "__main__":
    t0 = time.time()
    D = MT.prep(["m2hn", "hn21", "r_all"])
    attach(D)
    F0 = te44.FINAL
    arms = {"zoo_all": F0 + [f"z_{m}" for m in MEM + RANK], "zoo_stats": F0 + ["z_mean", "z_std", "z_min", "z_max", "z_dev"]}
    res = {}
    for fit in (0, 1):
        rep = 1 - fit
        tr = {c: D[c][0].half.to_numpy() == fit for c in E.C}
        base = bag(F0, tr, D)
        fb = {c: E.per_s1(D[c][0], D[c][1], base[c] >= E.T)[D[c][1].half.to_numpy() == rep] for c in E.C}
        for name, cols in arms.items():
            p = bag(cols, tr, D)
            fa = {c: E.per_s1(D[c][0], D[c][1], p[c] >= E.T)[D[c][1].half.to_numpy() == rep] for c in E.C}
            r = {c: ((fa[c] - fb[c]).mean(), (fa[c] - fb[c]).std(ddof=1) / np.sqrt(len(fa[c]))) for c in E.C}
            lb = sum(E.LB_W[c] * r[c][0] for c in E.C)
            se = float(np.sqrt(sum((E.LB_W[c] * r[c][1]) ** 2 for c in E.C)))
            res[f"{name}_h{fit}h{rep}"] = {"lbeq": lb, "se": se, **{f"d_{c}": r[c][0] for c in E.C}}
            print(f"{name} h{fit}->h{rep}: LB-eq {lb:+.6f} ± {se:.6f} | India {r['India'][0]:+.6f} US {r['US'][0]:+.6f}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open("data/work/p50/zoo50.json", "w"), indent=1, default=float)
