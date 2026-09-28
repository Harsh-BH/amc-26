"""P50 sub52_max India/US: the full-holdout refit (E) with P45's zoo members as extra combiner columns (zoo50.arms
zoo_all). fit -> data/work/p50/fullzoo_s{0..4}.txt; probs -> data/work/p50/prob_test_{c}_fullzoo.parquet.
    e_zoo.py fit | probs"""
import os, sys
os.environ["P44_VARIANT"] = "r"
import numpy as np, pandas as pd
sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch"]
import ev as E, meta as MT  # noqa: E402
import te44  # noqa: E402
import zoo50 as Z  # noqa: E402
from e_ship import P50, attach  # noqa: E402

COLS = te44.FINAL + [f"z_{m}" for m in Z.MEM + Z.RANK]


def zcols(A, zt):
    for m in Z.MEM:
        A[f"z_{m}"] = MT.lg(zt[f"p_{m}"].to_numpy(np.float64)).astype(np.float32)
    for m in Z.RANK:
        A[f"z_{m}"] = zt[f"p_{m}"].to_numpy(np.float32)
    return A


def fit():
    import lightgbm as lgb
    D = MT.prep(te44.MODELS)
    Z.attach(D)
    X = pd.concat([D[c][0][COLS] for c in E.C])
    y = np.concatenate([D[c][0].y.to_numpy() for c in E.C])
    base = np.concatenate([D[c][0].lp.to_numpy() for c in E.C])
    for s in te44.SEEDS:
        b = lgb.train(te44.PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True), 200)
        b.save_model(str(P50 / f"fullzoo_s{s}.txt"))
        print("booster", s, flush=True)


def probs(tag="fullzoo", chunk=400_000):
    import lightgbm as lgb
    bs = [lgb.Booster(model_file=str(P50 / f"{tag}_s{s}.txt")) for s in te44.SEEDS]
    for c in E.C:
        A = te44.derive(attach(c, pd.read_parquet(P50 / f"T_{c}.parquet")))
        A = zcols(A, pd.read_parquet(P50 / f"zoo_test_{c}.parquet"))
        p = np.empty(len(A))
        for i in range(0, len(A), chunk):                      # chunked: a full float64 copy of 67 cols x 3.7M rows is 2 GB
            X = A.iloc[i:i + chunk]
            outs = []
            for b in bs:
                s = MT.sig(b.predict(X[COLS], raw_score=True) + X.lp.to_numpy())
                outs.append(MT.lg(np.where(X.hc.to_numpy() == 1, X.p.to_numpy(), s)))
            p[i:i + chunk] = MT.sig(np.mean(outs, axis=0))
        pr = pd.read_parquet(te44.P34 / f"prob_test_{c}_a10.parquet")
        out = pr.p.to_numpy().copy()
        out[A.row.to_numpy()] = p.astype(out.dtype)
        pr.assign(p=out).to_parquet(P50 / f"prob_test_{c}_{tag}.parquet", index=False)
        old, new = A.pa.to_numpy() >= E.T, p >= E.T
        print(c, f"{tag}: #34 accepts {old.sum():,}, {tag} {new.sum():,}; adds {(new & ~old).sum():,}, drops {(old & ~new).sum():,}", flush=True)
        del A, pr


if __name__ == "__main__":
    {"fit": fit, "probs": probs}[sys.argv[1]]()
