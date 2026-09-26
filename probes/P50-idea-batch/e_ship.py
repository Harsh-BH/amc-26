"""P50 E ship: the sub44r combiner refit on the FULL holdout (both halves, 5 seeds), applied to test.
P44's te44 path with its work dir pointed at data/work/p50 (T_{c} frames rebuilt there; P44's were deleted for disk).
    e_ship.py fit             -> data/work/p50/full_s{0..4}.txt (India + US pooled, every holdout argmax row)
    e_ship.py build <c>       -> data/work/p50/T_{c}.parquet (te44.build; needs pred_m2hn / pred_hn21 test files here)
    e_ship.py probs <tag> <booster-prefix>   -> data/work/p50/prob_test_{c}_<tag>.parquet (shipr = P44's sub44r boosters)
    e_ship.py combine <tag> SRC OUT          -> te44.combine with the <tag> prob files (SRC India/US = #34, e.g. sub46_on43)
"""
import os, sys
os.environ["P44_VARIANT"] = "r"
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "probes/P44-meta-stack")
import ev as E, meta as MT  # noqa: E402
import te44  # noqa: E402

P50 = E.W / "p50"


def fit():
    import lightgbm as lgb
    D = MT.prep(te44.MODELS)
    X = pd.concat([D[c][0][te44.FINAL] for c in E.C])
    y = np.concatenate([D[c][0].y.to_numpy() for c in E.C])
    base = np.concatenate([D[c][0].lp.to_numpy() for c in E.C])
    for s in te44.SEEDS:
        b = lgb.train(te44.PRM(s), lgb.Dataset(X, y, init_score=base, categorical_feature=["cat"], free_raw_data=True), 200)
        b.save_model(str(P50 / f"full_s{s}.txt"))
        print("booster", s, "saved", flush=True)


def attach(c, A):
    """r_all = P45's every-argmax-pair test file (band pairs = P13's band r, the rest = the GPU slices)."""
    f = pd.read_parquet(E.W / f"p45/oof/r_all_test_{c}.parquet").drop_duplicates(["s1", "sx"])
    tp = pd.read_parquet(te44.P34 / f"te_pairs_{c}.parquet", columns=["s1", "sx"]).iloc[A.row.to_numpy()].reset_index(drop=True)
    A["p_r_all"] = tp.merge(f, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
    band = A.r.notna().to_numpy()
    assert A.p_r_all.notna().all(), f"r_all missing for {A.p_r_all.isna().sum()}"
    assert np.allclose(A.p_r_all.to_numpy()[band], A.r.to_numpy()[band], atol=1e-6), "band r != te_feat r"
    return A


def probs(tag, prefix):
    import lightgbm as lgb
    bs = [lgb.Booster(model_file=str((E.OUT if prefix == "shipr" else P50) / f"{prefix}_s{s}.txt")) for s in te44.SEEDS]
    for c in E.C:
        A = te44.derive(attach(c, pd.read_parquet(P50 / f"T_{c}.parquet")))
        p = te44.combine_p(bs, A)
        pr = pd.read_parquet(te44.P34 / f"prob_test_{c}_a10.parquet")
        out = pr.p.to_numpy().copy()
        out[A.row.to_numpy()] = p.astype(out.dtype)
        pr.assign(p=out).to_parquet(P50 / f"prob_test_{c}_{tag}.parquet", index=False)
        old, new = A.pa.to_numpy() >= E.T, p >= E.T
        print(c, f"{tag}: argmax pairs {len(A):,}; #34 accepts {old.sum():,}, {tag} {new.sum():,}; adds {(new & ~old).sum():,}, "
                 f"drops {(old & ~new).sum():,}", flush=True)
        del A, pr


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit()
    elif cmd == "build":
        E.OUT = P50                          # te44.build reads pred_<m>/test_<c> and writes T_<c> under E.OUT
        te44.build(sys.argv[2])
    elif cmd == "probs":
        probs(sys.argv[2], sys.argv[3])
    elif cmd == "combine":
        E.OUT, te44.TAG = P50, sys.argv[2]
        te44.combine(Path(sys.argv[3]), Path(sys.argv[4]))
