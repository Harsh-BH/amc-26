"""P19-bge-idf: P13's bge CE as a second opinion on P14's IDF-fixed France candidates (idf144).
The SageMaker job is P13's solution_p13.py, unchanged (phase bge_ckpt); only its data channel is ours.

    p19.py build             # data/work/p19/sm_data: test_France = idf144 band pairs without a P13 r (+ controls)
    p19.py launch <args...>  # = p13.py launch (P13's 3-file bundle + pinned reqs)
    p19.py fetch <job>       # s3://.../amc26-p19/results/<job>/ -> data/work/p19/ (never into p13/)
    p19.py check             # the job's r on the control pairs vs P13's r (same numerics?)
    p19.py blend             # idf144 p blended with bge r (P13's France recipe) -> p19/prob_test_France_idf144_bge
    p19.py profile           # P14's label-free profile of the blended file (t_eq_d4 = D4's empty-S1 rate)
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank"), str(ROOT / "probes/P14-france-shift")]
import p13                                              # also puts P4b's rerank on sys.path
import rerank as rr
from ledger import log

WORK = ROOT / "data/work"
OUT = WORK / "p19"
DATA = OUT / "sm_data"
IDF = WORK / "p14/prob_test_France_idf144.parquet"
RFR = WORK / "p13/r/r_bge_pool_test_France.parquet"
FINAL = OUT / "prob_test_France_idf144_bge.parquet"
LO, HI = p13.LO, p13.HI                                # 0.02, 0.98
W_FR = [0.9245, 0.2595, 0.0515]                        # P13's label-free out-of-domain France weights
N_CTL = {"holdout": 2000, "val": 200, "test": 500, "test_France": 2000}
PREFIX = "amc26-p19"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def band_r() -> pd.DataFrame:
    """idf144's band pairs (row order kept) with P13's r where it exists (NaN = not scored yet)."""
    d = pd.read_parquet(IDF)
    b = d[(d.p > LO) & (d.p < HI)][["s1", "sx", "p"]]
    return b.merge(pd.read_parquet(RFR), on=["s1", "sx"], how="left")


def build():
    """Every file bge_ckpt reads, P13's schema. test_France = all missing band pairs + N control pairs P13 already
    scored; holdout / val / test_India / test_US = small slices of P13's own files (controls, and the job needs them)."""
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    src = WORK / "p13/sm_data"
    for c in ("India", "US"):
        for k, n in (("holdout", N_CTL["holdout"]), ("val", N_CTL["val"]), ("test", N_CTL["test"])):
            pd.read_parquet(src / f"{k}_{c}.parquet").sample(n, random_state=0).to_parquet(DATA / f"{k}_{c}.parquet", index=False)
    b = band_r()
    miss, have = b[b.r.isna()], b[b.r.notna()].sample(N_CTL["test_France"], random_state=0)
    g = pd.concat([miss, have], ignore_index=True)
    q1, qx = rr.rows("test", "France", g.s1, g.sx)
    ta, tb = rr.texts("test", "France", q1, qx)
    f = pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb, "p": g.p.to_numpy(np.float32)})
    # our texts for the control pairs must be the ones P13 scored (same P3 text files; idf144 changes no text)
    old = pd.read_parquet(src / "test_France.parquet", columns=["s1", "sx", "ta", "tb"])
    m = f.iloc[len(miss):].merge(old, on=["s1", "sx"], suffixes=("", "_p13"))
    assert len(m) == len(have) and (m.ta == m.ta_p13).all() and (m.tb == m.tb_p13).all(), "texts differ from P13's"
    f.to_parquet(DATA / "test_France.parquet", index=False)
    res = {"band": len(b), "have_r": int(b.r.notna().sum()), "missing": len(miss), "control": len(have),
           "test_France_rows": len(f), "text_check": "ok"}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe="P19-bge-idf", subproblem="france-bge-idf", name="p19-build", world="test-France",
        metrics=res, config={"src": str(IDF.relative_to(ROOT)), "r": str(RFR.relative_to(ROOT)), "band": [LO, HI],
                             "controls": N_CTL}, hypothesis="the IDF fix changes France's candidate list, so part of "
        "its band has no bge score yet; score only those", verdict="keep", runtime_s=round(time.time() - t0),
        peak_gb=round(peak(), 2))


def fetch(job: str):
    subprocess.run(["aws", "s3", "sync", f"s3://{p13.BUCKET}/{PREFIX}/results/{job}/", str(OUT), "--exclude", "ckpt/*",
                    "--profile", "harsh", "--region", "ap-south-1"], check=True)
    print(sorted(p.name for p in (OUT / "r").glob("*.parquet")))


def check():
    """Control pairs: the job's r vs P13's r for the same pairs (P13's restored model reproduced its own job at
    median |dlogit| 0, p99 0.063)."""
    R = OUT / "r"
    res = {}
    pairs = [("holdout", "r_bge_poolck_holdout", "r_bge_poolck_holdout")] + \
            [(c, f"r_bge_pool_test_{c}", f"r_bge_pool_test_{c}") for c in ("India", "US", "France")]
    for k, new, old in pairs:
        m = pd.read_parquet(R / f"{new}.parquet").merge(pd.read_parquet(WORK / f"p13/r/{old}.parquet"),
                                                        on=["s1", "sx"], suffixes=("", "_p13"))
        dz = np.abs(rr.logit(m.r.to_numpy()) - rr.logit(m.r_p13.to_numpy()))
        res[k] = {"n": len(m), "corr": round(float(np.corrcoef(m.r, m.r_p13)[0, 1]), 6),
                  "med_dlogit": round(float(np.median(dz)), 4), "p99_dlogit": round(float(np.quantile(dz, 0.99)), 4)}
    print(json.dumps(res), flush=True)
    ok = all(v["corr"] > 0.999 and v["p99_dlogit"] < 0.2 for v in res.values()) and res["France"]["n"] == N_CTL["test_France"]
    log("exp", probe="P19-bge-idf", subproblem="france-bge-idf", name="p19-sm-repro", world="test+holdout",
        metrics=res, config={"job_r": str(R.relative_to(ROOT))}, hypothesis="the job reproduces P13's r on control pairs",
        verdict="keep" if ok else "drop", runtime_s=0, peak_gb=round(peak(), 2))
    assert ok, res


def blend():
    """P13's France recipe on idf144: r = P13's r, else the job's r; p' = rr.blend(p, r, W_FR, 0.02, 0.98).
    Same rows and order as idf144; outside the band p is idf144's p exactly."""
    t0 = time.time()
    d = pd.read_parquet(IDF)
    new = pd.read_parquet(OUT / "r/r_bge_pool_test_France.parquet")
    r_all = pd.concat([pd.read_parquet(RFR), new], ignore_index=True).drop_duplicates(["s1", "sx"])
    g = d.p.to_numpy(np.float32)
    band = (g > LO) & (g < HI)
    r = np.full(len(d), np.nan, np.float32)
    r[band] = d[band][["s1", "sx"]].merge(r_all, on=["s1", "sx"], how="left").r.to_numpy()
    assert not np.isnan(r[band]).any(), f"{int(np.isnan(r[band]).sum())} band pairs without r"
    p = rr.blend(g, r, W_FR, LO, HI)
    assert (p[~band] == g[~band]).all() and len(p) == len(d)
    out = pd.DataFrame({"s1": d.s1, "sx": d.sx, "p": p, "p_gbdt": g})
    out.to_parquet(FINAL, index=False)
    dis = float(((g[band] >= 0.5) != (r[band] >= 0.5)).mean())
    res = {"pairs": len(d), "band": int(band.sum()), "r_from_p13": int(band.sum()) - len(new.merge(d[band][["s1", "sx"]])),
           "r_from_p19": len(new.merge(d[band][["s1", "sx"]])), "ce_gbdt_disagree": round(dis, 4),
           "corr_band_p_vs_blend": round(float(np.corrcoef(g[band], p[band])[0, 1]), 4)}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe="P19-bge-idf", subproblem="france-bge-idf", name="p19-blend-idf144-bge", world="test-France",
        metrics=res, config={"w": W_FR, "band": [LO, HI], "out": str(FINAL.relative_to(ROOT))},
        hypothesis="P13's France bge blend (#9b) on top of P14's IDF-fixed candidates (#9a)", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def profile():
    """compare.profile on FINAL vs #6's France (t_eq_d4, band, churn), plus churn vs idf144 alone at its t 0.975."""
    import compare as C
    t0 = time.time()
    C.prob = lambda name, _p=C.prob: pd.read_parquet(FINAL) if name == "idf144_bge" else _p(name)
    o = C.profile("idf144_bge", C.prob("sub6"))
    a = C.decide(C.prob("idf144_bge"), o["t_eq_d4"])
    A = a.groupby("s1").sx.apply(frozenset).to_dict()
    # churn vs #9a (idf144 at 0.975) and #9b (P13's bge blend on the old candidates at 0.98)
    for k, b in (("9a", C.decide(C.prob("idf144"), 0.975)),
                 ("9b", C.decide(pd.read_parquet(WORK / "p13/prob_test_France_blend.parquet"), 0.98))):
        B = b.groupby("s1").sx.apply(frozenset).to_dict()
        o[f"s1_changed_vs_{k}"] = sum(A.get(s, frozenset()) != B.get(s, frozenset()) for s in A.keys() | B.keys())
    o = {k: round(v, 5) if isinstance(v, float) else v for k, v in o.items()}
    print(json.dumps(o), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe="P19-bge-idf", subproblem="france-bge-idf", name="profile-idf144_bge", world="test-France",
        metrics=o, config={"file": str(FINAL.relative_to(ROOT)), "d4_empty": C.D4_EMPTY},
        hypothesis="label-free profile of the blended France file; t_eq_d4 = the France threshold",
        verdict="keep", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    if cmd == "launch":
        p13.launch(pos)
    else:
        {"build": build, "fetch": lambda: fetch(*pos), "check": check, "blend": blend, "profile": profile}[cmd]()
