"""P48 prep: attach P45's in-domain r_all (p_r_all, ce_bge_pool on every holdout pair) to P42's LOCO frames, and list
the out-of-band pairs that have no CE score (LOCO g < 0.02 argmax pairs; France pairs with r NaN) for GPU scoring.

    .venv/bin/python probes/P48-fr-ce-all/prep48.py loco      # -> data/work/p48/loco_{dr}_ids.parquet (s1, sx, r_all)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W, OUT = ROOT / "data/work", ROOT / "data/work/p48"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
import features as F  # noqa: E402


def loco():
    OUT.mkdir(parents=True, exist_ok=True)
    for src, tgt in (("US", "India"), ("India", "US")):
        S = pq.read_table(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "am_all"],
                          filters=[("country", "=", CI[tgt]), ("am_all", "=", True)]).to_pandas()
        z = np.load(W / f"p42/loco_{src}_{tgt}.npz")
        assert len(S) == len(z["g"]) and (S.p.to_numpy(np.float64) == z["g"]).all()
        i1, ix = F.ids("train", tgt)                  # P6 ids = P3 text row + ci * OFF (P30's round trip)
        S["q1"], S["qx"] = S.s1.to_numpy() - CI[tgt] * OFF1, S.sx.to_numpy() - CI[tgt] * OFF2
        S["s1"], S["sx"] = i1.take(S.q1.to_numpy()).to_numpy(zero_copy_only=False), ix.take(S.qx.to_numpy()).to_numpy(zero_copy_only=False)
        del i1, ix
        a = pd.read_parquet(W / f"p45/oof/holdout_{tgt}.parquet", columns=["s1", "sx", "p_r_all", "holdout"])
        a = a[a.holdout.to_numpy()].drop(columns="holdout")
        S = S[["s1", "sx", "q1", "qx"]].merge(a, on=["s1", "sx"], how="left")
        assert len(S) == len(z["g"])
        g = z["g"]
        lo = g < 0.02
        print(f"{src}>{tgt}: {len(S):,} rows; r_all coverage all {S.p_r_all.notna().mean():.4f}, low {S.p_r_all[lo].notna().mean():.4f}, "
              f"low true covered {S.p_r_all[lo & z['y']].notna().mean():.4f}", flush=True)
        S.rename(columns={"p_r_all": "r_all"}).to_parquet(OUT / f"loco_{src}_{tgt}_ids.parquet")



def sm():
    """GPU channel data/work/p48/sm/run_<k>/: P45's 64-row stubs + one real test_<c>.parquet (s1, sx, ta, tb, p) each.
    run_India / run_US: LOCO target fold-0 argmax pairs with g < 0.02 (no source-only CE yet) -> scored by P30's
    ce_bge_loco<src>.pt; run_France: every France chain pair with no CE r (france.npz r NaN) -> ce_bge_pool.pt."""
    import shutil
    sys.path.insert(0, str(ROOT / "probes/P4b-rerank"))
    import rerank as rr
    stub = W / "p45/cetest/stub"
    for tgt, src in (("India", "US"), ("US", "India")):
        d = OUT / "sm" / f"run_{tgt}"
        d.mkdir(parents=True, exist_ok=True)
        for f in stub.glob("*.parquet"):
            shutil.copy(f, d / f.name)
        ids = pd.read_parquet(OUT / f"loco_{src}_{tgt}_ids.parquet")
        z = np.load(W / f"p42/loco_{src}_{tgt}.npz")
        k = ids[z["g"] < 0.02].reset_index(drop=True)
        ta, tb = rr.texts("train", tgt, k.q1.to_numpy(), k.qx.to_numpy())
        pd.DataFrame({"s1": k.s1.to_numpy(), "sx": k.sx.to_numpy(), "ta": ta, "tb": tb,
                      "p": z["g"][z["g"] < 0.02].astype(np.float32)}).to_parquet(d / f"test_{tgt}.parquet", index=False)
        print(tgt, len(k), flush=True)
        del ta, tb, k, ids
    d = OUT / "sm" / "run_France"
    d.mkdir(parents=True, exist_ok=True)
    for f in stub.glob("*.parquet"):
        shutil.copy(f, d / f.name)
    z = np.load(W / "p42/france.npz")
    sys.path.insert(0, str(ROOT / "probes/P42-fr-joint-tune"))
    import p42
    i1, ix = p42.fr_ids()
    m = np.isnan(z["r"])
    s1, sx = i1[z["s1"][m]], ix[z["sx"][m]]
    q1, qx = rr.rows("test", "France", pd.Series(s1), pd.Series(sx))
    ta, tb = rr.texts("test", "France", q1, qx)
    pd.DataFrame({"s1": s1, "sx": sx, "ta": ta, "tb": tb, "p": z["g"][m].astype(np.float32)}).to_parquet(d / "test_France.parquet", index=False)
    print("France", int(m.sum()), flush=True)


def fetch(job):
    """GPU job outputs -> data/work/p48/r_lo_{US_India,India_US,France}.parquet (s1, sx, r)."""
    import subprocess
    b = f"s3://sagemaker-114872709824-ap-south-1/amc26-p48/results/{job}"
    for k, c, name in (("India", "India", "r_lo_US_India"), ("US", "US", "r_lo_India_US"), ("France", "France", "r_lo_France")):
        f = OUT / f"{name}.parquet"
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", "--profile", "harsh", f"{b}/{k}/r/r_bge_pool_test_{c}.parquet", str(f)], check=True)
        d = pd.read_parquet(f)
        print(name, len(d), d.columns.tolist(), f"r mean {d.r.mean():.4f}", flush=True)


if __name__ == "__main__":
    {"loco": loco, "sm": sm, "fetch": fetch}[sys.argv[1]](*sys.argv[2:])
