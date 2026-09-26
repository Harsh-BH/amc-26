"""P30-bge-veto: let the bge CE reject GBDT-confident (p >= 0.98) pairs it disagrees with.

One SageMaker job (solution_p30.py = P13's job with new phases) scores:
  tgt_India / tgt_US  LOCO: P6's source-only LightGBM on the unseen country's fold 0, every pair with p > 0.02 (band +
                      confident), scored by a bge fine-tuned on the SOURCE country only (P13's LOCO recipe, retrained:
                      P13 never saved its LOCO weights)
  pool_India/pool_US  in-domain: #8c's holdout pairs with p_m2 >= 0.98 and argmax (P23 dump), pooled bge (P13 ckpt)
  pool_France         #9ab's France pairs with p_gbdt >= 0.98 (+ 2,000 control band pairs P13 already scored)

    p30.py build              # data/work/p30/sm_data
    p30.py launch <args...>   # = p13.launch with solution_p30.py
    p30.py fetch <job>        # s3://.../amc26-p30/results/<job>/ -> data/work/p30/
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes/P13-aws-rerank"), str(ROOT / "code/business_entity_resolution/src")]
import p13                                              # also puts P4b's rerank on sys.path
import rerank as rr
from ledger import log

WORK = ROOT / "data/work"
OUT = WORK / "p30"
DATA = OUT / "sm_data"
P6 = WORK / "p30/p6"                    # P6 S_holdout_<src> frames (copied from the p27 offload)
FR = WORK / "p19/prob_test_France_idf144_bge.parquet"
OFF1, OFF2 = 10_000_000, 100_000_000
CI = {"India": 0, "US": 1}
OTHER = {"US": "India", "India": "US"}
PREFIX = "amc26-p30"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _texts(split, c, q1, qx):
    ta, tb = rr.texts(split, c, np.asarray(q1), np.asarray(qx))
    return ta, tb


def loco_frame(src: str) -> pd.DataFrame:
    """P6's source-only model on the target's fold 0: s1, sx (P6 int ids), p, y, am (argmax over every train S1)."""
    tgt = OTHER[src]
    S = pd.read_parquet(P6 / f"S_holdout_{src}.parquet", columns=["s1", "sx", "country", "p", "is_true", "am_all"])
    S = S[S.country == CI[tgt]].drop(columns="country").reset_index(drop=True)
    return S.rename(columns={"is_true": "y", "am_all": "am"})


def build():
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    src = WORK / "p13/sm_data"
    res = {}
    for c in ("India", "US"):                        # train / val = P13's (#5's CE set); holdout = control slice
        for k in ("train", "val"):
            pd.read_parquet(src / f"{k}_{c}.parquet").to_parquet(DATA / f"{k}_{c}.parquet", index=False)
        pd.read_parquet(src / f"holdout_{c}.parquet").sample(2000, random_state=0).to_parquet(
            DATA / f"holdout_{c}.parquet", index=False)
    # LOCO targets: P6 ids are q + ci*OFF, q = P3 text row (features.ids order = rerank.texts order)
    for s in ("US", "India"):
        tgt, ci = OTHER[s], CI[OTHER[s]]
        S = loco_frame(s)
        S = S[S.p > 0.02].reset_index(drop=True)
        q1, qx = S.s1.to_numpy() - ci * OFF1, S.sx.to_numpy() - ci * OFF2
        # id round trip on a sample: P6's q rows -> entity ids -> rerank.rows must give the same rows
        import features as F
        i1, ix = F.ids("train", tgt)
        k = np.random.default_rng(0).choice(len(S), 500, replace=False)
        e1 = pd.Series(i1.take(q1[k]).to_pylist()); ex = pd.Series(ix.take(qx[k]).to_pylist())
        r1, rx = rr.rows("train", tgt, e1, ex)
        assert (r1 == q1[k]).all() and (rx == qx[k]).all(), "P6 ids are not P3 text rows"
        del i1, ix
        ta, tb = _texts("train", tgt, q1, qx)
        pd.DataFrame({"s1": S.s1.to_numpy(), "sx": S.sx.to_numpy(), "ta": ta, "tb": tb,
                      "y": S.y.to_numpy(np.int8), "p": S.p.to_numpy(np.float32)}).to_parquet(
            DATA / f"tgt_{tgt}.parquet", index=False)
        res[f"tgt_{tgt}"] = len(S)
        print(f"tgt_{tgt}: {len(S):,} pairs, {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        del S, ta, tb
    # in-domain: #8c's accepted-candidate pairs above the band edge (P23 dump of #8c's holdout decisions)
    for c in ("India", "US"):
        q = pd.read_parquet(WORK / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p_m2", "am"])
        q = q[q.holdout & q.am & (q.p_m2 >= 0.98)].reset_index(drop=True)
        q1, qx = rr.rows("train", c, q.s1, q.sx)
        ta, tb = _texts("train", c, q1, qx)
        pd.DataFrame({"s1": q.s1.to_numpy(), "sx": q.sx.to_numpy(), "ta": ta, "tb": tb, "y": q.y.to_numpy(np.int8),
                      "p": q.p_m2.to_numpy(np.float32)}).to_parquet(DATA / f"pool_{c}.parquet", index=False)
        res[f"pool_{c}"] = len(q)
        print(f"pool_{c}: {len(q):,} pairs, {time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
        del q, ta, tb
    # France: every p_gbdt >= 0.98 pair of #9ab's file + 2,000 band pairs P13 scored (numerics control)
    d = pd.read_parquet(FR, columns=["s1", "sx", "p_gbdt"])
    hi = d[d.p_gbdt >= 0.98]
    band = d[(d.p_gbdt > 0.02) & (d.p_gbdt < 0.98)]
    ctl = band.merge(pd.read_parquet(WORK / "p13/r/r_bge_pool_test_France.parquet"), on=["s1", "sx"]).sample(
        2000, random_state=0)[["s1", "sx", "p_gbdt"]]
    g = pd.concat([hi, ctl], ignore_index=True)
    del d, band
    q1, qx = rr.rows("test", "France", g.s1, g.sx)
    ta, tb = _texts("test", "France", q1, qx)
    f = pd.DataFrame({"s1": g.s1.to_numpy(), "sx": g.sx.to_numpy(), "ta": ta, "tb": tb,
                      "y": np.zeros(len(g), np.int8), "p": g.p_gbdt.to_numpy(np.float32)})
    old = pd.read_parquet(src / "test_France.parquet", columns=["s1", "sx", "ta", "tb"])
    m = f.iloc[len(hi):].merge(old, on=["s1", "sx"], suffixes=("", "_p13"))
    assert len(m) == 2000 and (m.ta == m.ta_p13).all() and (m.tb == m.tb_p13).all(), "texts differ from P13's"
    f.to_parquet(DATA / "pool_France.parquet", index=False)
    res |= {"pool_France": len(f), "france_hi": len(hi), "france_ctl": 2000}
    print(json.dumps(res), f"{time.time() - t0:.0f}s peak {peak():.2f} GB", flush=True)
    log("exp", probe="P30-bge-veto", subproblem="france-bge-veto", name="p30-build", world="holdout+test",
        metrics=res, config={"loco": "P6 S_holdout_<src> target fold 0, p>0.02", "pool": "P23 #8c holdout am&p_m2>=.98",
                             "france": str(FR.relative_to(ROOT)) + " p_gbdt>=.98 + 2000 ctl"},
        hypothesis="score every GBDT-confident pair so the CE can veto", verdict="keep",
        runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))


def fetch(job: str):
    subprocess.run(["aws", "s3", "sync", f"s3://{p13.BUCKET}/{PREFIX}/results/{job}/", str(OUT), "--exclude", "ckpt/*",
                    "--profile", "harsh", "--region", "ap-south-1"], check=True)
    print(sorted(p.name for p in (OUT / "r").glob("*.parquet")))


if __name__ == "__main__":
    cmd, pos = sys.argv[1], sys.argv[2:]
    if cmd == "launch":
        p13.launch(pos)
    else:
        {"build": build, "fetch": lambda: fetch(*pos)}[cmd]()
