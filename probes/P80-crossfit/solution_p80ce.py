"""P80 stage 2 (SageMaker GPU job): one half of the India/US bge CE cross-fit. Builds P3's "name | addr" texts for the
split's keys on the instance, then runs P13's shipped recipe unchanged (p13sol.py = solution_p13.py, phase bge_pool:
2 epochs, bs 128, lr 2e-5, max_len 96, bf16) and scores every key pair.

Channel 'train' = s3://.../amc26/p80/ce/data/ (build_ce.py): keys/<split>/{train,val,score}_<c>.parquet (q1, qx = rows
of P3's text files; train/val: y, p; score: fold, y), txt/ (P3's train texts), holdout/ (P13's fold-0 band files).
hp: split (A | B) + P13's (phases, s3, deadline_s, speed).
-> P13's files in hp s3 (r/r_bge_pool_holdout.parquet, r/r_bge_pool_test_<c>.parquet, ckpt/ce_bge_pool.pt, log.jsonl)
   + oof/oof_<split>_<c>.parquet: every score key (q1, qx, fold, y) with its s1, sx entity ids and r = sigmoid(logit).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

WORK = Path("/tmp/p80ce")


def lookup(txt: Path, c: str, q1: np.ndarray, qx: np.ndarray, cols: list, f):
    """(f(S1 row q1), f(SX row qx)) over P3's text files, SX files concatenated in sorted order = rerank.texts."""
    u1, i1 = np.unique(q1, return_inverse=True)
    a = f(pq.read_table(txt / f"{c}_s1.parquet", columns=cols).take(u1))[i1]
    ux, ix = np.unique(qx, return_inverse=True)
    tx, off = np.empty(len(ux), dtype=object), 0
    for p in sorted(txt.glob(f"{c}_sx_*.parquet")):
        n = pq.ParquetFile(p).metadata.num_rows
        lo, hi = np.searchsorted(ux, [off, off + n])
        if hi > lo:
            tx[lo:hi] = f(pq.read_table(p, columns=cols).take(ux[lo:hi] - off))
        off += n
    assert len(ux) == 0 or ux[-1] < off, "every qx inside the text files"
    return a, tx[ix]


def join(t: pa.Table) -> np.ndarray:                     # = rerank._join
    return np.array(pc.binary_join_element_wise(t["name"], t["addr"], pa.scalar(" | ", pa.large_string())).to_pylist(), dtype=object)


def ids(t: pa.Table) -> np.ndarray:
    return np.array(t["entity_id"].to_pylist(), dtype=object)


def build(src: Path, split: str) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    txt, kd, score = src / "txt", src / "keys" / split, {}
    for c in ("India", "US"):
        for nm in ("train", "val"):
            k = pd.read_parquet(kd / f"{nm}_{c}.parquet")
            ta, tb = lookup(txt, c, k.q1.to_numpy(), k.qx.to_numpy(), ["name", "addr"], join)
            pd.DataFrame({"ta": ta, "tb": tb, "y": k.y.to_numpy(np.int8), "p": k.p.to_numpy(np.float32)}).to_parquet(
                WORK / f"{nm}_{c}.parquet", index=False)
            print(f"[p80ce] {nm}_{c}: {len(k):,} pairs, true {k.y.mean():.3f}", flush=True)
        k = pd.read_parquet(kd / f"score_{c}.parquet")
        q1, qx = k.q1.to_numpy(), k.qx.to_numpy()
        s1, sx = lookup(txt, c, q1, qx, ["entity_id"], ids)
        ta, tb = lookup(txt, c, q1, qx, ["name", "addr"], join)
        pd.DataFrame({"s1": s1, "sx": sx, "ta": ta, "tb": tb}).to_parquet(WORK / f"test_{c}.parquet", index=False)
        score[c] = k.assign(s1=s1, sx=sx)
        print(f"[p80ce] test_{c}: {len(k):,} score pairs, folds {sorted(k.fold.unique().tolist())}", flush=True)
        del ta, tb
        shutil.copy(src / "holdout" / f"holdout_{c}.parquet", WORK / f"holdout_{c}.parquet")
    pd.read_parquet(WORK / "holdout_India.parquet").head(64)[["s1", "sx", "ta", "tb", "p"]].to_parquet(
        WORK / "test_France.parquet", index=False)                  # P13's bge_pool also scores France: a stub
    return score


def main(train_dir, test_dir, model_dir, out_dir, params):
    try:
        import p13sol                                   # shipped next to this file (launch_ce.py)
    except ModuleNotFoundError:
        import solution_p13 as p13sol                   # laptop smoke
    split = params["split"]
    assert split in ("A", "B") and params.get("phases") == ["bge_pool"], params
    score = build(Path(train_dir), split)
    p13sol.main(WORK, test_dir, model_dir, out_dir, params)
    s3 = p13sol.S3(params.get("s3", ""))
    (Path(model_dir) / "oof").mkdir(parents=True, exist_ok=True)
    for c, k in score.items():
        f = Path(model_dir) / "r" / f"r_bge_pool_test_{c}.parquet"
        if not f.exists():                              # skipped by P13's deadline planner: logged in log.jsonl
            print(f"[p80ce] no r for {c}", flush=True)
            continue
        r = pd.read_parquet(f)
        assert len(r) == len(k) and (r.s1.to_numpy() == k.s1.to_numpy()).all() and (r.sx.to_numpy() == k.sx.to_numpy()).all()
        g = Path(model_dir) / "oof" / f"oof_{split}_{c}.parquet"
        k.assign(r=r.r.to_numpy(np.float32)).to_parquet(g, index=False)
        s3.put(g, f"oof/{g.name}")
        print(f"[p80ce] {g.name}: {len(k):,} rows", flush=True)
