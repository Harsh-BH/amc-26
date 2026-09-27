"""P80 stage 2, laptop side: the channel for the India/US bge CE cross-fit (texts are built on the instance).

    build_ce.py band <c>   # P4b's quick band LightGBM, fit on fold 6, predicts folds 2-5 -> data/work/p80ce/band_<c>.parquet
    build_ce.py keys       # band + P4's X -> s3://.../amc26/p80/ce/data/keys/{A,B}/{train,val,score}_<c>.parquet
    build_ce.py static     # P3 train texts + P13's holdout band files -> s3://.../amc26/p80/ce/data/{txt,holdout}/
    SMOKE_DIR=<scratch> build_ce.py smoke   # tiny channel from the S3 keys, solution_p80ce.main on CPU (e5-small)

Recipe = P13's shipped CE (ce_bge_pool.pt): P4b band (0.1 < p < 0.9 of a quick LightGBM, + 5% of the rest), train =
band + 0.5x as many out-of-band pairs. P4b fit its band model on fold 2, which cannot label fold 2, so the band here is
refit on fold 6 (outside every trained or scored fold). Split A trains on folds {2,3}, B on {4,5}; each scores EVERY
P4 X pair (P3's full candidate set, a superset of any pruned / argmax set) of the other two folds plus folds 0 and 1.
"""
import gc, io, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = ROOT / "data/work"
OUT = W / "p80ce"
X = W / "p4/X"
BUCKET, DATA = "sagemaker-114872709824-ap-south-1", "amc26/p80/ce/data"
ID = ["q1", "qx", "is_true", "fold"]                 # = rerank.ID
LO, HI = 0.1, 0.9                                     # = rerank.LO / HI (P4b's CE training band)
BAND_FIT = 6
SPLITS = {"A": ((2, 3), (4, 5)), "B": ((4, 5), (2, 3))}
EXTRA = (0, 1)                                        # scored by both CEs (seen by neither) -> mean downstream
peak = lambda: __import__("resource").getrusage(0).ru_maxrss / 1e6


def xfiles(c):
    return sorted(X.glob(f"train_{c}_sx_*.parquet"))


def band(c: str, seed: int = 0):
    """= rerank.band with the fit fold moved from 2 to 6 and the scored folds 3-9 -> 2-5."""
    import lightgbm as lgb
    t0, files = time.time(), xfiles(c)
    cols = [x for x in pq.read_schema(files[0]).names if x not in ID]
    Xs, ys = [], []
    for p in files:
        t = pq.read_table(p, columns=cols + ["is_true"], filters=[("fold", "==", BAND_FIT)])
        Xs.append(np.column_stack([t.column(k).to_numpy().astype(np.float32) for k in cols]))
        ys.append(t.column("is_true").to_numpy().astype(np.float32))
        del t
    Xf, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6),
                  lgb.Dataset(Xf, y, feature_name=cols, free_raw_data=True), 300)
    print(f"{c}: band model fit on {len(y):,} fold-{BAND_FIT} pairs {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)
    del Xf, y
    gc.collect()
    rng, out = np.random.default_rng(seed), []
    for p in files:
        d = pq.read_table(p, filters=[("fold", ">=", 2), ("fold", "<=", 5)]).to_pandas()
        pr = m.predict(d[cols].to_numpy(np.float32), num_threads=6).astype(np.float32)
        inb = (pr > LO) & (pr < HI)
        keep = inb | (rng.random(len(d)) < 0.05)
        out.append(d.loc[keep, ID].assign(p=pr[keep], inband=inb[keep]))
        del d, pr
        gc.collect()
    b = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    b.to_parquet(OUT / f"band_{c}.parquet", index=False)
    print(b.groupby("fold").agg(n=("p", "size"), inb=("inband", "sum"), pos=("is_true", "sum")), flush=True)
    print(f"{c}: {int(b.inband.sum()):,} band pairs, {time.time() - t0:.0f}s, peak {peak():.2f} GB", flush=True)


def _put(df: pd.DataFrame, key: str):
    import boto3
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    boto3.Session(profile_name="harsh", region_name="ap-south-1").client("s3").put_object(
        Bucket=BUCKET, Key=f"{DATA}/{key}", Body=buf.getvalue())
    print(f"  s3://{BUCKET}/{DATA}/{key}: {len(df):,} rows, {buf.tell() / 1e6:.1f} MB", flush=True)


def keys():
    """Per split and country: train = fit folds' band + 0.5x out-of-band (P13 build's sampling); val = 20k band pairs of
    the scored folds (a monitor only: P13 logs its AUC, nothing is tuned on it); score = every X pair of the scored
    folds + folds 0, 1 (q1, qx, fold, y)."""
    for c in ("India", "US"):
        b = pd.read_parquet(OUT / f"band_{c}.parquet")
        for s, (fit, sc) in SPLITS.items():
            f = b[b.fold.isin(fit)]
            ib, ob = f[f.inband], f[~f.inband]
            tr = pd.concat([ib, ob.sample(min(len(ob), int(0.5 * len(ib))), random_state=0)])
            v = b[b.fold.isin(sc) & b.inband].sample(20_000, random_state=0)
            for nm, d in (("train", tr), ("val", v)):
                _put(pd.DataFrame({"q1": d.q1.to_numpy(np.int32), "qx": d.qx.to_numpy(np.int32),
                                   "y": d.is_true.to_numpy(np.int8), "p": d.p.to_numpy(np.float32)}), f"keys/{s}/{nm}_{c}.parquet")
            folds = list(sc + EXTRA)
            parts = [pq.read_table(p, columns=ID, filters=[("fold", "in", folds)]).to_pandas() for p in xfiles(c)]
            k = pd.concat(parts, ignore_index=True)
            del parts
            _put(pd.DataFrame({"q1": k.q1.to_numpy(np.int32), "qx": k.qx.to_numpy(np.int32), "fold": k.fold.to_numpy(np.int8),
                               "y": k.is_true.to_numpy(np.int8)}), f"keys/{s}/score_{c}.parquet")
            print(f"{c} {s}: train {len(tr):,} (band {len(ib):,}, folds {fit}), val {len(v):,}, score {len(k):,} "
                  f"(folds {folds}), peak {peak():.2f} GB", flush=True)
            del k, tr, v, f, ib, ob
            gc.collect()


def static():
    aws = ["--profile", "harsh", "--region", "ap-south-1", "--only-show-errors"]
    subprocess.run(["aws", "s3", "sync", str(W / "p3/txt"), f"s3://{BUCKET}/{DATA}/txt/", *aws], check=True)
    for c in ("India", "US"):
        subprocess.run(["aws", "s3", "cp", str(W / f"p13/sm_data/holdout_{c}.parquet"), f"s3://{BUCKET}/{DATA}/holdout/", *aws],
                       check=True)
    subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", f"s3://{BUCKET}/{DATA}/", "--profile", "harsh"], check=True)


def smoke(n: int = 400, split: str = "A"):
    """The exact launch hp (launch_ce.hp) with only the model swapped and S3 writes off; real S3 keys, n rows each."""
    import boto3, shutil
    sys.path[:0] = [str(HERE), str(ROOT / "probes/P13-aws-rerank")]
    import launch_ce
    n = int(n)
    tmp = Path(os.environ.get("SMOKE_DIR", "/tmp/p80ce_smoke"))
    shutil.rmtree(tmp, ignore_errors=True)
    ch = tmp / "train"
    s3 = boto3.Session(profile_name="harsh", region_name="ap-south-1").client("s3")
    get = lambda key: pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{DATA}/{key}")["Body"].read()))
    for c in ("India", "US"):
        for nm in ("train", "val", "score"):
            k = get(f"keys/{split}/{nm}_{c}.parquet")
            (ch / "keys" / split).mkdir(parents=True, exist_ok=True)
            k.sample(min(n, len(k)), random_state=0).to_parquet(ch / "keys" / split / f"{nm}_{c}.parquet", index=False)
        (ch / "holdout").mkdir(exist_ok=True)
        pd.read_parquet(W / f"p13/sm_data/holdout_{c}.parquet").sample(n, random_state=0).to_parquet(
            ch / "holdout" / f"holdout_{c}.parquet", index=False)
    (ch / "txt").mkdir()
    for f in (W / "p3/txt").glob("*.parquet"):
        (ch / "txt" / f.name).symlink_to(f)
    hp = launch_ce.hp(split, "smoke") | {"bge": "intfloat/multilingual-e5-small", "s3": ""}   # S3 writer: P13's, proven
    os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")                          # cached e5-small, no download
    os.environ["HF_HUB_OFFLINE"] = "1"
    import solution_p80ce as S
    S.WORK = tmp / "work"
    S.main(ch, None, tmp / "model", tmp / "out", hp | {"checkpoint_dir": str(tmp / "ck")})
    for c in ("India", "US"):
        o = pd.read_parquet(tmp / "model" / "oof" / f"oof_{split}_{c}.parquet")
        k = pd.read_parquet(ch / "keys" / split / f"score_{c}.parquet")
        assert len(o) == len(k) and (o.q1.to_numpy() == k.q1.to_numpy()).all() and o.r.between(0, 1).all(), c
        print(c, o.shape, o.columns.tolist(), o.head(2).to_dict("records"))
    print(f"smoke ok, peak {peak():.2f} GB")


if __name__ == "__main__":
    {"band": band, "keys": keys, "static": static, "smoke": smoke}[sys.argv[1]](*sys.argv[2:])
