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
BUCKET, DATA = os.environ.get("BUCKET", "your-bucket"), "amc26/p80/ce/data"
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
    boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3").put_object(
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
    aws = ["--profile", os.environ.get("AWS_PROFILE", "default"), "--region", "ap-south-1", "--only-show-errors"]
    subprocess.run(["aws", "s3", "sync", str(W / "p3/txt"), f"s3://{BUCKET}/{DATA}/txt/", *aws], check=True)
    for c in ("India", "US"):
        subprocess.run(["aws", "s3", "cp", str(W / f"p13/sm_data/holdout_{c}.parquet"), f"s3://{BUCKET}/{DATA}/holdout/", *aws],
                       check=True)
    subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", f"s3://{BUCKET}/{DATA}/", "--profile", os.environ.get("AWS_PROFILE", "default")], check=True)


def smoke(n: int = 400, split: str = "A"):
    """The exact launch hp (launch_ce.hp) with only the model swapped and S3 writes off; real S3 keys, n rows each."""
    import boto3, shutil
    sys.path[:0] = [str(HERE), str(ROOT / "probes/P13-aws-rerank")]
    import launch_ce
    n = int(n)
    tmp = Path(os.environ.get("SMOKE_DIR", "/tmp/p80ce_smoke"))
    shutil.rmtree(tmp, ignore_errors=True)
    ch = tmp / "train"
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3")
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


SDATA = "amc26/p80/ce/score_data"                    # channel of the checkpoint-scoring jobs (mode 'score')


def score_channel():
    """Stage C's missing rows -> s3://.../score_data/: keys/<c>/train69 = sc_<c> frame rows of folds 6-9 (q1, qx, fold),
    keys/<c>/test = every test X pair with cross-fit M2 p >= stagec.PMIN (p80-sct files, = stagec.apply_test's set);
    txt/ (train texts, server-side copy), txt_test/ (P3's India/US test texts), holdout/ (2k-pair samples)."""
    import boto3
    PMIN = 0.01                                        # = stagec.PMIN (checked below against its source)
    assert "PMIN, ANC, T_BASE = 0.01," in (HERE / "stagec.py").read_text()
    global DATA
    DATA, s3 = SDATA, boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3")
    sct = ROOT / "out/p80-sct-g4dn16xlarge-0927-2143"
    for c in ("India", "US"):
        t = pq.read_table(W / f"p80/sc_{c}.parquet", columns=["q1", "qx", "fold"], filters=[("fold", ">=", 6)]).to_pandas()
        _put(t.astype({"q1": np.int32, "qx": np.int32, "fold": np.int8}), f"keys/{c}/train69.parquet")
        parts = []
        for xf in sorted(X.glob(f"test_{c}_sx_*.parquet")):
            (tp,) = sorted(sct.rglob(f"*{xf.name}"))
            o = pq.read_table(tp, columns=["q1", "qx", "p"]).to_pandas()
            assert len(o) == pq.read_metadata(xf).num_rows
            parts.append(o.loc[o.p.to_numpy() >= PMIN, ["q1", "qx"]].astype(np.int32))
        _put(pd.concat(parts, ignore_index=True), f"keys/{c}/test.parquet")
        _put(pd.read_parquet(W / f"p13/sm_data/holdout_{c}.parquet").sample(2000, random_state=0), f"holdout/holdout_{c}.parquet")
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="amc26/p80/ce/data/txt/")["Contents"]:
        s3.copy_object(Bucket=BUCKET, Key=f"{SDATA}/txt/{Path(o['Key']).name}", CopySource={"Bucket": BUCKET, "Key": o["Key"]})
    aws = ["--profile", os.environ.get("AWS_PROFILE", "default"), "--region", "ap-south-1", "--only-show-errors"]
    subprocess.run(["aws", "s3", "sync", str(W / "p3/txt_test"), f"s3://{BUCKET}/{SDATA}/txt_test/", "--exclude", "France*", *aws],
                   check=True)
    subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", f"s3://{BUCKET}/{SDATA}/", "--profile", os.environ.get("AWS_PROFILE", "default")], check=True)


def smoke_score(n: int = 300):
    """Score mode on n keys per set (train69 + test, one slice), two local e5-small 'checkpoints' (P13 save format)."""
    import boto3, shutil, torch
    sys.path[:0] = [str(HERE), str(ROOT / "probes/P13-aws-rerank"), str(ROOT / "probes/P4b-rerank")]
    import launch_ce, rerank, solution_p13 as P, solution_p80ce as S
    n, tmp = int(n), Path(os.environ["SMOKE_DIR"])
    shutil.rmtree(tmp, ignore_errors=True)
    ch = tmp / "train"
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3")
    get = lambda key: pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{SDATA}/{key}")["Body"].read()))
    for c in ("India", "US"):
        (ch / "keys" / c).mkdir(parents=True)
        for nm in ("train69", "test"):
            get(f"keys/{c}/{nm}.parquet").sample(n, random_state=0).to_parquet(ch / "keys" / c / f"{nm}.parquet", index=False)
    (ch / "holdout").mkdir()
    for c in ("India", "US"):
        get(f"holdout/holdout_{c}.parquet").to_parquet(ch / "holdout" / f"holdout_{c}.parquet", index=False)
    for d, src in (("txt", W / "p3/txt"), ("txt_test", W / "p3/txt_test")):
        (ch / d).mkdir()
        for f in src.glob("*.parquet"):
            (ch / d / f.name).symlink_to(f)
    os.environ.update(HF_HOME=str(Path.home() / ".cache/huggingface"), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    S.WORK = tmp / "work"
    ck = {}
    for nm, seed in (("A", 0), ("B", 1)):                      # two different random heads = two distinct "checkpoints"
        torch.manual_seed(seed)
        job = P.Job(ch, tmp / "mk", {"bge": "intfloat/multilingual-e5-small", "checkpoint_dir": str(tmp / "ck")})
        _, sc, _ = job.load("bge")
        frozen = {k for k, p in sc.named_parameters() if not p.requires_grad}
        ck[nm] = str(tmp / f"ce_{nm}.pt")
        torch.save({k: v.half() for k, v in sc.state_dict().items() if k not in frozen}, ck[nm])
        del job, sc
    for c in ("India", "US"):
        hp = launch_ce.score_hp(c, 0, 1, "smoke") | {"bge": "intfloat/multilingual-e5-small", "s3": "", "ckpts": ck}
        S.main(ch, None, tmp / f"model_{c}", tmp / "out", hp | {"checkpoint_dir": str(tmp / "ck")})
        o = pd.read_parquet(tmp / f"model_{c}" / "score" / f"score_{c}_0of1.parquet")
        assert len(o) == 2 * n and (o.fold == -1).sum() == n and (o.r_A != o.r_B).any() and o.r.between(0, 1).all(), c
        for split, m in (("train", o.fold >= 6), ("test", o.fold == -1)):
            q1, qx = rerank.rows(split, c, o.s1[m], o.sx[m])
            assert (q1 == o.q1[m].to_numpy()).all() and (qx == o.qx[m].to_numpy()).all(), f"{split} ids = P3 rows"
            ta, tb = rerank.texts(split, c, q1, qx)
            w = pd.read_parquet(S.WORK / f"test_{c}.parquet")[m.to_numpy()]
            assert (w.ta.to_numpy() == ta).all() and (w.tb.to_numpy() == tb).all(), f"{split} texts = rerank.texts"
        print(c, o.shape, o.columns.tolist(), o.head(2).to_dict("records"))
    print(f"smoke_score ok, peak {peak():.2f} GB")


def check_set(c: str, *jobs):
    """A complete slice set of one country (jobs in slice order k = 0..n-1): concatenated score files == the channel's
    keys row for row (train69 then test, fold -1), r_A / r_B / r finite in (0, 1). Streams from S3, nothing on disk."""
    import boto3
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3")
    rd = lambda key, cols=None: pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()), columns=cols)
    K = pd.concat([rd(f"{SDATA}/keys/{c}/train69.parquet"), rd(f"{SDATA}/keys/{c}/test.parquet").assign(fold=np.int8(-1))],
                  ignore_index=True)
    n, parts = len(jobs), []
    for k, j in enumerate(jobs):
        parts.append(rd(f"amc26/p80/ce/results/{j}/score/score_{c}_{k}of{n}.parquet", ["q1", "qx", "fold", "r_A", "r_B", "r"]))
    o = pd.concat(parts, ignore_index=True)
    ok = len(o) == len(K) and all((o[x].to_numpy() == K[x].to_numpy()).all() for x in ("q1", "qx", "fold"))
    fin = all(np.isfinite(o[x]).all() and o[x].between(0, 1).all() for x in ("r_A", "r_B", "r"))
    print(f"{c} {n}-way: {len(o):,} rows vs {len(K):,} keys, exact key match {ok}, r finite in [0,1] {fin}, "
          f"corr logit A/B {np.corrcoef(*(np.log(np.clip(o[x], 1e-6, 1 - 1e-6) / (1 - np.clip(o[x], 1e-6, 1 - 1e-6))) for x in ('r_A', 'r_B')))[0, 1]:.4f}",
          flush=True)
    assert ok and fin


def assemble(c: str, dest: str = "amc26/p80/ce/results/p80cs-combined/score"):
    """Union of every finished score_<c>_<k>of<n> slice (any mix of n) over all p80cs-<c>* jobs. Slice k of n = key rows
    [N*k//n, N*(k+1)//n) of train69 + test (score_main's slicing). If the union covers all N keys: build the combined
    file (each row once), assert it equals the keys on q1, qx, fold, and upload <dest>/score_<c>.parquet from memory.
    Returns the S3 uri, or None (printing the coverage) when a gap remains."""
    import re, boto3
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1").client("s3")
    rd = lambda key, cols=None: pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()), columns=cols)
    N = sum(pq.read_metadata(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{SDATA}/keys/{c}/{nm}.parquet")["Body"].read())).num_rows
            for nm in ("train69", "test"))
    iv = []                                                           # (lo, hi, key)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"amc26/p80/ce/results/p80cs-{c}"):
        for o in page.get("Contents", []):
            m = re.search(rf"/score/score_{c}_(\d+)of(\d+)\.parquet$", o["Key"])
            if m:
                k, n = int(m[1]), int(m[2])
                iv.append((N * k // n, N * (k + 1) // n, o["Key"]))
    iv.sort()
    cols, got = ["q1", "qx", "fold", "r_A", "r_B", "r"], {}
    while True:
        pos, pick = 0, []                                             # greedy interval cover of [0, N)
        while pos < N:
            best = max((x for x in iv if x[0] <= pos), key=lambda x: x[1], default=None)
            if best is None or best[1] <= pos:
                print(f"{c}: {pos:,} of {N:,} keys covered from row 0 ({pos / N:.1%}); slices done: "
                      f"{[re.search(r'p80cs-[^/]+', x[2])[0] + ':' + re.search(r'score_.*', x[2])[0] for x in iv]}", flush=True)
                return None
            pick.append((pos, best))
            pos = best[1]
        bad = []
        for _, x in pick:                                             # a slice counts only if its r are sane
            if x[2] not in got:
                d = rd(x[2], cols)
                ok = len(d) == x[1] - x[0] and all(np.isfinite(d[q]).all() and d[q].between(0, 1).all() for q in cols[3:])
                got[x[2]] = d if ok else None
            if got[x[2]] is None:
                bad.append(x)
        if not bad:
            break
        print(f"{c}: dropping bad slices {[x[2] for x in bad]}", flush=True)
        iv = [x for x in iv if x not in bad]
    out = pd.concat([got[key].iloc[a - lo: hi - lo] for a, (lo, hi, key) in pick], ignore_index=True)
    K = pd.concat([rd(f"{SDATA}/keys/{c}/train69.parquet"), rd(f"{SDATA}/keys/{c}/test.parquet").assign(fold=np.int8(-1))],
                  ignore_index=True)
    assert len(out) == N == len(K) and all((out[x].to_numpy() == K[x].to_numpy()).all() for x in ("q1", "qx", "fold"))
    assert all(np.isfinite(out[x]).all() and out[x].between(0, 1).all() for x in ("r_A", "r_B", "r"))
    _put_key = f"{dest}/score_{c}.parquet"
    buf = io.BytesIO()
    out.to_parquet(buf, index=False, compression="zstd")
    s3.put_object(Bucket=BUCKET, Key=_put_key, Body=buf.getvalue())
    print(f"{c}: COVERED {N:,} keys from {[re.search(r'p80cs-[^/]+', k)[0] + ':' + re.search(r'score_.*', k)[0] for _, (_, _, k) in pick]} "
          f"-> s3://{BUCKET}/{_put_key} ({buf.tell() / 1e6:.0f} MB)", flush=True)
    return f"s3://{BUCKET}/{_put_key}"


def _done_iv(s3, c, N):
    import re
    iv = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"amc26/p80/ce/results/p80cs-{c}"):
        for o in page.get("Contents", []):
            if m := re.search(rf"/score/score_{c}_(\d+)of(\d+)\.parquet$", o["Key"]):
                iv.append((N * int(m[1]) // int(m[2]), N * (int(m[1]) + 1) // int(m[2])))
    return iv


def sched(queue: str = "3/8,10/16,11/16,12/16,13/16,14/16,15/16", types: str = "ml.g6e.xlarge,ml.g5.4xlarge,ml.g5.2xlarge,"
          "ml.g4dn.12xlarge,ml.g4dn.16xlarge,ml.g6.4xlarge,ml.g6.2xlarge,ml.g5.xlarge,ml.g6.xlarge,ml.g4dn.xlarge"):
    """Slot-driven India launcher (account cap: 15 training instances, Pending included): every 60 s, while a slot is free,
    launch the next queued k/n slice not yet covered by finished slices on the first preferred GPU type with no job of
    ours in progress; run assemble() for India and US until both are covered. Prints one line per event."""
    import boto3
    ses = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name="ap-south-1")
    sm, s3 = ses.client("sagemaker"), ses.client("s3")
    N = sum(pq.read_metadata(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{SDATA}/keys/India/{nm}.parquet")["Body"].read())).num_rows
            for nm in ("train69", "test"))
    Q = [tuple(map(int, x.split("/"))) for x in queue.split(",")]
    todo, last = {"India": True, "US": True}, {}
    while any(todo.values()):
        live = [j["TrainingJobName"] for j in sm.list_training_jobs(StatusEquals="InProgress", MaxResults=100)["TrainingJobSummaries"]]
        busy = {sm.describe_training_job(TrainingJobName=j)["ResourceConfig"]["InstanceType"] for j in live}
        iv = _done_iv(s3, "India", N)

        def covered(lo, hi):                                          # [lo, hi) inside the union of finished slices
            pos = lo
            for a, b in sorted(iv):
                if a <= pos < b:
                    pos = b
            return pos >= hi
        while Q and len(live) < 15:
            k, n = Q[0]
            if covered(N * k // n, N * (k + 1) // n):
                print(f"{time.strftime('%H:%M')} skip India {k}of{n}: already covered", flush=True)
                Q.pop(0)
                continue
            free = [t for t in types.split(",") if t not in busy]
            if not free:
                break
            r = subprocess.run([sys.executable, str(HERE / "launch_ce.py"), "score", "India", str(k), str(n), free[0]],
                               capture_output=True, text=True)
            msg = next((x for x in (r.stdout + r.stderr).splitlines() if "submitted" in x), (r.stdout + r.stderr)[-300:])
            print(f"{time.strftime('%H:%M')} India {k}of{n} on {free[0]}: {msg.strip()}", flush=True)
            busy.add(free[0])
            live.append(free[0])
            if "submitted" in msg:
                Q.pop(0)
            else:
                break
        for c in [c for c, t in todo.items() if t]:
            buf = io.StringIO()
            with __import__("contextlib").redirect_stdout(buf):
                uri = assemble(c)
            if buf.getvalue() != last.get(c):                         # only changes become monitor events
                print(f"{time.strftime('%H:%M')} {buf.getvalue().strip()}", flush=True)
                last[c] = buf.getvalue()
            if uri:
                todo[c] = False
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1] == "sched":
        sched(*sys.argv[2:])
        sys.exit()
    if sys.argv[1] == "assemble":
        sys.exit(0 if assemble(*sys.argv[2:]) else 3)
    if sys.argv[1] == "check_set":
        check_set(*sys.argv[2:])
        sys.exit()
    {"band": band, "keys": keys, "static": static, "smoke": smoke, "score_channel": score_channel,
     "smoke_score": smoke_score}[sys.argv[1]](*sys.argv[2:])
