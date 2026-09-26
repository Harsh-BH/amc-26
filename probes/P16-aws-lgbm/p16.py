"""P16-aws-lgbm, laptop side. The SageMaker job itself is solution_p16.py (self-contained, shipped as solution.py).

    p16.py check                 # solution_p16.load == train_matcher.load (float16 train folds, float32 fold-1 val)
    p16.py tiny                  # 1% S1-hash slice of every train X file -> data/work/p16/sm_tiny/p4/X/
    p16.py rehearse              # sm_train.py -> solution.py on the slice, SageMaker env + hp encoding, S3 writes
    p16.py launch                # ONE job: ml.m7i.4xlarge on-demand, s3://.../amc26-p16/data/v1/, MaxRuntime 4 h
    p16.py status [job]          # status.json the job writes to S3 + DescribeTrainingJob
    p16.py fetch [job | ckpt]    # m2_p16.txt -> data/work/p16/m2_p16.txt; ckpt: latest checkpoint -> p16/ckpt/
    p16.py eval [booster ...]    # #7 yardstick vs m2_big (see evaluate)
    p16.py france [booster]      # France test profile (empty-S1 rate, matches / matched S1) vs m2_big
    p16.py test                  # adopted booster: pruned test scores -> data/work/p16/prob_test_<c>.parquet
"""
import gc, io, json, os, resource, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "code/business_entity_resolution/src"
sys.path[:0] = [str(HERE), str(SRC), str(ROOT)]
import solution_p16 as S

WORK = ROOT / "data/work"
OUT = WORK / "p16"
TINY = OUT / "sm_tiny"
BOOSTER = OUT / "m2_p16.txt"
BUCKET, PREFIX, REGION = "sagemaker-114872709824-ap-south-1", "amc26-p16", "ap-south-1"
DATA_S3 = f"s3://{BUCKET}/{PREFIX}/data/v1/"
INSTANCE, MAX_RUN = "ml.m7i.4xlarge", 14400        # 16 vCPU / 64 GB, $1.018/h on-demand (Pricing API) -> <= $4.07
JOB_FILE = OUT / "job.txt"
CE7 = WORK / "p11" / "ce_s2P.pt"                  # #7's cross-encoder (P9 sub7 recipe)
W7, T7 = [0.617, 0.666, 0.261], 0.71              # #7's blend w and t (sub7)
TAU = 0.03
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
os.environ.setdefault("AWS_PROFILE", "harsh")


def _boto(svc):
    import boto3
    return boto3.Session(profile_name="harsh", region_name=REGION).client(svc)


def hp(**kw) -> dict:
    import model as M
    return {"params": M.PARAMS, "folds": [2, 3, 4, 5, 6, 7, 8, 9], "rounds": 8000, "es": 100, "val_frac": 0.25} | kw


# ---------------------------------------------------------------- job correctness

def check():
    """The job's pyarrow loader against train_matcher.load itself (two real files, 30% S1 sample)."""
    import features as F
    import model as M
    import train_matcher as TM
    small = {"India": [F.xfiles("train", "India")[-1]], "US": [F.xfiles("train", "US")[-1]]}
    TM.F.xfiles = lambda split, c: small[c]
    fs = small["India"] + small["US"]
    cols = [c for c in pq.read_schema(fs[0]).names if c not in S.DROP]
    assert S.DROP == M.DROP
    for folds, frac, dt in (((2, 3, 4, 5, 6, 7, 8, 9), 0.3, np.float16), ((1,), 0.25, np.float32)):
        Xa, ya, ca = TM.load(folds, frac=frac, dtype=dt)
        Xb, yb = S.load(fs, cols, folds, frac=frac, dtype=dt)
        assert ca == cols and Xa.dtype == Xb.dtype and Xa.shape == Xb.shape and len(Xa) > 1000, (Xa.shape, Xb.shape)
        assert np.array_equal(Xa, Xb, equal_nan=True) and np.array_equal(ya, yb)
        print(f"folds {folds} frac {frac} {dt.__name__}: {Xa.shape} identical, {np.isnan(Xa).mean():.4f} NaN")
        del Xa, Xb
        gc.collect()
    print("check ok: solution_p16.load == train_matcher.load")


def check_order():
    """solution_p16.files (sorted glob of the channel) == train_matcher's file order (countries, then F.xfiles)."""
    import features as F
    import train_matcher as TM
    want = [p.name for c in TM.countries() for p in F.xfiles("train", c)]
    got = [p.name for p in S.files(WORK)]
    assert got == want, (got, want)
    print(f"file order ok ({len(got)} files)")


def tiny(frac: float = 0.01):
    """1% S1-hash slice (seed 7, independent of the job's val sample) of every train X file, same names/schema."""
    d = TINY / "p4/X"
    d.mkdir(parents=True, exist_ok=True)
    for p in S.files(WORK):
        t = pq.read_table(p)
        q1 = t.column("q1").to_numpy().astype(np.uint64)
        t = t.filter((q1 * np.uint64(2654435761) + np.uint64(7)) % np.uint64(1000) < frac * 1000)
        pq.write_table(t, d / p.name)
        print(f"{p.name}: {t.num_rows:,} rows, peak {peak():.2f} GB", flush=True)
        del t
        gc.collect()


def _hp_json(h: dict) -> dict:
    """Exactly what sm_launch.build_request sends (every value JSON-encoded)."""
    return {k: json.dumps(v) for k, v in {"sagemaker_program": "sm_train.py", "sagemaker_submit_directory": "x", **h}.items()}


def rehearse():
    """sm_train.py -> solution.py (this file's job) on the 1% slice with SageMaker's env/hp contract and the real S3
    writer (prefix amc26-p16/rehearse/); then the booster is loaded and scored through model.predict."""
    import lightgbm as lgb
    import model as M
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="p16-", dir=OUT))
    shutil.copy(ROOT / "sm_train.py", tmp / "sm_train.py")
    shutil.copy(HERE / "solution_p16.py", tmp / "solution.py")
    (tmp / "cfg").mkdir()
    s3 = f"s3://{BUCKET}/{PREFIX}/rehearse/"
    (tmp / "cfg/hyperparameters.json").write_text(json.dumps(_hp_json(hp(s3=s3, ckpt_every=20))))
    env = dict(os.environ, SM_CHANNEL_TRAIN=str(TINY), SM_MODEL_DIR=str(tmp / "model"), SM_OUTPUT_DATA_DIR=str(tmp / "o"),
               SM_INPUT_CONFIG_DIR=str(tmp / "cfg"))
    subprocess.run([sys.executable, str(tmp / "sm_train.py")], cwd=tmp, env=env, check=True)
    res = json.loads((tmp / "model/result.json").read_text())
    m = lgb.Booster(model_file=str(tmp / "model/m2_p16.txt"))
    X = pq.read_table(TINY / "p4/X/train_US_sx_010.parquet").to_pandas()
    p = M.predict(m, X)
    assert len(p) == len(X) and np.isfinite(p).all() and m.feature_name() == M.feats(X), "booster unusable"
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(X.is_true, p)
    keys = [o["Key"] for o in _boto("s3").list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/rehearse/").get("Contents", [])]
    print("s3:", keys)
    assert {f"{PREFIX}/rehearse/m2_p16.txt", f"{PREFIX}/rehearse/status.json"} <= set(keys)
    st = json.loads(_boto("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}/rehearse/status.json")["Body"].read())
    assert st["state"] == "done", st
    for k in keys:
        _boto("s3").delete_object(Bucket=BUCKET, Key=k)
    shutil.rmtree(tmp)
    print(f"rehearse ok: {res}, auc on a slice file {auc:.4f}, {time.time() - t0:.0f}s")
    return res | {"auc_slice": auc, "runtime_s": round(time.time() - t0)}


# ---------------------------------------------------------------- SageMaker

REQS = HERE / "requirements_sm.txt"


def _pack(solution=None):
    """sm_launch.pack_source replacement: only sm_train.py, the job (as solution.py) and OUR pinned requirements."""
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p16.py", "requirements.txt": REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


def launch(dry: bool = False):
    import sm_launch
    sm_launch.pack_source = _pack
    name = f"p16-lgbm-{time.strftime('%m%d-%H%M')}"
    h = hp(s3=f"s3://{BUCKET}/{PREFIX}/results/{name}/", deadline_s=MAX_RUN - 1200, ckpt_every=1000)
    argv = ["--profile", "harsh", "--region", REGION, "--prefix", PREFIX, "--name", name, "--data-s3", DATA_S3,
            "--solution", str(HERE / "solution_p16.py"), "--instance", INSTANCE, "--no-spot", "--volume", "30",
            "--max-run", str(MAX_RUN), "--hp", json.dumps(h)] + (["--dry-run"] if dry else [])
    sm_launch.main(argv)
    if not dry:
        JOB_FILE.write_text(name)
    return name


def _job(job=None):
    return job or JOB_FILE.read_text().strip()


def status(job=None):
    job = _job(job)
    d = _boto("sagemaker").describe_training_job(TrainingJobName=job)
    try:
        st = json.loads(_boto("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}/results/{job}/status.json")["Body"].read())
    except Exception:
        st = None
    tr = [(t["Status"], t["StartTime"].strftime("%H:%M:%S")) for t in d.get("SecondaryStatusTransitions", [])]
    print(json.dumps({"job": job, "status": d["TrainingJobStatus"], "secondary": d["SecondaryStatus"],
                      "failure": d.get("FailureReason"), "billable_s": d.get("BillableTimeInSeconds"),
                      "transitions": tr, "s3_status": st}, default=str, indent=1))
    return d, st


def fetch(job=None, ckpt=False):
    """The final booster -> data/work/p16/m2_p16.txt, or (ckpt) the latest checkpoint -> p16/ckpt/m2_p16_<iter>.txt."""
    job = _job(job)
    OUT.mkdir(parents=True, exist_ok=True)
    if ckpt:
        _, st = status(job)
        dst = OUT / "ckpt" / f"m2_p16_{st['best_iter']}.txt"
        dst.parent.mkdir(exist_ok=True)
        _boto("s3").download_file(BUCKET, f"{PREFIX}/results/{job}/ckpt/m2_p16_ckpt.txt", str(dst))
    else:
        dst = BOOSTER
        _boto("s3").download_file(BUCKET, f"{PREFIX}/results/{job}/m2_p16.txt", str(dst))
    print(f"-> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
    return dst


# ---------------------------------------------------------------- evaluation (#7 yardstick)

def _ce_score(ta, tb) -> np.ndarray:
    """#7's CE (p11/ce_s2P.pt) on new band pairs, in its own process under the GPU lock (as run.rerank does)."""
    with tempfile.TemporaryDirectory(dir=WORK) as d:
        pd.DataFrame({"ta": ta, "tb": tb}).to_parquet(f"{d}/in.parquet", index=False)
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import rerank; "
                f"rerank.score_file({d + '/in.parquet'!r}, {d + '/out.parquet'!r}, {str(CE7)!r})")
        subprocess.run(["flock", "/tmp/amc-gpu.lock", sys.executable, "-c", code], check=True)
        return pd.read_parquet(f"{d}/out.parquet").r.to_numpy(np.float32)


def country_parts(booster: Path, c: str, split: str = "holdout") -> dict:
    """The pruned kept pairs of one country (run.model_parts' holdout path): q1, qx, M2 p (M.split_probs, cached in
    data/work/p9 by booster hash), #7 CE r on the band (cache p9/r_<split>_<c>_<ce hash>; new pairs scored)."""
    import model as M
    import rerank as RR
    M.BOOSTER, M.TAU, M._M = booster, TAU, None
    M.CACHE = WORK / "p9" if booster.parent != OUT / "ckpt" else OUT / "ckpt"     # previews stay out of p9
    d = M.split_probs(split, c)
    q1, qx, p = d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32)
    del d
    lo, hi = RR.CFG["band"]
    b = np.flatnonzero((p > lo) & (p < hi))
    k = (q1[b].astype(np.int64) << 32) | qx[b].astype(np.int64)
    cache = WORK / "p9" / f"r_{split}_{c}_{M.tag(CE7)}.parquet"
    old = pd.read_parquet(cache)
    r = pd.Series(old.r.to_numpy(np.float32), index=old.k.to_numpy(np.int64)).reindex(k).to_numpy(np.float32).copy()
    todo = np.isnan(r)
    print(f"{booster.name} {split} {c}: {len(p):,} kept pairs, {len(b):,} band, {int(todo.sum()):,} new band pairs "
          f"for the CE", flush=True)
    if todo.any():
        xs = "train" if split == "holdout" else split
        ta, tb = RR.texts(xs, c, q1[b[todo]], qx[b[todo]])
        r[todo] = _ce_score(ta, tb)
        pd.DataFrame({"k": np.concatenate([old.k.to_numpy(np.int64), k[todo]]),
                      "r": np.concatenate([old.r.to_numpy(np.float32), r[todo]])}).to_parquet(cache, index=False)
    full = np.full(len(p), np.nan, np.float32)
    full[b] = r
    return {"q1": q1, "qx": qx, "p": p, "r": full, "new_band": int(todo.sum()), "band": len(b)}


def holdout_frame(c: str) -> dict:
    """Fold-0 S1 of country c: pos[q1] = row in the scored list (-1 = competes, not scored), n_true per scored S1,
    half (P5 tune 0 / report 1), the true pair keys (q1 << 32 | qx) -- all from gt_pairs and the id orders."""
    import features as F
    import run
    id1, idx = F.ids("train", c)
    s1, _ = run.holdout_s1()
    ids = pa.array(s1.entity_id[s1.country == c].to_numpy(object), pa.string())
    pos = np.full(len(id1), -1)
    pos[pc.index_in(ids, value_set=id1).to_numpy()] = np.arange(len(ids))
    gt = pq.read_table(WORK / "gt_pairs.parquet", columns=["s1", "sx"])
    gt = gt.filter(pc.is_in(gt.column("s1"), value_set=ids))
    g1 = pc.index_in(gt.column("s1"), value_set=id1).to_numpy(zero_copy_only=False)
    gx = pc.index_in(gt.column("sx"), value_set=idx).to_numpy(zero_copy_only=False)
    ntrue = np.bincount(pos[g1.astype(np.int64)], minlength=len(ids))
    ok = ~np.isnan(gx.astype(float))
    tkeys = (g1[ok].astype(np.int64) << 32) | gx[ok].astype(np.int64)
    return {"pos": pos, "ntrue": ntrue, "half": run.half(ids.to_pylist()), "tkeys": np.unique(tkeys), "n": len(ids)}


def decide(d: dict, h: dict, p: np.ndarray) -> tuple:
    """The t-independent part of the decision: each SX's argmax-p S1 -> (row of the scored S1, is true, p)."""
    import model as M
    am = M.argmax_sx(d["qx"], p)
    q1, qx = d["q1"][am].astype(np.int64), d["qx"][am].astype(np.int64)
    ps = h["pos"][q1]
    m = ps >= 0
    return ps[m], np.isin((q1[m] << 32) | qx[m], h["tkeys"]), p[am][m]


def per_s1(dec: tuple, h: dict, t: float) -> np.ndarray:
    """F0.5 of every scored S1 (singletons included) under the decision: argmax S1 per SX, p >= t."""
    import train_matcher as TM
    ps, y, pa = dec
    sel = pa >= t
    return TM.f05_per_s1(np.bincount(ps[sel & y], minlength=h["n"]), np.bincount(ps[sel], minlength=h["n"]), h["ntrue"])


def evaluate(boosters=None) -> dict:
    """#7 yardstick: full-corpus holdout (every train S1 / S2 / S3 competes), #6 pruner (tau 0.03), #7 CE
    (p11/ce_s2P.pt) on the 0.02-0.98 band. Per booster: blend w refit (P4b logistic stack) on the half-0 band pairs,
    t swept 0.50-0.95 on half 0; F0.5 reported on half 1 (P5's report half) and paired per-S1 vs the control
    (m2_big, the same procedure). Also every booster at #7's shipped (w, t) as a reproduction check."""
    import rerank as RR
    boosters = [WORK / "p4/m2_big.txt"] + [Path(b) for b in (boosters or [BOOSTER]) if b != "ctl"]   # ctl: control only
    H = {c: holdout_frame(c) for c in ("India", "US")}
    res, f_half1, f_7 = {}, {}, {}
    for b in boosters:
        parts = {c: country_parts(b, c) for c in ("India", "US")}
        P, R, Y = [], [], []                          # half-0 band pairs of the scored S1, for the blend refit
        for c, d in parts.items():
            h = H[c]
            bi = np.flatnonzero(~np.isnan(d["r"]))
            ps = h["pos"][d["q1"][bi]]
            m = ps >= 0
            m[m] &= h["half"][ps[m]] == 0
            bi = bi[m]
            P.append(d["p"][bi]), R.append(d["r"][bi])
            Y.append(np.isin((d["q1"][bi].astype(np.int64) << 32) | d["qx"][bi].astype(np.int64), h["tkeys"]))
        w = RR.fit_blend(*(np.concatenate(z) for z in (P, R, Y)))
        del P, R, Y

        def sweep(wt, ts):                          # wt None = M2 alone (no CE)
            dec = {c: decide(d, H[c], d["p"] if wt is None else RR.blend(d["p"], d["r"], wt, *RR.CFG["band"]))
                   for c, d in parts.items()}
            return {t: np.concatenate([per_s1(dec[c], H[c], t) for c in parts]) for t in ts}
        hv = np.concatenate([H[c]["half"] for c in parts])
        grp = np.concatenate([[c] * H[c]["n"] for c in parts])
        ts = np.round(np.arange(0.50, 0.951, 0.01), 2)
        sw = sweep(w, ts)
        t0 = max(ts, key=lambda t: sw[t][hv == 0].mean())
        f, f7 = sw[t0], sweep(W7, [T7])[T7]
        nc = sweep(None, ts)
        tn = max(ts, key=lambda t: nc[t][hv == 0].mean())
        f_half1[b.name], f_7[b.name] = f[hv == 1], f7
        r = {"w_refit": [round(x, 4) for x in w], "t_half0": float(t0), "f05_half1": f[hv == 1].mean(),
             "f05_half0": f[hv == 0].mean(), "f05_all": f.mean(),
             **{f"f05_half1_{c}": f[(hv == 1) & (grp == c)].mean() for c in parts},
             "f05_noce_half1": nc[tn][hv == 1].mean(), "t_noce_half0": float(tn),
             "f05_w7t7_all": f7.mean(), "f05_w7t7_half1": f7[hv == 1].mean(), "f05_w7t7_half0": f7[hv == 0].mean(),
             "kept_pairs": int(sum(len(d["p"]) for d in parts.values())),
             "band_pairs": int(sum(d["band"] for d in parts.values())),
             "new_band_pairs": int(sum(d["new_band"] for d in parts.values()))}
        ctl = boosters[0].name
        if b.name != ctl:
            dlt = f_half1[b.name] - f_half1[ctl]
            d7 = f_7[b.name] - f_7[ctl]                 # #7's shipped (w, t): nothing tuned, so every S1 counts
            r |= {"gain_vs_m2_big": dlt.mean(), "gain_se": dlt.std(ddof=1) / np.sqrt(len(dlt)),
                  "s1_up": int((dlt > 0).sum()), "s1_down": int((dlt < 0).sum()),
                  "gain_w7t7_all": d7.mean(), "gain_w7t7_all_se": d7.std(ddof=1) / np.sqrt(len(d7))}
        res[b.name] = r
        print(b.name, {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        del parts
        gc.collect()
    return res


def test_scores(booster: Path = BOOSTER):
    """The adopted booster on the test split with the #6 pruner (M.split_probs -> p9 cache, the file run.py reads)
    -> data/work/p16/prob_test_<c>.parquet (s1, sx, q1, qx, p; p = M2 p, not blended), all test labels incl. France.
    India / US band pairs missing from #7's CE cache are CE-scored so run.py's rerank finds them cached."""
    import features as F
    import rerank as RR
    labels = sorted(pq.read_table(WORK / "test_source1.parquet", columns=["country"]).column(0).unique().to_pylist())
    out = {}
    for c in labels:
        t0 = time.time()
        if c in RR.CFG["countries"]:
            d = country_parts(booster, c, "test")
        else:
            import model as M
            M.BOOSTER, M.TAU, M._M = booster, TAU, None
            x = M.split_probs("test", c)
            d = {"q1": x.q1.to_numpy(), "qx": x.qx.to_numpy(), "p": x.p.to_numpy(np.float32)}
            del x
        id1, idx = F.ids("test", c)
        pq.write_table(pa.table({"s1": id1.take(pa.array(d["q1"])), "sx": idx.take(pa.array(d["qx"])),
                                 "q1": d["q1"], "qx": d["qx"], "p": d["p"]}), OUT / f"prob_test_{c}.parquet")
        out[c] = {"pairs": len(d["p"]), "s": round(time.time() - t0)}
        print(c, out[c], f"peak {peak():.2f} GB", flush=True)
        del d
        gc.collect()
    return out


def profile(q1: np.ndarray, qx: np.ndarray, p: np.ndarray, n_s1: int, ts=(0.69, 0.78, 0.88, 0.95)) -> dict:
    """Label-free output profile (P14's): share of S1 with no match, matches per S1 (P14 mps), at each t."""
    import model as M
    am = M.argmax_sx(qx, p)
    out = {}
    for t in ts:
        n = np.bincount(q1[am & (p >= t)], minlength=n_s1)
        out |= {f"empty_{t}": float((n == 0).mean()), f"mps_{t}": float(n.mean())}
    return out


def france(boosters=None) -> dict:
    """France test profile (pruned pairs, M2 p: France is not reranked) for m2_big and the new booster."""
    import features as F
    import model as M
    n = len(F.ids("test", "France")[0])
    res = {}
    for b in [WORK / "p4/m2_big.txt"] + [Path(x) for x in (boosters or [BOOSTER])]:
        M.BOOSTER, M.TAU, M._M, M.CACHE = b, TAU, None, WORK / "p9"
        d = M.split_probs("test", "France")
        res[b.name] = profile(d.q1.to_numpy(), d.qx.to_numpy(), d.p.to_numpy(np.float32), n)
        print(b.name, {k: round(v, 4) for k, v in res[b.name].items()}, flush=True)
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        check_order(), check()
    elif cmd == "tiny":
        tiny()
    elif cmd == "rehearse":
        print(rehearse())
    elif cmd == "launch":
        launch(dry=bool(args and args[0] == "dry"))
    elif cmd == "status":
        status(*args)
    elif cmd == "fetch":
        fetch(None, ckpt=True) if args == ["ckpt"] else fetch(*args)
    elif cmd == "eval":
        print(json.dumps(evaluate(args or None), default=float))
    elif cmd == "france":
        print(json.dumps(france(args or None)))
    elif cmd == "test":
        print(test_scores(*(Path(a) for a in args)))
    else:
        sys.exit(__doc__)
