"""P22-ensemble, laptop side. The SageMaker job is solution_p22.py (self-contained, shipped as solution.py).

    p22.py tiny                  # 1% slices -> data/work/p22/sm_tiny/{train,test}/ (rehearsal channels)
    p22.py rehearse [xgb|cat]    # sm_train.py -> solution.py on the slice, SageMaker env + hp encoding, S3 writes
    p22.py upload                # test channel -> s3://.../amc26-p22/data/test/ (pruned keys + test X files)
    p22.py launch xgb|cat [dry]  # ONE job: ml.m7i.4xlarge on-demand, train = amc26-p16/data/v1/, MaxRuntime 3 h
    p22.py status [xgb|cat]      # status.json the job writes to S3 + DescribeTrainingJob
    p22.py fetch xgb|cat         # booster + pred/*.parquet -> data/work/p22/<model>/
    p22.py evstage               # ev.py's inputs -> s3://.../amc26-p22/data/ev/ (the eval job's channel)
    p22.py evlaunch RUNS [dry]   # ev.py on ml.m5.4xlarge: repro + evaluate(models, name) per run -> eval_<name>.json
    p22.py evfetch [job]         # -> data/work/p22/ev/<job>/
"""
import io, json, os, resource, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("AWS_PROFILE", "harsh")

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "code/business_entity_resolution/src"
sys.path[:0] = [str(HERE), str(ROOT)]
WORK = ROOT / "data/work"
OUT = WORK / "p22"
TINY = OUT / "sm_tiny"
BUCKET, PREFIX, REGION = "sagemaker-114872709824-ap-south-1", "amc26-p22", "ap-south-1"
TRAIN_S3 = f"s3://{BUCKET}/amc26-p16/data/v1/"              # P16's copy of data/work/p4/X/train_*
TEST_S3 = f"s3://{BUCKET}/{PREFIX}/data/test/"
INSTANCE, MAX_RUN = "ml.m7i.4xlarge", 10800         # 16 vCPU / 64 GB, $1.018/h on-demand -> <= $3.05 per job
M2 = "d2b6569f_6a23c2ec_0.03"                       # m2_big booster hash, pruner hash, tau: #6's pruned rows
VENV = Path("/tmp/p22v/bin/python")                 # scratch venv: xgboost-cpu, catboost (not in .venv)
REQS = HERE / "requirements_sm.txt"
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _boto(svc):
    import boto3
    return boto3.Session(profile_name="harsh", region_name=REGION).client(svc)


def keys(split: str, c: str) -> pa.Table:
    return pq.read_table(WORK / "p9" / f"m2_{split}_{c}_{M2}.parquet", columns=["q1", "qx"])


def _k(t) -> np.ndarray:
    return (t.column("q1").to_numpy().astype(np.int64) << 32) | t.column("qx").to_numpy().astype(np.int64)


def tiny(frac: float = 0.01):
    """1% S1-hash slice of one train and one test X file per country, and the pruned keys that fall in them."""
    for d in ("train/p4/X", "test/p4/X", "test/keys"):
        (TINY / d).mkdir(parents=True, exist_ok=True)
    for split, cs in (("train", ("India", "US")), ("test", ("France", "India", "US"))):
        for c in cs:
            name = f"{split}_{c}_sx_000.parquet"
            t = pq.read_table(WORK / "p4/X" / name)
            q1 = t.column("q1").to_numpy().astype(np.uint64)
            t = t.filter((q1 * np.uint64(2654435761) + np.uint64(7)) % np.uint64(1000) < frac * 1000)
            ch = "train" if split == "train" else "test"
            pq.write_table(t, TINY / ch / "p4/X" / name)
            ks = keys("holdout" if split == "train" else "test", c)
            ks = ks.filter(pc.is_in(pa.array(_k(ks)), value_set=pa.array(_k(t))))
            pq.write_table(ks, TINY / "test/keys" / f"{'holdout' if split == 'train' else 'test'}_{c}.parquet")
            print(f"{name}: {t.num_rows:,} rows, {ks.num_rows:,} keys", flush=True)


def hp(model: str, **kw) -> dict:
    return {"model": model, "folds": [2, 3, 4, 5], "rounds": 8000, "es": 100, "val_frac": 0.25} | kw


def _hp_json(h: dict) -> dict:
    """Exactly what sm_launch.build_request sends (every value JSON-encoded)."""
    return {k: json.dumps(v) for k, v in {"sagemaker_program": "sm_train.py", "sagemaker_submit_directory": "x", **h}.items()}


def rehearse(model: str = "xgb"):
    """sm_train.py -> solution.py on the 1% slices with SageMaker's env/hp contract and the real S3 writer (prefix
    amc26-p22/rehearse/), in the scratch venv; then the outputs are checked and the S3 objects deleted."""
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="p22-", dir=OUT))
    shutil.copy(ROOT / "sm_train.py", tmp / "sm_train.py")
    shutil.copy(HERE / "solution_p22.py", tmp / "solution.py")
    (tmp / "cfg").mkdir()
    s3 = f"s3://{BUCKET}/{PREFIX}/rehearse-{model}/"
    (tmp / "cfg/hyperparameters.json").write_text(json.dumps(_hp_json(hp(model, s3=s3, ckpt_every=20, rounds=300))))
    env = dict(os.environ, SM_CHANNEL_TRAIN=str(TINY / "train"), SM_CHANNEL_TEST=str(TINY / "test"),
               SM_MODEL_DIR=str(tmp / "model"), SM_OUTPUT_DATA_DIR=str(tmp / "o"), SM_INPUT_CONFIG_DIR=str(tmp / "cfg"),
               OMP_NUM_THREADS="4")
    subprocess.run([str(VENV), str(tmp / "sm_train.py")], cwd=tmp, env=env, check=True)
    res = json.loads((tmp / "model/result.json").read_text())
    for f in (TINY / "test/keys").glob("*.parquet"):
        got = pq.read_table(tmp / "model/pred" / f.name)
        assert got.num_rows == pq.read_metadata(f).num_rows and np.isfinite(got.column("p").to_numpy()).all(), f.name
    keys_ = [o["Key"] for o in _boto("s3").list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/rehearse-{model}/").get("Contents", [])]
    print("s3:", keys_)
    st = json.loads(_boto("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}/rehearse-{model}/status.json")["Body"].read())
    assert st["state"] == "done", st
    for k in keys_:
        _boto("s3").delete_object(Bucket=BUCKET, Key=k)
    shutil.rmtree(tmp)
    print(f"rehearse {model} ok: {res}, {time.time() - t0:.0f}s")
    return res | {"runtime_s": round(time.time() - t0)}


def upload():
    """Test channel: keys/{holdout,test}_<c>.parquet (#6's pruned pairs) + every test X file."""
    s3 = _boto("s3")
    stage = OUT / "keys"
    stage.mkdir(parents=True, exist_ok=True)
    for split, cs in (("holdout", ("India", "US")), ("test", ("France", "India", "US"))):
        for c in cs:
            pq.write_table(keys(split, c), stage / f"{split}_{c}.parquet")
            s3.upload_file(str(stage / f"{split}_{c}.parquet"), BUCKET, f"{PREFIX}/data/test/keys/{split}_{c}.parquet")
    t0 = time.time()
    for p in sorted((WORK / "p4/X").glob("test_*_sx_*.parquet")):
        s3.upload_file(str(p), BUCKET, f"{PREFIX}/data/test/p4/X/{p.name}")
        print(f"{p.name} {time.time() - t0:.0f}s", flush=True)


def _pack(solution=None):
    """sm_launch.pack_source replacement: only sm_train.py, the job (as solution.py) and OUR pinned requirements."""
    files = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p22.py", "requirements.txt": REQS}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(files):
            tar.add(files[arc], arcname=arc)
    return buf.getvalue(), sorted(files)


def launch(model: str, dry: bool = False):
    import sm_launch
    sm_launch.pack_source = _pack
    name = f"p22-{model}-{time.strftime('%m%d-%H%M')}"
    h = hp(model, s3=f"s3://{BUCKET}/{PREFIX}/results/{name}/", deadline_s=MAX_RUN - 2400, ckpt_every=500)
    argv = ["--profile", "harsh", "--region", REGION, "--prefix", PREFIX, "--name", name, "--data-s3", TRAIN_S3,
            "--test-s3", TEST_S3, "--solution", str(HERE / "solution_p22.py"), "--instance", INSTANCE, "--no-spot",
            "--volume", "30", "--max-run", str(MAX_RUN), "--hp", json.dumps(h)] + (["--dry-run"] if dry else [])
    sm_launch.main(argv)
    if not dry:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"job_{model}.txt").write_text(name)
    return name


def status(model: str = "xgb"):
    job = (OUT / f"job_{model}.txt").read_text().strip()
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


def fetch(model: str):
    job = (OUT / f"job_{model}.txt").read_text().strip()
    dst = OUT / model
    (dst / "pred").mkdir(parents=True, exist_ok=True)
    s3 = _boto("s3")
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/results/{job}/").get("Contents", []):
        rel = o["Key"].split(f"{job}/", 1)[1]
        if rel.startswith("ckpt/"):
            continue
        s3.download_file(BUCKET, o["Key"], str(dst / rel))
        print(f"-> {dst / rel} ({o['Size'] / 1e6:.1f} MB)")


# ---------------------------------------------------------------- evaluation job (the laptop's memguard pauses ev.py)

EV = OUT / "sm_ev"
EV_S3 = f"s3://{BUCKET}/{PREFIX}/data/ev/"
EV_REQS = HERE / "requirements_ev.txt"
EV_CODE = {"sm_train.py": ROOT / "sm_train.py", "solution.py": HERE / "solution_p22ev.py", "ev.py": HERE / "ev.py",
           "requirements.txt": EV_REQS} | {f: SRC / f for f in ("rerank.py", "stack.py", "model.py", "train_matcher.py")}


def ev_stage():
    """The eval channel (hard links into data/work) -> s3 (aws s3 sync: only changed files move)."""
    names = [f"p9/r_holdout_{c}_{t}.parquet" for c in ("India", "US") for t in ("3bd0b930", "bb0c29f0")] + \
            [f"p9/m2_holdout_{c}_0726cb9a_6a23c2ec_0.03.parquet" for c in ("India", "US")] + \
            ["p17/stack_sup-bge-l21.0_US.txt"] + [f"p22/{f}_{c}.{e}" for c in ("India", "US") for f, e in
                                                   (("frame", "parquet"), ("s1", "npz"))] + \
            [f"p22/{p.name}" for p in OUT.glob("r_extra_*.parquet")]
    for n in names:
        dst = EV / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        os.link(WORK / n, dst)
    subprocess.run(["aws", "s3", "sync", str(EV), EV_S3, "--delete", "--profile", "harsh", "--region", REGION], check=True)


def _pack_ev(solution=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc in sorted(EV_CODE):
            tar.add(EV_CODE[arc], arcname=arc)
    return buf.getvalue(), sorted(EV_CODE)


def ev_launch(runs: list, dry: bool = False):
    import sm_launch
    sm_launch.pack_source = _pack_ev
    name = f"p22-ev-{time.strftime('%m%d-%H%M')}"
    results = {m: f"s3://{BUCKET}/{PREFIX}/results/{(OUT / f'job_{m}.txt').read_text().strip()}/"
               for m in ("xgb", "cat") if any(m in r[0] for r in runs)}
    h = {"runs": runs, "results": results, "s3": f"s3://{BUCKET}/{PREFIX}/results/{name}/"}
    argv = ["--profile", "harsh", "--region", REGION, "--prefix", PREFIX, "--name", name, "--data-s3", EV_S3,
            "--solution", str(HERE / "solution_p22ev.py"), "--instance", "ml.m5.4xlarge", "--no-spot",
            "--volume", "30", "--max-run", "3600", "--hp", json.dumps(h)] + (["--dry-run"] if dry else [])
    sm_launch.main(argv)
    if not dry:
        (OUT / "job_ev.txt").write_text(name)
    return name


def ev_fetch(job=None):
    job = job or (OUT / "job_ev.txt").read_text().strip()
    subprocess.run(["aws", "s3", "sync", f"s3://{BUCKET}/{PREFIX}/results/{job}/", str(OUT / "ev" / job),
                    "--profile", "harsh", "--region", REGION], check=True)
    print(sorted(p.name for p in (OUT / "ev" / job).iterdir()))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "tiny":
        tiny()
    elif cmd == "rehearse":
        print(rehearse(*args))
    elif cmd == "upload":
        upload()
    elif cmd == "launch":
        launch(args[0], dry=args[1:] == ["dry"])
    elif cmd == "status":
        status(*args)
    elif cmd == "fetch":
        fetch(args[0])
    elif cmd == "evstage":
        ev_stage()
    elif cmd == "evlaunch":             # evlaunch '[[["p16"], "dry_p16"]]' [dry]
        ev_launch(json.loads(args[0]), dry=args[1:] == ["dry"])
    elif cmd == "evfetch":
        ev_fetch(*args)
    else:
        sys.exit(__doc__)
